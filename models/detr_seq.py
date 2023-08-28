# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import copy
from typing import Optional, List
import math
import random

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from models.structures import Boxes, matched_boxlist_iou, pairwise_iou

from util.misc import inverse_sigmoid
from util.box_ops import box_cxcywh_to_xyxy
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, decoder_self_cross=True, sigmoid_attn=False,
                 extra_track_attn=False, temporal=0):
        super().__init__()

        self.new_frame_adaptor = None
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        # self.vocab_embedding = nn.Embedding(3002, d_model, padding_idx=3000)
        self.temporal_position_embeddings = nn.Embedding(temporal, d_model)  # 历史帧
        self.spatio_position_embeddings = nn.Embedding(200, d_model)  # 身份位置编码
        self.object_position_embeddings = nn.Embedding(5, d_model)  # 内部位置编码
        self.query_embeddings = nn.Embedding(1, d_model)  # ref位置编码
        self.start = 4001
        self.end = 4000
        self.bins = 2000

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points,
                                                          sigmoid_attn=sigmoid_attn)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, decoder_self_cross,
                                                          sigmoid_attn=sigmoid_attn, extra_track_attn=extra_track_attn)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 4)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, seq=None, ref_pts=None, frame_index=None):
        assert self.two_stage or seq is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)
        # prepare input for decoder
        bs, _, c = memory.shape
        # 对seq处理, 添加位置编码
        # print(frame_index)
        id_num = max(len(s) for s in seq)
        # print(f"id_num={id_num}")
        list_id = random.sample(range(0, 199), id_num)
        N = []
        input_seqs = torch.tensor([])
        mask_seqs = torch.tensor([])
        pos_seqs = torch.tensor([])
        ref_pts = torch.tensor([]) # L, 4
        for i in range(frame_index + 1):
            # 去掉id
            seq_frame = seq[i]
            # print(seq_frame.shape)
            input_seq = seq_frame[:, :-1]  # N, 5
            device = input_seq.device

            # 编写ref_pts, 上一帧有的按照上一帧的，上一帧没有的用可学习
            if i == 0:
                # 初始化
                #print(self.query_embeddings.weight[:].shape)
                ref_pts_now = self.reference_points(self.query_embeddings.weight[:].repeat(len(input_seq), 1)).sigmoid() # N, 4
                #print(ref_pts_now.shape)
                ref_pts_now = ref_pts_now.squeeze(1).repeat(1,5,1).flatten(0, 1) # N*5, 4
                #print(ref_pts_now.shape)
            else:
                seq_frame_last = seq[i-1]
                seq_frame_last = seq_frame_last[:, :-2] / (self.bins - 1)# N, 4
                n1 = len(seq_frame_last)
                n2 = len(seq_frame) # n2一定>=n1
                ref_pts_now = seq_frame_last.unsqueeze(1).repeat(1, 5, 1).flatten(0,1) # n1*5,4
                if n2 > n1:
                    ref_pts_new = self.reference_points(self.query_embeddings.weight[:].repeat((n2 - n1), 1)).sigmoid() # N, 4
                    ref_pts_new = ref_pts_new.unsqueeze(1).repeat(1, 5, 1).flatten(0,1)  # n2-n1 * 5, 4
                    ref_pts_now = torch.cat([ref_pts_now, ref_pts_new], dim=0) # N*5, 4
            ref_pts = torch.cat([ref_pts.to(device), ref_pts_now], dim=0)
            if i == frame_index:
                ref_pts_now = self.reference_points(self.query_embeddings.weight[:].sigmoid()) # 1, 4
                ref_pts = torch.cat([ref_pts, ref_pts_now], dim=0) # N*5+1, 4
            #print(f"ref_pts={ref_pts}")

            id = seq_frame[:, -1]  # N,1
            # 添加身份编码（同一个id拥有同一个编码）  x y w h label都加一个相同的id_pos
            # print(list_id)
            # print(len(id))
            # print(list_id[:len(id)])
            id_pos = self.spatio_position_embeddings.weight[list_id[:len(id)]].unsqueeze(1)  # N, D
            # print(id_pos.shape)
            id_pos = id_pos.repeat(1, 5, 1)  # N, 5, D

            # 添加时序编码，每一帧的所有物体都加一个相同的time_pos
            time_pos = self.temporal_position_embeddings.weight.unsqueeze(1)  # f, 1, D
            time_pos_frame = time_pos[i]  # 第i帧的时序位置, 1, 1, D
            time_pos_frame = time_pos_frame.repeat(len(input_seq), 5, 1)  # N, 5, D

            # 添加内部位置编码，x y w h label
            object_pos = self.object_position_embeddings.weight.unsqueeze(0)  # 1, 5, D
            object_pos = object_pos.repeat(len(input_seq), 1, 1)  # N, 5, D

            # 添加start和end, 编写位置编码（总），start和end无位置编码，用0取代
            # 预测当前帧开始和最后有end
            # print(id_pos.shape)
            # print(time_pos_frame.shape)
            # print(object_pos.shape)
            # 记录每一帧物体数量
            N.append(len(seq_frame) * 5)
            input_seq = input_seq.flatten()
            pos_seq = (id_pos + time_pos_frame + object_pos).flatten(0, 1)  # N,5,D
            if i == frame_index:
                # 1+N*5
                input_seq = torch.cat([torch.ones(1).to(input_seq) * self.start, input_seq], dim=-1)
                # 1+N*5, D
                # print((torch.zeros((1)).to(pos_seq) * self.start).shape)
                # print(pos_seq.flatten(0, 1).shape)
                pos_seq = torch.cat([(torch.zeros((1, self.d_model)).to(pos_seq) * self.start), pos_seq], dim=0)
                N[-1] += 1
            # print(f"input_seqs.shape={input_seqs.shape}")
            # print(f"input_seq.shape={input_seq.shape}")
            input_seqs = torch.cat([input_seqs.to(input_seq), input_seq], dim=-1)  # L, 1
            pos_seqs = torch.cat([pos_seqs.to(pos_seq), pos_seq], dim=0)  # L, 1, D

            # 编写每一帧的mask：暂定均为下三角
            # 根据每一帧的物体数量编写（不含start和end）
            # mask_seq = generate_square_subsequent_mask(len(input_seqs))
            mask_seq = torch.tensor([]).to(pos_seqs)
            if i == 0:
                mask_seq = generate_square_subsequent_mask(N[0])
                mask_seqs = torch.cat([mask_seqs, mask_seq], dim=-1)
            else:
                for n in N:
                    n_now = N[-1]
                    # print(n_now)
                    # print(N)
                    mask_seq_n = torch.full((n_now, n), float('-inf'), dtype=torch.float).to(pos_seqs)
                    if n >= n_now:
                        mask_seq_n[:, :n_now] = generate_square_subsequent_mask(n_now)
                    else:
                        # print(n)
                        mask_seq_n[:n, :n] = generate_square_subsequent_mask(n)
                        mask_seq_n[n:, :] = 0.
                    # print(mask_seq, mask_seq_n)
                    mask_seq = torch.cat([mask_seq, mask_seq_n], dim=-1)
                    # print(mask_seq.shape)
                n1 = len(mask_seqs)
                n2 = len(mask_seq)
                # print(n1, n2)
                mask_seqs = torch.cat(
                    [mask_seqs.to(pos_seqs), torch.full((n1, n2), float('-inf'), dtype=torch.float).to(pos_seqs)],
                    dim=-1)
                # print(mask_seqs.shape, mask_seq.shape)
                # print(f"mask_seq={mask_seq.shape}")
                # print(f"mask_seqs={mask_seqs.shape}")
                mask_seqs = torch.cat([mask_seqs.to(pos_seqs), mask_seq], dim=0)
                # print(mask_seq)

            # print(mask_seq.shape)
        # print(mask_seqs.shape)
        # print(frame_index)
        # print(input_seqs.shape, pos_seqs.shape, mask_seqs.shape)
        '''
        if ref_pts is None:
           reference_points = self.reference_points(query_embed).sigmoid()
        else:
           reference_points = ref_pts.unsqueeze(0).repeat(bs, 1, 1).sigmoid()
        '''
        # init_reference_out = reference_points

        # decoder
        input_seqs = input_seqs.unsqueeze(-1)
        input_seqs = input_seqs.repeat(1, bs)  # L, bs
        pos_seqs = pos_seqs.unsqueeze(1)  # N, 1, D
        # print(input_seqs.shape)
        pos_seqs = pos_seqs.repeat(1, bs, 1)  # L, bs, D
        mask_seqs = mask_seqs.to(pos_seqs)
        hs = self.decoder(input_seqs, ref_pts, memory,
                          spatial_shapes, level_start_index, valid_ratios, pos_seqs, mask_flatten,
                          mask_seqs, lvl_pos_embed_flatten, len(seq_frame))

        # inter_references_out = inter_references
        # if self.two_stage:
        #    return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        # return hs, init_reference_out, inter_references_out, None, None
        return hs


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, self_cross=True, sigmoid_attn=False, extra_track_attn=False):
        super().__init__()

        self.self_cross = self_cross
        self.num_head = n_heads

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # update track query_embed
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            print('Training with Extra Self Attention in Every Decoder.', flush=True)
            self.update_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

        if self_cross:
            print('Training with Self-Cross Attention.')
        else:
            print('Training with Cross-Self Attention.')

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward_self_attn(self, tgt, query_pos, attn_mask=None):
        if self.extra_track_attn:
            tgt = self._forward_track_attn(tgt, query_pos)

        q = k = self.with_pos_embed(tgt, query_pos)
        if attn_mask is not None:
            # print(q.shape)
            # print(attn_mask.shape)
            tgt2 = self.self_attn(q, k, tgt,
                                  attn_mask=attn_mask)[0]
        else:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0]
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)

    def _forward_track_attn(self, tgt, query_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        if q.shape[1] > 300:
            tgt2 = self.update_attn(q[:, 300:].transpose(0, 1),
                                    k[:, 300:].transpose(0, 1),
                                    tgt[:, 300:].transpose(0, 1))[0].transpose(0, 1)
            tgt = torch.cat([tgt[:, :300], self.norm4(tgt[:, 300:] + self.dropout5(tgt2))], dim=1)
        return tgt

    def _forward_self_cross(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None, lvl_pos_embed_flatten=None):

        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
        # print(f"tgt_after_self={tgt.shape}")
        # cross attention
        if reference_points == None:
            # print(src.shape)
            tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                       self.with_pos_embed(src, lvl_pos_embed_flatten).transpose(0, 1),
                                       src.transpose(0, 1))[0]  # L, bs, D

        else:
            #print(f"tgt.shape={tgt.shape}")
            #print(f"query_pos={query_pos.shape}")
            #print(f"src={src.shape}")
            #print(reference_points.shape)
            tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos).transpose(0, 1),
                                   reference_points,
                                   src, src_spatial_shapes, level_start_index, src_padding_mask).transpose(0, 1)
        # print(tgt2.shape)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    def _forward_cross_self(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None):
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None, attn_mask=None, lvl_pos_embed_flatten=None):
        if self.self_cross:
            return self._forward_self_cross(tgt, query_pos, reference_points, src, src_spatial_shapes,
                                            level_start_index, src_padding_mask, attn_mask, lvl_pos_embed_flatten)
        return self._forward_cross_self(tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                                        src_padding_mask, attn_mask, lvl_pos_embed_flatten)


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.vocab_embedding = None
        self.output_bias = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, mask_seqs=None, lvl_pos_embed_flatten=None, N=None):
        output = tgt # L, bs

        intermediate = []
        # intermediate_coordinate = []
        # print(f"output={output}")
        input_ = self.vocab_embedding(output)
        # print(input_.shape)
        # input_ = torch.rand(output.shape[0], output.shape[1], 256).to(src)
        # share_weight = self.vocab_embedding.weight.T.clone() # D, B
        #print(f"input_shape={input_.shape}")
        #print(reference_points.shape)
        ### 非datb detr
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

                # input_ = self.vocab_embedding(output)  # L, bs, D
            output = layer(input_, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                           src_padding_mask, mask_seqs, lvl_pos_embed_flatten)
            input_ = output  # L, bs, D


            # 生成坐标 xywh, 当前帧共有N个物体
            # 去掉start和end，提取出当前帧的N个物体
            # output_seq = output[1:-1]
            # output_seq = output_seq[-N*5:].squeeze() # N, D

            # 坐标xywh由词表反乘得到
            # output_coordinate = torch.matmul(output_seq, share_weight) # N, B
            # output_coordinate += self.output_bias

            if self.return_intermediate:
                intermediate.append(output[-N * 5 - 1:])
                # intermediate_coordinate.append(output_coordinate)
            # print(output)
            # print(output.shape)
            # print(output_coordinate)

            '''
            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

            '''
        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer_seq(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        decoder_self_cross=not args.decoder_cross_self,
        sigmoid_attn=args.sigmoid_attn,
        extra_track_attn=args.extra_track_attn,
        temporal=5
    )


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask

