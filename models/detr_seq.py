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


class DecoderEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id, max_position_embeddings, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        self.new_frame_adaptor = None
        self.d_model = d_model
        self.nhead = nhead

        # self.vocab_embedding = nn.Embedding(3002, d_model, padding_idx=3000)
        self.start = 4000
        self.end = 4001
        self.bins = 2000
        self.ranges = 2
        self.embedding = DecoderEmbeddings(self.bins * self.ranges + 4, d_model, self.bins * self.ranges, 501, dropout)
        self.temporal_position_embeddings = nn.Embedding(5, d_model)  # 历史帧
        self.spatio_position_embeddings = nn.Embedding(200, d_model)  # 身份位置编码
        self.object_position_embeddings = nn.Embedding(5, d_model)  # 内部位置编码
        self.query_embeddings = nn.Embedding(1, d_model)  # ref位置编码
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        #        self.token_drop = SpatialDropout(drop=0.5)
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, seq, vocab_embed, frame_index):

        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        #        tgt = self.token_drop(self.embedding(seq), noise_shape=(bs, 501, 1)).permute(1, 0, 2)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed.half())

        # prepare input for decoder
        device = memory.device
        # 对seq处理, 添加位置编码
        # 推理的时候seq为4,n,5+L(当前输出)
        # print(frame_index)
        list_id = random.sample(range(0, 199), 199)
        N = []
        input_seqs = torch.tensor([])
        mask_seqs = torch.tensor([])
        pos_seqs = torch.tensor([])
        ref_pts = torch.tensor([])  # L, 4
        if self.training:
            for i in range(frame_index + 1):
                # 去掉id
                seq_frame = seq[i]
                # print(seq)
                # print(seq_frame.shape)
                input_seq = seq_frame[:, :-1]  # N, 5
                device = input_seq.device

                id = seq_frame[:, -1]  # N,1
                # 添加身份编码（同一个id拥有同一个编码）  x y w h label都加一个相同的id_pos
                id_pos = self.spatio_position_embeddings.weight[list_id[:len(id)]].clone().unsqueeze(1)  # N, D
                id_pos = id_pos.repeat(1, 5, 1)  # N, 5, D

                # 添加时序编码，每一帧的所有物体都加一个相同的time_pos
                time_pos = self.temporal_position_embeddings.weight.clone().unsqueeze(1)  # f, 1, D
                time_pos_frame = time_pos[i]  # 第i帧的时序位置, 1, 1, D
                time_pos_frame = time_pos_frame.repeat(len(input_seq), 5, 1)  # N, 5, D

                # 添加内部位置编码，x y w h label
                object_pos = self.object_position_embeddings.weight.clone().unsqueeze(0)  # 1, 5, D
                object_pos = object_pos.repeat(len(input_seq), 1, 1)  # N, 5, D

                # 添加start和end, 编写位置编码（总），start和end无位置编码，用0取代
                # 预测当前帧开始和最后有end
                # 记录每一帧物体数量
                N.append(len(seq_frame) * 5)
                input_seq = input_seq.flatten()
                pos_seq = (id_pos + time_pos_frame + object_pos).flatten(0, 1)  # N,5,D
                if i == frame_index:
                    # 1+N*5
                    input_seq = torch.cat([torch.ones(1).to(input_seq) * self.start, input_seq], dim=-1)
                    # 1+N*5, D
                    pos_seq = torch.cat([torch.zeros((1, self.d_model)).to(pos_seq), pos_seq], dim=0)
                    N[-1] += 1
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
                        mask_seq = torch.cat([mask_seq, mask_seq_n], dim=-1)
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
            mask_seqs = mask_seqs.to(pos_seqs)
            N = len(seq_frame) * 5

            # decoder
            input_seqs = input_seqs.unsqueeze(-1)
            input_seqs = input_seqs.repeat(1, bs)  # L, bs
            tgt = self.embedding(input_seqs)
            pos_seqs = pos_seqs.unsqueeze(1)  # N, 1, D
            # print(input_seqs.shape)
            pos_seqs = pos_seqs.repeat(1, bs, 1)  # L, bs, D
            if self.training:
                hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos=pos_seqs[:len(tgt)],
                                  tgt_mask=mask_seqs.to(tgt.device))
                return hs.transpose(1, 2)



def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_deforamble_transformer_seq(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
