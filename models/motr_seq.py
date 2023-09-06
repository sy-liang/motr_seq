# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
DETR model and criterion classes.
"""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List
from timm.models.layers import trunc_normal_
from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .matcher import build_matcher
from .detr_seq import build_deforamble_transformer_seq
from .qim import build as build_query_interaction_layer
from .memory_bank import build_memory_bank
from .deformable_detr import SetCriterion, MLP
from .segmentation import sigmoid_focal_loss

def random_drop_tracks(sequence, frame_res, drop_probability: float, frame_index):
    pred_boxes = frame_res['pred_boxes'] # N, 4
    pred_logits = frame_res['pred_logits'] # N, 1
    sequence_news = []
    # 历史删去，下一帧和未来的把删掉的放到seq后面
    if drop_probability > 0 and len(pred_boxes) > 0:
        keep_idxes = torch.rand_like(pred_logits) > drop_probability
        for n in range(frame_index + 1):
            frame = sequence[n]
            keep_idxes_n = [i for i in keep_idxes if i < len(frame)]
            sequence_new = frame[keep_idxes_n]
            sequence_news.append(sequence_new)
        for n in range(frame_index + 1, len(sequence)):
            # 放到seq后面
            frame = sequence[n]
            full_indices = torch.ones(len(frame))
            full_indices[keep_idxes] = -1
            sequence_keep = frame[keep_idxes]
            sequence_new = torch.cat([sequence_keep, sequence[full_indices != -1]], dim=-1)
            sequence_news.append(sequence_new)
        pred_boxes_new = pred_boxes[keep_idxes]
        pred_logits_new = pred_logits[keep_idxes]
        frame_res['pred_boxes'] = pred_boxes_new
        frame_res['pred_logits'] = pred_logits_new
    else:
        sequence_news = sequence
    return sequence_news, frame_res





class ClipMatcher(SetCriterion):
    def __init__(self, num_classes,
                 matcher,
                 weight_dict,
                 losses,
                 bins,
                 ranges):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.bins = bins
        self.ranges = ranges

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(frame_id, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        # TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)],
                                   dim=0)  # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :,
                               :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
                                         gt_labels_target.flatten(1),
                                         alpha=0.25,
                                         gamma=2,
                                         num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def match_for_single_frame(self, outputs: dict, sequence, last_boxes):
        # sequence不含end n, 6
        end = 4001

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.

        pred_boxes_i = outputs['pred_boxes']  # predicted boxes of i-th image.
        pred_seqs_i = outputs['pred_seqs'].squeeze(0)  # N, B

        obj_idxes = gt_instances_i.obj_ids
        boj_boxes = gt_instances_i.boxes

        outputs_i = {
            'pred_boxes': pred_boxes_i,
            'pred_seqs_i': pred_seqs_i,
        }
        # 计算序列损失, 只计算真实物体
        # 筛选真实物体 sequence:N, 1
        #print(pred_boxes_i.shape)
        #print(pred_seqs_i.shape)
        N, B = pred_seqs_i.shape
        # print(N)
        # print(sequence[:, -1])
        true_id = sequence[:, -1]  # n, 1
        sequence_without_id = sequence[:, :-1]  # n, 5
        sequence_true = sequence_without_id[true_id != -1]
        n = len(sequence_true)  # 真实物体
        # 添加end
        sequence_true = torch.cat([sequence_true.flatten(), torch.ones(1).to(sequence) * end], dim=-1)  # N

        pred_seqs = pred_seqs_i[:-1].reshape(int((N - 1) / 5), 5, B)
        pred_seqs_true = pred_seqs[true_id != -1]
        pred_seqs_true = torch.cat([pred_seqs_true.reshape(n * 5, B), pred_seqs_i[-1].unsqueeze(0)], dim=0)

        weight = torch.ones(int(self.bins * self.ranges + 4)) * 1
        weight[int(self.bins * self.ranges + 1)] = 0.1
        weight[int(self.bins * self.ranges)] = 0.1
        weight.to(sequence)
        focal = torch.nn.CrossEntropyLoss(weight=weight, reduction="none").to(pred_boxes_i)
        # print(pred_seqs_true.shape, sequence_true.shape)
        ce_loss = focal(pred_seqs_true, sequence_true)
        self.losses_dict.update(
            {'frame_{}_{}'.format(self._current_frame_idx, 'loss_ce'): ce_loss.sum() / 5})
        # print(f"varifocal_loss={varifocal_loss}")

        # 计算iou损失，只需要xywh, 也只需要真实值
        i, j = torch.where(true_id[:, None] == obj_idxes)
        target_box = boj_boxes[j]  # 按照seq的顺序
        pred_box = pred_boxes_i[i]
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(pred_box),
            box_ops.box_cxcywh_to_xyxy(target_box)))
        loss_giou_true = loss_giou.sum().unsqueeze(-1)
        # print(f"loss_giou={loss_giou}")

        device = sequence.device
        self.losses_dict.update(
            {'frame_{}_{}'.format(self._current_frame_idx, 'loss_giou'): loss_giou_true.sum()})

        self.num_samples += len(pred_boxes_i)
        self.sample_device = sequence.device

        self._step()
        return outputs_i

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.7, filter_score_thresh=0.6, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # Set the obj_id to -1.
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        prob = out_logits.sigmoid()
        # prob = out_logits[...,:1].sigmoid()
        scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = labels
        track_instances.remove('pred_logits')
        track_instances.remove('pred_boxes')
        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MOTR_seq(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False,
                 bins=None,
                 ranges=None, random_drop=None, fp_ratio=None, criterion=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.random_drop = random_drop
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.bins = bins
        self.ranges = ranges
        self.fp_ratio = fp_ratio


        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.criterion = criterion

        self.start = 4000
        self.end = 4001
        self.label_true = 4002
        self.label_false = 4003
        self.vocab_embed = MLP(hidden_dim, hidden_dim, self.bins * self.ranges + 4, 3)

    def _add_fp_tracks(self, sequence, frame_res, frame_index):
        pred_boxes = frame_res['pred_boxes']  # N, 4
        pred_logits = frame_res['pred_logits']  # N, 1
        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(pred_logits) * self.fp_ratio
        activate_boxes = pred_boxes[torch.bernoulli(fp_prob).bool()]

        if len(activate_boxes) > 0:
            num_fp = len(activate_boxes)
            random_boxes = torch.rand(num_fp, 4)
            add_boxes = box_ops.box_cxcywh_to_xyxy(random_boxes)
            add_boxes = torch.clamp(add_boxes, min=-0.5, max=1.5)
            add_boxes = box_ops.box_xyxy_to_cxcywh(add_boxes)
            seq = (add_boxes + 0.5) * self.bins
            seq = torch.cat([seq, torch.ones((num_fp, 1)) * self.label_false, torch.ones((num_fp, 1)) * -1], dim=-1)
            seq_old = sequence[:frame_index]
            for n in range(frame_index, len(sequence)):
                frame = sequence[n]
                sequence_true_old = frame[:len(sequence[frame])]
                sequence_true_false = frame[len(sequence[frame]):]
                sequence_new = torch.cat([sequence_true_old, seq, sequence_true_false], dim=0) # 添加在同一个位置
                seq_old.append(sequence_new)
            pred_boxes = torch.cat([pred_boxes, add_boxes], dim=0)
            frame_res['pred_boxes'] = pred_boxes
        else:
            seq_old = sequence

        return seq_old

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, output_seqs):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_seqs': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], output_seqs[:-1])]

    def _forward_single_image(self, samples: NestedTensor, sequence, frame_index=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        if self.training:
            #print(self.transformer)
            hs = self.transformer(self.input_proj(src), mask, None, pos[-1], sequence, self.vocab_embed, frame_index)
            out = self.vocab_embed(hs[0])
            out_coord = self.vocab_embed(hs.transpose(1, 2)[-1, :, -1, :]) # n, B
            out_coord = out_coord[:-1].reshape(-1, 5, self.bins * self.ranges + 4)
            out_coord = out_coord[:, :-1, :self.bins * self.ranges]
            out_coord = out_coord.softmax(-1)
            magic_num = (self.ranges - 1) * 0.5
            mul = torch.range(-1 * magic_num + 1 / (self.bins * self.ranges),
                              1 + magic_num - 1 / (self.bins * self.ranges), 2 / (self.bins * self.ranges + 1)).to(out_coord)
            coord = out_coord * mul
            coord = coord.sum(dim=-1)  # n, 4
        frame_res = {'pred_seqs': out, 'pred_boxes': coord}
        return frame_res

    def _post_process_single_image(self, frame_res, frame_index, sequence, last_boxes):
        # 将预测的结果代替历史
        if self.training:
            # the track id will be assigned by the mather.
            outputs_i = self.criterion.match_for_single_frame(frame_res, sequence[frame_index], last_boxes)

        # 随机删除物体
        #if self.random_drop > 0:
        #    sequence, frame_res = random_drop_tracks(sequence, frame_res, self.random_drop, frame_index)

        # 随机增加物体
        #if self.fp_ratio > 0:
        #    sequence, frame_res = self._add_fp_tracks(sequence, frame_res, frame_index)

        return frame_res, sequence

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, seq_last):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)

        extra_seq, values = self._forward_single_image(img,
                                                       seq_last, 0, is_train=False)

        return extra_seq, values

    def forward(self, data: dict, sequence):
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        # print(self.vocab_embedding.weight)
        frames = data['imgs']  # list of Tensor.
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
            'pred_seqs': [],
        }
        # print(sequence)
        last_boxes = None
        for frame_index, frame in enumerate(frames):
            # print(frame_index)
            frame.requires_grad = False

            frame = nested_tensor_from_tensor_list([frame])
            outputs = self._forward_single_image(frame, sequence, frame_index)
            frame_res, sequence = self._post_process_single_image(outputs, frame_index, sequence, last_boxes)

            #outputs['pred_logits'] = frame_res['pred_logits']
            #outputs['pred_boxes'] = frame_res['pred_boxes']
            #outputs['pred_seqs'] = frame_res['pred_seqs']
            #last_boxes = frame_res['pred_boxes']

        if self.training:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'e2e_mot': 1,
        'e2e_dance': 1,
        'e2e_joint': 1,
        'e2e_static_mot': 1,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer_seq(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(args, args.query_interaction_layer, d_model, hidden_dim,
                                                            d_model * 2)

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_giou".format(i): args.cls_loss_giou,
                            'frame_{}_loss_ce'.format(i): args.loss_varifocal,
                            'frame_{}_loss_label'.format(i): args.loss_label,
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_giou".format(i, j): args.cls_loss_giou,
                                    'frame_{}_aux{}_loss_varifocal'.format(i, j): args.loss_varifocal,
                                    'frame_{}_aux{}_loss_label'.format(i, j): args.loss_label,
                                    })
    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        for i in range(num_frames_per_batch):
            weight_dict.update({"frame_{}_track_loss_ce".format(i): args.cls_loss_coef})
    else:
        memory_bank = None
    losses = ['labels', 'boxes']
    bins = 2000
    ranges = 2
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses, bins=bins,
                            ranges=ranges)
    criterion.to(device)
    postprocessors = {}
    model = MOTR_seq(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=False,
        bins=bins,
        ranges=ranges,
        random_drop=args.random_drop,
        fp_ratio=args.fp_ratio,
        criterion=criterion,
    )
    return model, criterion, postprocessors
