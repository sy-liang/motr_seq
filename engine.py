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
Train and eval functions used in main.py
"""
import cv2
import math
import numpy as np
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from util import box_ops

from torch import Tensor
from util.plot_utils import draw_boxes, draw_ref_pts, image_hwc2chw
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher, data_dict_to_cuda


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        # print("start!!")
        data_dict = data_dict_to_cuda(data_dict, device)

        # 编写input sequence，按帧编写
        gt_instances = data_dict['gt_instances']
        bins = 2000
        max_box = max([len(gt_instance.boxes) for gt_instance in gt_instances])
        num_box = max(max_box + 2, 100)
        input_seqs = []
        start = 4001
        end = 4000
        label = 0
        id_gt_past = []
        input_seqs = []
        no_know = 3999
        for index, gt_instance in enumerate(gt_instances):
            # print(f"index={index}")
            # 每一帧
            # x, y, w, h, label, id + EOS
            box_gt = gt_instance.boxes
            box_gt = torch.clamp(box_gt, min=-0.5, max=1.5)
            id_gt = gt_instance.obj_ids.long()
            label_gt = gt_instance.labels.long()
            label_gt += label

            # detect query随机初始化
            # 随机
            # print(f"box_gt={box_gt}")
            box = ((box_gt + 0.5) * (bins - 1)).long()
            # print(f"box={box}")
            # print(box.shape)
            # print(label_gt.shape)
            # print(id_gt.shape)
            box_label = torch.cat([box, label_gt.unsqueeze(-1), id_gt.unsqueeze(-1)], dim=-1)
            idx = torch.randperm(box_label.shape[0])
            box_label = box_label[idx]
            # print(id_gt)
            # print(f"box_label={box_label}")
            # print(box_label)

            #####测试新物体加入#####
            # if index == 2:
            #    input_seq_box = torch.rand(1, 4).to(box_gt)
            #    input_seq_box = (input_seq_box * (bins - 1)).long()
            #    input_seq_past = torch.cat(
            #      [input_seq_box,  torch.full((1, 1), no_know).to(input_seq_box), torch.full((1, 1), -1).to(input_seq_box)], dim=-1)
            #    box_label = torch.cat([box_label, input_seq_past], dim=0)
            #    input_seq_box = torch.rand(1, 4).to(box_gt)
            #    input_seq_box = (input_seq_box * (bins - 1)).long()
            #    input_seq_past = torch.cat(
            #      [input_seq_box,  torch.full((1, 1), 1).to(input_seq_box), torch.full((1, 1), -1).to(input_seq_box)], dim=-1)
            #    box_label = torch.cat([box_label, input_seq_past], dim=0)
            # print(box_label)
            #####测试旧物体消失#####
            # if index == 1:
            #    print("begin test")
            #    box_label = box_label[:-1]
            #    print(box_label)

            input_seq = []
            full_track_idxes = torch.arange(len(box_label), dtype=torch.long)
            if index != 0:
                input_seq = torch.zeros(len(id_gt_past), 6, dtype=torch.int).to(box_label)
                # print(input_seq.shape)
                # 后续帧按照上一帧的检出顺序
                # print(f"id_gt_past={id_gt_past}")
                # print(f"id_gt={id_gt}")
                # print(box_label[:, -1])
                # print(id_gt_past[0])
                # print(list(box_label[:, -1]).index(id_gt_past[0]))
                for i in range(len(id_gt_past)):
                    id_last = id_gt_past[i]
                    # print(id_last)
                    # print(box_label[:, -1])
                    if id_last in box_label[:, -1]:
                        # print("True")
                        idx = list(box_label[:, -1]).index(id_last)
                        # print(idx)
                        input_seq_past = box_label[idx]
                        # print(input_seq_past)
                        # box_label[idx, -1] = -1 # 已进入input_seq
                        full_track_idxes[idx] = -1
                        # print(input_seq_past)
                    else:
                        # print("now")
                        input_seq_box = torch.rand(1, 4).to(box_gt)
                        input_seq_box = torch.clamp(input_seq_box, min=-0.5, max=1.5)
                        input_seq_box = ((input_seq_box + 0.5) * (bins - 1)).long()
                        input_seq_label = torch.full((1, 1), no_know).to(label_gt)
                        input_seq_past = torch.cat(
                            [input_seq_box, input_seq_label, torch.full((1, 1), -1).to(input_seq_box)], dim=-1)
                        input_seq_past = input_seq_past.squeeze(0)
                    # 上一帧出现过的按照顺序检出排序
                    # print(input_seq_past)
                    input_seq[i] = input_seq_past
                # print(id_gt_past)
                # print(box_label[:, -1])
                # print(f"input_seq={input_seq}")
                # input_seq = torch.stack(input_seq)
                # print(input_seq)
                # print(i)
                # print(input_seq)
                # 这一帧新出现的放后面
                # print(input_seq)
                unmatched_track_idxes = full_track_idxes[full_track_idxes != -1]
                # print(len(unmatched_track_idxes))
                if len(unmatched_track_idxes) > 0:
                    input_seq_new = box_label[unmatched_track_idxes.to(box_label).type(torch.long)]
                    input_seq = torch.cat([input_seq.to(box_label), input_seq_new], dim=0)
            else:
                input_seq = box_label
            # print(f"input_seq={input_seq}")
            # print(index)
            # print(input_seq)

            # random_box = torch.rand(num_box - input_seq.shape[0], 4).to(box_gt)
            # random_box = (random_box * (bins - 1)).int()
            # random_label = torch.randint(0, 91, (num_box - input_seq.shape[0], 1)).to(label_gt)
            # random_label = random_label + category_start
            # random_box_label = torch.cat([random_box, random_label, torch.full((num_box - input_seq.shape[0], 1), -1)], dim=-1)

            # input_seq_det = torch.cat([input_seq, random_box_label], dim=0)
            # input_seq_det = torch.cat([torch.ones(1).to(input_seq) * start, input_seq_det.flatten()])
            # input_seqs_det.append(input_seq_det.unsqueeze(0))

            id_gt_past = input_seq[:, -1]
            # if index == 0:
            # 在第一帧前添加start
            #    input_seq = torch.cat([torch.ones(1).to(input_seq) * start, input_seq.flatten()], dim=-1)
            # 每一帧后添加end
            # input_seq = torch.cat([input_seq.flatten(),
            #                       torch.ones(1).to(input_seq) * end], dim=-1)
            # print(input_seq)
            input_seqs.append(input_seq)

            # print(f"input_seqs={input_seqs}")
        outputs = model(data_dict, input_seqs)
        input_seqs = []

        loss_dict = criterion(outputs, data_dict)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        # gather the stats from all processes

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             )
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
