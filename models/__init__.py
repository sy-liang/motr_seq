# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .detr_seq import build_deforamble_transformer_seq as build_deformable_detr
from .motr_seq import build as build_motr


def build_model(args):
    arch_catalog = {
        'deformable_detr': build_deformable_detr,
        'motr': build_motr,
    }
    assert args.meta_arch in arch_catalog, 'invalid arch: {}'.format(args.meta_arch)
    build_func = arch_catalog[args.meta_arch]
    return build_func(args)

