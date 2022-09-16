# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
from operator import mod
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmdet.models import build_detector
from mmcv import Config, DictAction
from argparse import Namespace
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from mmdet.apis import init_detector, inference_detector


def test_predict():
    checkpoint_file = "/workspace/SummerCamp2022/weights/latest.pth"
    config_file = "/workspace/SummerCamp2022/configs/faster_rcnn_r50_fpn.py"
    model = init_detector(config_file, checkpoint_file, device='cuda:1')

    # test a single image and show the results
    img = '/workspace/SummerCamp2022/data/Problem_04/images/0120.png'  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    print(result)
    assert len(result) > 0
