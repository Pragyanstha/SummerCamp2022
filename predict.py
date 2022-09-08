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

checkpoint_file = "work_dirs/faster_rcnn_r50_fpn/latest.pth"
config_file = "configs/faster_rcnn_r50_fpn.py"
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'data/Problem_04/images/0000.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)

# or save the visualization results to image files
model.show_result(img, result, out_file='results/test.jpg')
