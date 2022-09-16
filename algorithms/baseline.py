import os

import numpy as np
import imageio
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

from common.visulaizations import draw_bb

class Baseline():
    def __init__(self, configs):
        self.model = init_detector(configs.config_file, configs.weight_file, device=configs.device)
        self.result_dir = configs.result_dir
        self.score_th = 0.5
    def __call__(self, dataset):
        # test a single image and show the results
        total_imgs = len(dataset)

        # Inference using the model
        for idx in range(0, total_imgs):
            print(f"Processing : {idx} / {total_imgs}")
            out_filename = os.path.join(self.result_dir, f"{idx}.png")
            img = dataset.get_images(idx)
            result = inference_detector(self.model, img)
            det_img = draw_bb(img, result, self.score_th)
            imageio.imwrite(out_filename, det_img)

        return [50, 20, 1] # Still dummy 
