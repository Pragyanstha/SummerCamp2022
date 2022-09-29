import os
import cv2
import numpy as np
import imageio
import mmcv
import torch
import torch.distributed as dist
from mmdet.models import build_detector
from mmcv import Config, DictAction

from PIL import Image

from argparse import Namespace
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from mmdet.apis import init_detector, inference_detector

from common.visulaizations import draw_bb

from metriclearning.calc_metric import inference,calc_metric

class Baseline_ML():
    def __init__(self, configs):
        self.model = init_detector(configs.config_file, configs.weight_file, device=configs.device)
        self.result_dir = configs.result_dir
        self.score_th = 0.9
        self.ML_model_path = configs.ML_model_path
        self.npz = np.load(configs.ML_features_path)

    def __call__(self, dataset):
        # test a single image and show the results
        total_imgs = len(dataset)
        tracked = []
        # Inference using the model
        for idx in range(0, total_imgs, 10):
            print(f"Processing : {idx} / {total_imgs}")
            out_filename = os.path.join(self.result_dir, f"{idx}.png")
            img = dataset.get_images(idx)
            result = inference_detector(self.model, img)

            result_neo = result.copy()

            for fish_id,fish in enumerate(result):
                 for fi,f in enumerate(fish):
                     if f[4] > self.score_th:
                         bb = f
                         bbox_img = img[int(bb[1]):int(bb[3]),int(bb[0]):int(bb[2])]
                         bbox_img_n = bbox_img.copy()
                         bbox_img_n = Image.fromarray(cv2.cvtColor(bbox_img_n, cv2.COLOR_BGR2RGB))
                         bbox_img_n.save("results/gomi/{}{}.png".format(fish_id,int(f[4])))
                         model_path_ML = self.ML_model_path
                         numpy_data,numpy_labels = self.npz["arr_0"],self.npz["arr_1"]
                         ML_class = calc_metric(numpy_data,numpy_labels,bbox_img_n,model_path_ML)
                         if not fish_id==ML_class:
                             result_neo[fish_id][fi][4] = 0.0

            tracked.append(self._count_fish(result_neo, self.score_th))
            det_img = draw_bb(img, result_neo, self.score_th)
            imageio.imwrite(out_filename, det_img)

        tracked = np.array(tracked)
        median_tracked = np.max(tracked, axis=0)
        return median_tracked

    # def _count_fish(self, dets):
    #     res = [0, 0, 0, 0, 0] # Five fishes
    #     for label_id, class_det in enumerate(dets):
    #         count = len(class_det)
    #         res[label_id] = count
    
    #     return res

    def _count_fish(self, dets, score_th):
        res = [0, 0, 0, 0, 0] # Five fishes
        for label_id, class_det in enumerate(dets):
            scores = class_det[:, -1]
            selected_ids = scores > score_th
            count = len(class_det[selected_ids, :])
            res[label_id] = count
        
        return res