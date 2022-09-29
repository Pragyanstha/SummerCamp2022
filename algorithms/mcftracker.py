import os
import time
from typing import Any, Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass

import numpy as np
import imageio
import cv2
import mmcv
import scipy
import torch
import torch.distributed as dist
from mmdet.models import build_detector
from mmcv import Config, DictAction
from argparse import Namespace
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from mmdet.apis import init_detector, inference_detector
import globalflow as gflow

from common.visulaizations import draw_bb

@dataclass(frozen=True)
class Detection:
    center: np.ndarray
    minc: np.ndarray
    maxc: np.ndarray
    area: float
    reid: Optional[np.ndarray] = None


@dataclass
class Stats:
    minc: np.ndarray
    maxc: np.ndarray
    num_max_det: int


def quiet_divide(a, b):
    """Quiet divide function that does not warn about (0 / 0)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.true_divide(a, b)

def calc_HS_histogram(image, roi):
    roi = [roi.minc[0], roi.minc[1], roi.maxc[0], roi.maxc[1]]
    roi = [int(a) for a in roi]
    cropped = image[roi[1]:roi[3], roi[0]:roi[2], :]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
    return hist


def calc_bhattacharyya_distance(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def boxiou(det1, det2):
    """Computes IOU of two rectangles. Taken from
    https://github.com/cheind/py-motmetrics/blob/6597e8a4ed398b9f14880fa76de26bc43d230836/motmetrics/distances.py#L64
    """
    a_min, a_max = det1.minc, det1.maxc
    b_min, b_max = det2.minc, det2.maxc
    # Compute intersection.
    i_min = np.maximum(a_min, b_min)
    i_max = np.minimum(a_max, b_max)
    i_size = np.maximum(i_max - i_min, 0)
    i_vol = np.prod(i_size, axis=-1)
    # Get volume of union.
    a_size = np.maximum(a_max - a_min, 0)
    b_size = np.maximum(b_max - b_min, 0)
    a_vol = np.prod(a_size, axis=-1)
    b_vol = np.prod(b_size, axis=-1)
    u_vol = a_vol + b_vol - i_vol
    return np.where(
        i_vol == 0, np.zeros_like(i_vol, dtype=float), quiet_divide(i_vol, u_vol)
    )


class MCFTracker():
    def __init__(self, configs):
        self.model = init_detector(configs.config_file, configs.weight_file, device=configs.device)
        self.result_dir = configs.result_dir
        self.score_th = 0.9


    def __call__(self, dataset):
        # test a single image and show the results
        total_imgs = len(dataset)

        results = []
        # Inference using the model
        for idx in range(0, total_imgs):
            print(f"Processing : {idx} / {total_imgs}")
            # out_filename = os.path.join(configs.result_dir, f"{idx}.png")
            img = dataset.get_images(idx)
            result = inference_detector(self.model, img)
            results.append(result)

        count_0 = self._flow(results, 0,  dataset)
        count_1 = self._flow(results, 1,  dataset)
        count_2 = self._flow(results, 2,  dataset)
        count_3 = self._flow(results, 3,  dataset)
        count_4 = self._flow(results, 4,  dataset)

        return [count_0, count_1, count_2, count_3, count_4]
    
    def _flow(self, result, class_id, dataset):
        
        total_imgs = len(dataset)
        detections = {}
        tags = {}
        images = {}
        t2f = []   
        for idx in range(0, total_imgs):
            class_dets = result[idx][class_id]
            img_id = f"{idx:04d}"
            scores = class_dets[:, -1]
            class_dets = class_dets[scores>self.score_th,:]
            detections[img_id] = class_dets
            tags[img_id] = [a[0:4] for a in class_dets]
            img = dataset.get_images(idx)
            images[img_id] = img
            t2f.append(img_id)

            timeseries = []
            fnames = []

        stats = Stats(minc=np.array([1e3] * 2), maxc=np.array([-1e3] * 2), num_max_det=0)
        for t, (fname, objs) in enumerate(detections.items()):
            fnames.append(fname)
            tdata = []

            for oidx, obj in enumerate(objs):
                minc = np.array([obj[0], obj[1]])
                maxc = np.array([obj[2], obj[3]])
                c = (minc + maxc) * 0.5
                area = (maxc[0] - minc[0]) * (maxc[1] - minc[1])

                tdata.append(Detection(c, minc, maxc, area))
                stats.minc = np.minimum(stats.minc, minc)
                stats.maxc = np.maximum(stats.maxc, maxc)
            timeseries.append(tdata)
            stats.num_max_det = max(stats.num_max_det, len(tdata))

        class GraphCosts(gflow.StandardGraphCosts):
            def __init__(self):
                super().__init__(
                    penter=1e-2,
                    pexit=1e-4,
                    beta=2e-2,
                    max_obs_time=len(timeseries) - 1,
                )

            def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
                """Log-probability of pairing xi(t-1) with xj(t).
                Modelled by intersection over union downweighted by an
                exponential decreasing probability on the time-difference.
                """
                iou_logprob = np.log(boxiou(x.obs, y.obs) + 1e-8)
                tdiff = y.time_index - x.time_index
                tlogprob = scipy.stats.expon.logpdf(
                    tdiff, loc=1.0, scale=1 / 1.0
                )
                hist1 = calc_HS_histogram(images[t2f[x.time_index]], x.obs)
                hist2 = calc_HS_histogram(images[t2f[y.time_index]], x.obs)
                prob_color = 1.0 - calc_bhattacharyya_distance(hist1, hist2)
                return -(iou_logprob)

        flowgraph = gflow.build_flow_graph(
        timeseries, GraphCosts(), num_skip_layers=3, cost_scale=1e4, max_cost=3e3
        )
        try:
            flowdict, _, _ = gflow.solve(flowgraph, (1, 30))
            traj = gflow.find_trajectories(flowdict)
            obs_to_traj = gflow.label_observations(timeseries, traj)
            traj_info = [
                {"idx": tidx, "start": fnames[t[0].time_index], "end": fnames[t[-1].time_index]}
                for tidx, t in enumerate(traj)
            ]
            # Use filenames instead of time indices
            obs_to_traj = {fname: ids for fname, ids in zip(fnames, obs_to_traj)}
            count = stats.num_max_det
        except ValueError as e:
            print(f"could not solve for class {class_id}")
            count = 0

        return count
