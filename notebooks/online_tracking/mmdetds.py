import numpy as np
from mmdet.apis import init_detector, inference_detector

from common.aux import xyxy_to_xywh


# detectionモデルの定義
class MMdetDS(object):
     def __init__(self, configs):
         self.model = init_detector(configs.config_file, configs.weight_file, device=configs.device)
         self.result_dir = configs.result_dir
         self.score_th = configs.score_th
     
     def __call__(self, img):
         bbox_result = inference_detector(self.model, img)
         bboxes = np.vstack(bbox_result)

         if len(bboxes) == 0:
             bbox = np.array([]).reshape([0, 4])
             cls_conf = np.array([])
             cls_ids = np.array([])
             return bbox, cls_conf, cls_ids

         bbox = bboxes[:, :4]
         cls_conf = bboxes[:, 4]
         cls_ids = [
             np.full(bbox.shape[0], i, dtype=np.int32)
             for i, bbox in enumerate(bbox_result)
         ]
         cls_ids = np.concatenate(cls_ids)

         selected_idx = cls_conf > self.score_th
         bbox = bbox[selected_idx, :]
         cls_conf = cls_conf[selected_idx]
         cls_ids = cls_ids[selected_idx]

         bbox = xyxy_to_xywh(bbox)

         return bbox, cls_conf, cls_ids, bbox_result