import numpy as np
import cv2
import torch
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

from common.constants import LABEL_MAP, FONT_FILE


def draw_bb(img, bboxes, score_th):
    img = torch.from_numpy(img)
    img = img.permute([2, 0, 1]) # Channels first  
    for label_id, class_bboxes in enumerate(bboxes):
        labels = np.array([LABEL_MAP[label_id] for _ in range(len(class_bboxes))])
        class_bboxes = torch.from_numpy(np.array(class_bboxes))
        scores = class_bboxes[:, -1]
        valid_ids = (scores > score_th).numpy()
        class_bboxes = class_bboxes[valid_ids]

        labels = labels[valid_ids]
        img = draw_bounding_boxes(img, class_bboxes[:, :4],labels=labels, 
            width=2, colors="blue", fill=False, font_size=12, font=FONT_FILE)
    img = img.permute([1, 2, 0]).numpy() # Revert back to channels last
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
    return img
