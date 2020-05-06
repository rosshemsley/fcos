import cv2
from dataclasses import dataclass
from typing import List
import numpy as np
import torch

from fcos.models import FCOS

MIN_SCORE = 0.05

@dataclass
class Detection:
    score: float
    object_class: int
    # encoded as [min_x, min_y, max_x, max_y]
    bbox: np.ndarray

def render_detections_to_image(img: np.ndarray, detections: List[Detection]):
    for detection in detections:
        start_point = (detection.bbox[0], detection.bbox[1])
        end_point = (detection.bbox[2], detection.bbox[3])
        img = cv2.rectangle(img, start_point, end_point, (0,255,0), 1)

    return img

def compute_detections(model: FCOS, img: np.ndarray, device) -> List[Detection]:
    """
    Take an image using opencv conventions and return a list of detections.
    """
    tensor = img
    return compute_detections_for_tensor(model, tensor, device)


def compute_detections_for_tensor(model: FCOS, x, device) -> List[Detection]:
    with torch.no_grad():
        x = x.to(device)
        x = torch.unsqueeze(x, 0)

        scores, classes, boxes = model(x)
        return detections_from_net(boxes, classes, scores)[0]


def detections_from_net(boxes_by_batch, classes_by_batch, scores_by_batch=None) -> List[List[Detection]]:
    """
    - BHW[c] class index of each box (int)
    - BHW[p] class probability of each box (float)
    - BHW[min_x, y_min, x_min, y_max, x_max] (box dimensions, floats)
    """
    result = []

    for batch in range(classes_by_batch.shape[0]):
        scores = scores_by_batch[batch] if scores_by_batch is not None else None
        classes = classes_by_batch[batch]
        boxes = boxes_by_batch[batch]

        result.append([
            Detection(
                score=scores[i].item() if scores is not None else 1.0,
                object_class=classes[i].item(),
                bbox=boxes[i].cpu().numpy().astype(int),
            )
            for i in range(boxes.shape[0])
            if classes[i] != 0 and scores is None or scores[i] > MIN_SCORE
        ])
    
    return result
