import cv2
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torchvision

from fcos.models import FCOS, normalize_batch

MIN_SCORE = 0.05
DEFAULT_MAX_DETECTIONS = 3000

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
        img = cv2.rectangle(img, start_point, end_point, (0, 255, 0), 1)

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

        batch_size = x.shape[0]
        batch = normalize_batch(x)
        classes_by_feature, boxes_by_feature = model(batch)

        all_classes = []
        all_boxes = []

        for feature_classes, feature_boxes in zip(classes_by_feature, boxes_by_feature):

            # print("Class shape", feature_classes.shape)
            all_classes.append(feature_classes.view(batch_size, -1, len(model.classes)))
            all_boxes.append(feature_boxes.view(batch_size, -1, 4))


        classes_ = torch.cat(all_classes, dim=1)
        boxes_ = torch.cat(all_boxes, dim=1)

        scores, classes, boxes = _gather_detections(classes_, boxes_)
        return detections_from_net(boxes, classes, scores)[0]



def detections_from_net(boxes_by_batch, classes_by_batch, scores_by_batch=None) -> List[List[Detection]]:
    """
    - BHW[c] class index of each box (int)
    - BHW[p] class probability of each box (float)
    - BHW[min_x, y_min, x_min, y_max, x_max] (box dimensions, floats)
    """
    result = []

    # print("SCORE by batch SHAPE", scores_by_batch.shape)
    # print("CLASS by batch SHAPE", classes_by_batch.shape)
    # print("BOXES by batch SHAPE", boxes_by_batch.shape)

    for batch in range(classes_by_batch.shape[0]):
        scores = scores_by_batch[batch] if scores_by_batch is not None else None
        classes = classes_by_batch[batch]
        boxes = boxes_by_batch[batch]

        # print("SCORE SHAPE", scores.shape)
        # print("CLASS SHAPE", classes.shape)
        # print("BOXES SHAPE", boxes.shape)

        result.append(
            [
                Detection(
                    score=scores[i].item() if scores is not None else 1.0,
                    object_class=classes[i].item(),
                    bbox=boxes[i].cpu().numpy().astype(int),
                )
                for i in range(boxes.shape[0])
                if classes[i] != 0 and scores is None or scores[i] > MIN_SCORE
            ]
        )

    return result


#         # if box_labels is None or class_labels is None:
#         #     # box_labels = box_labels.to(x.device)
#         #     # class_labels = class_labels.to(x.device)
#         #     # boxes: B[ltrb]
#         #     # classes: B[classes]
#         #     boxes = torch.cat(
#         #         [t.view(n_batches, -1, 4) for t in boxes_by_feature], dim=1
#         #     )
#         #     classes = torch.cat(
#         #         [t.view(n_batches, -1, len(CLASSES)) for t in classes_by_feature], dim=1
#         #     )
#         #     scores, classes, boxes = _gather_detections(
#         #         boxes, classes, self.max_detections
#         #     )
#         #     return scores, classes, boxes

#         # losses = []




# def _feature_loss_targets(labels, features, img_height, img_width, stride):
#     """
#     Takes a feature map for a single layer of the FPN
#     """


def _gather_detections(classes, boxes, max_detections=DEFAULT_MAX_DETECTIONS):
    # print("INPUT TO GATHER, b", boxes.shape)
    # print("INPUT TO GATHER, c", classes.shape)

    # classes: BHW[c] class index of each box
    class_scores, class_indices = torch.max(classes.sigmoid(), dim=2)

    boxes_by_batch = []
    classes_by_batch = []
    scores_by_batch = []

    # TODO(Ross): there must be a nicer way
    n_batches = boxes.shape[0]
    for i in range(n_batches):
        non_background_points = class_indices[i] > 0

        print("non-background found", non_background_points.nonzero().shape)

        class_scores_i = class_scores[i][non_background_points]
        boxes_i = boxes[i][non_background_points]
        class_indices_i = class_indices[i][non_background_points]

        num_detections = min(class_scores_i.shape[0], max_detections)
        _, top_detection_indices = torch.topk(class_scores_i, num_detections, dim=0)

        top_boxes_i = torch.index_select(boxes_i, 0, top_detection_indices)
        top_classes_i = torch.index_select(class_indices_i, 0, top_detection_indices)
        top_scores_i = torch.index_select(class_scores_i, 0, top_detection_indices)

        # print("top_boxes_i", top_boxes_i.shape)
        # print("top_classes_i", top_classes_i.shape)
        # print("top_scores_i", top_scores_i.shape)

        # print("BEFORE filter", top_boxes_i.shape)
        boxes_to_keep = torchvision.ops.nms(top_boxes_i, top_scores_i, 0.6)

        top_boxes_i = top_boxes_i[boxes_to_keep]
        top_classes_i = top_classes_i[boxes_to_keep]
        top_scores_i = top_scores_i[boxes_to_keep]

        # print("top_boxes_i", top_boxes_i.shape)
        # print("top_classes_i", top_classes_i.shape)
        # print("top_scores_i", top_scores_i.shape)

        boxes_by_batch.append(top_boxes_i)
        classes_by_batch.append(top_classes_i)
        scores_by_batch.append(top_scores_i)

    top_boxes = torch.stack(boxes_by_batch, dim=0)
    top_classes = torch.stack(classes_by_batch, dim=0)
    top_scores = torch.stack(scores_by_batch, dim=0)

    return top_scores, top_classes, top_boxes



