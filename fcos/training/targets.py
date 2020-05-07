from typing import Tuple, List
import torch
import math


def generate_targets(
    img_shape: torch.LongTensor,
    class_labels_by_batch: List[torch.LongTensor],
    box_labels_by_batch: List[torch.FloatTensor],
    strides: List[int],
) -> List[Tuple[torch.FloatTensor, torch.LongTensor]]:
    """
    Given the shape of the input image, the box and class labels, and the stride of each
    feature map, construct the model targets for FCOS at each feature map scale.
    """
    if not len(box_labels_by_batch) == len(class_labels_by_batch) == img_shape[0]:
        raise ValueError("labels and batch size must match")

    batch_size = img_shape[0]

    class_targets_by_feature = []
    box_targets_by_feature = []

    for i, stride in enumerate(strides):
        h = int(img_shape[2] / stride)
        w = int(img_shape[3] / stride)

        class_target_for_feature = torch.zeros(batch_size, h, w, dtype=int)
        box_target_for_feature = torch.zeros(batch_size, h, w, 4)

        min_box_side = 0 if i == 0 else strides[i - 1]
        max_box_side = math.inf if i == len(strides) - 1 else stride

        for batch_idx, (class_labels, box_labels) in enumerate(
            zip(class_labels_by_batch, box_labels_by_batch)
        ):
            heights = box_labels[:, 2] - box_labels[:, 0]
            widths = box_labels[:, 3] - box_labels[:, 1]
            areas = torch.mul(widths, heights)

            for j in torch.argsort(areas, dim=0, descending=True):
                if heights[j] < min_box_side or heights[j] > max_box_side:
                    continue
                if widths[j] < min_box_side or widths[j] > max_box_side:
                    continue

                min_x = int(box_labels[j][0] / stride)
                min_y = int(box_labels[j][1] / stride)
                max_x = int(box_labels[j][2] / stride) + 1
                max_y = int(box_labels[j][3] / stride) + 1

                class_target_for_feature[batch_idx, min_y:max_y, min_x:max_x] = class_labels[j]
                box_target_for_feature[batch_idx, min_y:max_y, min_x:max_x] = box_labels[j]

        class_targets_by_feature.append(class_target_for_feature)
        box_targets_by_feature.append(box_target_for_feature)

    # Return [BHWC, BHW[min_x, min_y, max_x, max_y]]
    return class_targets_by_feature, box_targets_by_feature
