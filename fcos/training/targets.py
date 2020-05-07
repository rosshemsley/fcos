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
    # print("gen targets. imshape: ", img_shape, "batch size", batch_size)
    class_targets_by_feature = []
    box_targets_by_feature = []

    for i, stride in enumerate(strides):
        h = int(img_shape[2] / stride)  
        w = int(img_shape[3] / stride)

        class_target_for_feature = torch.zeros(batch_size, h, w, dtype=int)
        box_target_for_feature = torch.zeros(batch_size, h, w, 4)

        min_box_side = 0 if i == 0 else strides[i-1]
        max_box_side = math.inf if i == len(strides) - 1 else stride

        for batch_idx, (class_labels, box_labels) in enumerate(zip(class_labels_by_batch, box_labels_by_batch)):
            # print("batch", batch_idx, "has", len(class_labels), len(box_labels), "labels")
            heights = box_labels[:, 2] - box_labels[:, 0]
            widths = box_labels[:, 3] - box_labels[:, 1]
            areas = torch.mul(widths, heights)

            for j in torch.argsort(areas, dim=0, descending=True):
                # print("  AREA", areas[j])
                if heights[j] < min_box_side or heights[j] > max_box_side:
                    continue
                if widths[j] < min_box_side or widths[j] > max_box_side:
                    continue

                min_x = int(box_labels[j][0] / stride)
                min_y = int(box_labels[j][1] / stride)
                max_x = int(box_labels[j][2] / stride) + 1
                max_y = int(box_labels[j][3] / stride) + 1

                # print("from", min_x, "to", max_x)
                # print("    setting label", box_labels[j])
                # print("    setting class", class_labels[j])


                class_target_for_feature[batch_idx, min_y:max_y, min_x:max_x] = class_labels[j]
                box_target_for_feature[batch_idx, min_y:max_y, min_x:max_x] = box_labels[j]

                # print('for feature map layer', i, "there are", (class_target_for_feature > 0).sum() )

        class_targets_by_feature.append(class_target_for_feature)
        box_targets_by_feature.append(box_target_for_feature)

    # Return [BHWC, BHW[min_x, min_y, max_x, max_y]]
    return class_targets_by_feature, box_targets_by_feature


# def _targets(img_height, img_width, box_labels, class_labels, num_classes, stride):
#     """
#     labels: B[x_min, y_min, x_max, y_max, class]
#     """
#     batches = len(class_labels)

#     box_targets = torch.zeros(batches, img_height, img_width, 4)
#     box_classes = torch.zeros(batches, img_height, img_width, dtype=torch.int64)

#     for batch in range(batches):
#         widths = box_labels[batch][:, 2] - box_labels[batch][:, 0]
#         heights = box_labels[batch][:, 3] - box_labels[batch][:, 1]
#         areas = torch.mul(widths, heights)

#         indices = torch.argsort(areas, dim=0, descending=True)
#         for index in indices:
#             bbox = box_labels[batch][index]
#             box_x_min = int(bbox[0])
#             box_y_min = int(bbox[1])
#             box_x_max = int(bbox[2]) + 1
#             box_y_max = int(bbox[3]) + 1
#             box_targets[batch, box_y_min:box_y_max, box_x_min:box_x_max] = bbox
#             box_classes[batch, box_y_min:box_y_max, box_x_min:box_x_max] = 1

#     return box_targets, box_classes
