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
    centerness_target_by_feature = []
    box_targets_by_feature = []

    m = (0, 64, 128, 256, 512, math.inf)

    for i, stride in enumerate(strides):
        feat_h = int(img_shape[2] / stride)
        feat_w = int(img_shape[3] / stride)

        class_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, dtype=int)
        centerness_target_for_feature = torch.zeros(batch_size, feat_h, feat_w)
        box_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, 4)

        min_box_side = m[i]
        max_box_side = m[i + 1]

        for batch_idx, (class_labels, box_labels) in enumerate(
            zip(class_labels_by_batch, box_labels_by_batch)
        ):
            heights = box_labels[:, 3] - box_labels[:, 1]
            widths = box_labels[:, 2] - box_labels[:, 0]
            areas = torch.mul(widths, heights)

            for j in torch.argsort(areas, dim=0, descending=True):
                if heights[j] < min_box_side or heights[j] > max_box_side:
                    continue
                if widths[j] < min_box_side or widths[j] > max_box_side:
                    continue

                min_x = max(int(box_labels[j][0] / stride), 0)
                min_y = max(int(box_labels[j][1] / stride), 0)
                max_x = min(int(box_labels[j][2] / stride) + 1, feat_w)
                max_y = min(int(box_labels[j][3] / stride) + 1, feat_h)

                mid_x = (max_x + min_x) / 2.0
                mid_y = (max_y + min_y) / 2.0
                b_w = max_x - min_x
                b_h = max_y - min_y
                for x in range(min_x, max_x):
                    for y in range(min_y, max_y):
                        if x == 0 or y == 0 or x == max_x - 1 or y == max_y - 1:
                            dist = 0
                        # dy = mid_y - y
                        # dx = mid_x - x
                        else:
                            l = x - min_x
                            r = max_x - 1 - x
                            t = y - min_y
                            b = max_y - 1 - y
                            dist = math.sqrt(min(l, r) / float(max(l, r)) * min(t, b) / float(max(t, b)))

                        # dist = math.exp(-(dy*dy /b_h  + dx * dx / b_w ))

                        centerness = dist
                        # if y < 0 or x < 0 or y >= centerness_target_for_feature[batch_idx].shape[0] or x >= centerness_target_for_feature[batch_idx].shape[1]:
                        # continue
                        centerness_target_for_feature[batch_idx, y, x] = centerness

                class_target_for_feature[batch_idx, min_y:max_y, min_x:max_x] = class_labels[j]

                for x in range(min_x, max_x):
                    for y in range(min_y, max_y):
                        x_img = x * stride
                        y_img = y * stride
                        target = torch.Tensor(
                            [
                                float(x_img - box_labels[j][0]),
                                float(y_img - box_labels[j][1]),
                                float(box_labels[j][2] - x_img),
                                float(box_labels[j][3] - y_img),
                            ]
                        )
                        box_target_for_feature[batch_idx, y, x] = target

        class_targets_by_feature.append(class_target_for_feature)
        centerness_target_by_feature.append(centerness_target_for_feature)
        box_targets_by_feature.append(box_target_for_feature)

    # Return [BHWC, BHW[min_x, min_y, max_x, max_y]]
    return class_targets_by_feature, centerness_target_by_feature, box_targets_by_feature
