import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np


from fcos.datasets import tensor_to_image, collate_fn
from fcos.inference import (
    compute_detections_for_tensor,
    render_detections_to_image,
    detections_from_net,
)
from fcos.models import FCOS, normalize_batch

from .targets import generate_targets


def train(dataset):
    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device("cuda")
    else:
        print("using cpu")
        device = torch.device("cpu")

    model = FCOS()
    model.to(device)

    with SummaryWriter() as writer:
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=3, shuffle=True, num_workers=2, collate_fn=collate_fn
        )

        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        learning_rate = 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

        steps = 0
        for epoch in range(100):
            model.train()

            print("START EPOCH", epoch)
            if epoch == 0:
                print("FREEZE BACKBONE")
                model.freeze_backbone()
            else:
                print("UNFREEZE BACKBONE")
                model.unfreeze_backbone()

            for train_idx, (x, class_labels, box_labels) in enumerate(trainloader, 0):
                batch_size = x.shape[0]
                print("batch size", batch_size, "training labels count", len(class_labels))

                optimizer.zero_grad()

                x = x.to(device)

                batch = normalize_batch(x)
                batch = batch.to(device)
                classes_by_feature, boxes_by_feature = model(batch)

                class_targets_by_feature, box_targets_by_feature = generate_targets(x.shape, class_labels, box_labels, model.strides)

                class_loss = torch.nn.CrossEntropyLoss()
                box_loss = torch.nn.L1Loss()

                # loss = torch.tensor(0.0).to(device)
                losses = []
                for feature_idx in range(len(classes_by_feature)):

                    # print("target has ", class_targets_by_feature[j][0].nonzero().sum(), "elements" )
                    # writer.add_image(f"CLASS TARGET test {j}", class_targets_by_feature[j][0] * 255, i, dataformats="HW")

                    cls_target = class_targets_by_feature[feature_idx].to(device).view(batch_size, -1)
                    box_target = box_targets_by_feature[feature_idx].to(device).view(batch_size, -1, 4)

                    cls_view = classes_by_feature[feature_idx].view(batch_size, -1, len(model.classes))
                    box_view = boxes_by_feature[feature_idx].view(batch_size, -1, 4)

                    for batch_idx in range(batch_size):
                        l = class_loss(cls_view[batch_idx], cls_target[batch_idx])
                        # print('class loss', l.item())
                        losses.append(l)

                        mask = cls_target[batch_idx] > 0
                        if mask.nonzero().shape[0] > 0:
                            bl = box_loss(box_view[batch_idx][mask], box_target[batch_idx][mask]) / 50.0
                            # print('box loss', bl.item())
                            losses.append(bl)

                loss = torch.stack(losses).mean()

                print(
                    "EPOCH:", epoch, "batch item i", train_idx, "of", len(trainloader), "LOSS", loss.item(),
                )

                writer.add_scalar("Loss/train", loss.item(), train_idx)
                loss.backward()
                optimizer.step()
                    

                steps += 1

                # forward + backward + optimize
                # loss = fcos_loss(targets, classes_by_feature, boxes_by_feature)
                # loss.backward()

                if train_idx % 100 == 0:
                    print("test model")
                    with torch.no_grad():
                        model.eval()
                        _test_model(train_idx, writer, model, dataset, device)

                    path = os.path.join("checkpoints", f"{train_idx}.chkpt")
                    print("save to ", path)
                    torch.save(model.state_dict(), path)

            scheduler.step()


def _test_model(i, writer, model, dataset, device):
    images = []
    for j in range(10):
        x, _, _ = dataset[j]
        img = tensor_to_image(x)

        x = x.to(device)
        detections = compute_detections_for_tensor(model, x, device)
        render_detections_to_image(img, detections)

        writer.add_image(f"fcos test {j}", img, i, dataformats="HWC")


#         # # Compute loss for each feature map
#         # for i, feature in enumerate(feature_pyramid):
#         #     box_target, class_target = _targets(
#         #         img_height, img_width, box_labels, class_labels, len(self.classes), self.strides[i]
#         #     )

#         #     box_target = box_target.to(device)
#         #     class_target = class_target.to(device)

#         #     boxes_i = boxes_by_feature[i]
#         #     classes_i = classes_by_feature[i]

#         #     l = _box_loss(boxes_i, self.strides[i], box_target, class_target) / 50.0
#         #     lc = _class_loss(classes_i, self.strides[i], class_target)

#         #     losses.append(l)
#         #     losses.append(lc)

#         # loss = torch.stack(losses)
#         # return loss.mean()


# def _box_loss(boxes, stride, box_target, class_target):
#     target_view = box_target[:, ::stride, ::stride, :]
#     mask = class_target[:, ::stride, ::stride] > 0
#     loss = nn.L1Loss()
#     v = loss(boxes[mask], target_view[mask])
#     return v


# def _class_loss(classes, stride, class_target):
#     target_view = class_target[:, ::stride, ::stride]

#     loss = nn.CrossEntropyLoss()

#     inval = classes.reshape(-1, 2)
#     tar = target_view.reshape(-1)

#     return loss(inval, tar)
