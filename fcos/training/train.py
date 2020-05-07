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

        learning_rate = 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

        for epoch in range(100):
            print("START EPOCH", epoch)

            model.train()

            if epoch == 0:
                print("FREEZE BACKBONE")
                model.freeze_backbone()
            else:
                print("UNFREEZE BACKBONE")
                model.unfreeze_backbone()

            for train_idx, (x, class_labels, box_labels) in enumerate(trainloader, 0):
                batch_size = x.shape[0]

                optimizer.zero_grad()

                x = x.to(device)
                batch = normalize_batch(x)

                classes_by_feature, boxes_by_feature = model(batch)
                class_targets_by_feature, box_targets_by_feature = generate_targets(
                    x.shape, class_labels, box_labels, model.strides
                )

                class_loss = torch.nn.CrossEntropyLoss()
                box_loss = torch.nn.L1Loss()

                losses = []
                for feature_idx in range(len(classes_by_feature)):
                    cls_target = class_targets_by_feature[feature_idx].to(device).view(batch_size, -1)
                    box_target = box_targets_by_feature[feature_idx].to(device).view(batch_size, -1, 4)

                    cls_view = classes_by_feature[feature_idx].view(batch_size, -1, len(model.classes))
                    box_view = boxes_by_feature[feature_idx].view(batch_size, -1, 4)

                    for batch_idx in range(batch_size):
                        losses.append(class_loss(cls_view[batch_idx], cls_target[batch_idx]))

                        mask = cls_target[batch_idx] > 0
                        if mask.nonzero().shape[0] > 0:
                            losses.append(
                                box_loss(box_view[batch_idx][mask], box_target[batch_idx][mask]) / 50.0
                            )

                loss = torch.stack(losses).mean()

                print(
                    "EPOCH:", epoch, "batch item i", train_idx, "of", len(trainloader), "LOSS", loss.item(),
                )

                writer.add_scalar("Loss/train", loss.item(), train_idx)
                loss.backward()
                optimizer.step()

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
