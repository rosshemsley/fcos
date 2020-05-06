import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np


from fcos.datasets import tensor_to_image, collate_fn
from fcos.inference import compute_detections_for_tensor, render_detections_to_image, detections_from_net
from fcos.models import FCOS

def train(dataset):
    # x, box_labels, class_labels = dataset[0]
    # print("image", img)

    # img = np.zeros((3, 100, 100))
    # img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    # img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    # img_HWC = np.zeros((100, 100, 3))
    # img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
    # img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device("cuda")
    else:
        print("using cpu")
        device = torch.device("cpu")

    model = FCOS()
    model.to(device)

    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=5,
    #     shuffle=True,
    #     num_workers=8,
    # )

    # grid = torchvision.utils.make_grid([img])
    with SummaryWriter() as writer:
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2, collate_fn=collate_fn)

        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        learning_rate = 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

        steps = 0
        for epoch in range(100):
            print("START EPOCH", epoch)
            if epoch == 0:
                print("FREEZE BACKBONE")
                model.freeze_backbone()
            else:
                print("UNFREEZE BACKBONE")
                model.unfreeze_backbone()

            for i, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                x, box_labels, class_labels = data

                x = x.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                loss = model(x, box_labels, class_labels)
                loss.backward()
                optimizer.step()
                print("EPOCH:", epoch, "batch item i", i, "of", len(trainloader), "LOSS", loss.item())
                writer.add_scalar('Loss/train', loss.item(), i)
                steps += 1

                if i %100 ==0:
                    with torch.no_grad():
                        _test_model(i, writer, model, dataset, device)

                    path = os.path.join("checkpoints", f"{i}.chkpt")
                    print("save to ", path)
                    torch.save(model.state_dict(), path)

            print("learning rate step")
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


# x = x.to(\)
# box_labels = box_labels.to(device)
# class_labels = class_labels.to(device)
# loss = model(
#     torch.unsqueeze(x, dim=0),
#     torch.unsqueeze(box_labels, dim=0),
#     torch.unsqueeze(class_labels, dim=0),
# )

# print("LOSS", loss)

# # detections = compute_detections_for_tensor(model, x, device)
# # img = tensor_to_image(x)
# # render_detections_to_image(img, detections)
# # writer.add_image("fcos detections", img, 0, dataformats="HWC")

# img_labels = tensor_to_image(x.cpu())
# ground_truth_detections = detections_from_net(
#     torch.unsqueeze(box_labels, dim=0),
#     torch.unsqueeze(class_labels, dim=0)
# )[0]
# render_detections_to_image(img_labels, ground_truth_detections)
# writer.add_image("fcos labels", img_labels, 0, dataformats="HWC")






