import torch.nn as nn
from torchvision.models import resnet50


class Backbone(nn.Module):
    """
    Backbone based on a pretrained resnet50.

    Returns three layers
    - layer 1: 512,  H/8, W/8
    - layer 2: 1028, H/16, W/16
    - layer 3: 2048, H/32, W/32
    """

    def __init__(self):
        super(Backbone, self).__init__()
        self.resnet = resnet50(pretrained=True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        layer_1 = self.resnet.layer1(x)
        layer_2 = self.resnet.layer2(layer_1)
        layer_3 = self.resnet.layer3(layer_2)
        layer_4 = self.resnet.layer4(layer_3)

        return layer_2, layer_3, layer_4
