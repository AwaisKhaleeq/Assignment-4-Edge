"""
Teacher network: ResNet-50 for CIFAR-100.

Design notes
------------
* Backbone = torchvision ResNet-50 (bottleneck blocks 3-4-6-3). By default it is
  initialised from ImageNet weights (transfer learning) and then fine-tuned on
  CIFAR-100, which gives a strong teacher in a small number of epochs.
* CIFAR images are 32x32. The ImageNet stem (7x7/2 conv + maxpool) would shrink them
  to 8x8 immediately, so the teacher internally up-samples its input to
  ``input_res`` (default 64) before the stem. Students and teacher therefore receive
  exactly the same 32x32 batch during distillation.
* Like every model in this project, the teacher exposes
      features(x)   -> last convolutional feature map  (used for feature KD)
      classifier(f) -> logits
  and forward(x) = classifier(features(x)).
"""
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNet50Teacher(nn.Module):
    def __init__(self, num_classes=100, input_res=64, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        self.input_res = input_res
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(2048, num_classes)
        self.feature_channels = 2048

    def features(self, x):
        if self.input_res and x.shape[-1] != self.input_res:
            x = F.interpolate(x, size=(self.input_res, self.input_res), mode="bilinear", align_corners=False)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def classifier(self, f):
        return self.fc(self.dropout(self.pool(f).flatten(1)))

    def forward(self, x):
        return self.classifier(self.features(x))
