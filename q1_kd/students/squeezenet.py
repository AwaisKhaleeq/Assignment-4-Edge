"""
SqueezeNet v1.1 (Iandola et al., 2016) implemented from scratch, adapted for 32x32 inputs.

Core idea: the *Fire module*
    squeeze : 1x1 conv that reduces channels (s1x1 filters)
    expand  : parallel 1x1 (e1x1) and 3x3 (e3x3) convs whose outputs are concatenated
Replacing most 3x3 filters by 1x1 filters and feeding 3x3 filters with few input
channels gives AlexNet-level accuracy with ~50x fewer parameters. There are no
fully connected layers: a final 1x1 conv produces one map per class and global
average pooling turns them into logits.

Modifications for CIFAR / stable training from scratch:
  * stem = 3x3 conv stride 1 (instead of 3x3 stride 2 on 224px images),
  * BatchNorm after each convolution,
  * no ReLU after the final classifier conv (it would clip negative logits).
"""
import torch
import torch.nn as nn


def conv_bn_relu(cin, cout, k, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Fire(nn.Module):
    def __init__(self, cin, squeeze, e1, e3):
        super().__init__()
        self.squeeze = conv_bn_relu(cin, squeeze, 1)
        self.expand1 = conv_bn_relu(squeeze, e1, 1)
        self.expand3 = conv_bn_relu(squeeze, e3, 3)

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.expand1(s), self.expand3(s)], 1)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=100, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn_relu(3, 64, 3, stride=1),             # 32x32
            nn.MaxPool2d(3, 2, 1),                        # 16x16
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(3, 2, 1),                        # 8x8
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1),                        # 4x4
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256))                      # -> 512 x 4 x 4
        self.feature_channels = 512
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Conv2d(512, num_classes, 1),   # class maps
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def features(self, x):
        return self.body(x)

    def classifier(self, f):
        return self.head(f)

    def forward(self, x):
        return self.classifier(self.features(x))
