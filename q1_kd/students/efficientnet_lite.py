"""
EfficientNet-Lite0 implemented from scratch, adapted for 32x32 inputs.

EfficientNet-Lite is the edge/mobile variant of EfficientNet-B0 (Tan & Le, 2019):
  * built from MBConv (inverted residual) blocks, same stage layout as B0,
  * Squeeze-and-Excitation removed (poorly supported on many edge accelerators),
  * Swish replaced by ReLU6 (quantisation friendly),
  * stem and head are fixed (not scaled) when compound scaling is applied.
Lite0 uses width = depth = 1.0 (the base network).

The model also implements stochastic depth ("drop connect") on residual blocks.

CIFAR adaptation: the stem uses stride 1 (instead of 2), so a 32x32 image results in a
2x2 final feature map (total stride 16), matching the teacher's last stage.
"""
import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1, groups=1, act=True):
        layers = [nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class MBConv(nn.Module):
    """Mobile inverted bottleneck WITHOUT squeeze-excitation (Lite variant)."""

    def __init__(self, cin, cout, k, stride, expand, drop_path=0.0):
        super().__init__()
        hidden = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.drop_path = drop_path
        layers = []
        if expand != 1:
            layers.append(ConvBNAct(cin, hidden, 1))
        layers += [ConvBNAct(hidden, hidden, k, stride, groups=hidden),   # depthwise kxk
                   ConvBNAct(hidden, cout, 1, act=False)]                 # linear projection
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if not self.use_res:
            return out
        if self.training and self.drop_path > 0:
            keep = 1 - self.drop_path
            mask = torch.empty(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep)
            out = out / keep * mask
        return x + out


class EfficientNetLite0(nn.Module):
    # expand, kernel, stride, out_channels, repeats   (B0 stage layout)
    CFG = [(1, 3, 1, 16, 1),
           (6, 3, 2, 24, 2),
           (6, 5, 2, 40, 2),
           (6, 3, 2, 80, 3),
           (6, 5, 1, 112, 3),
           (6, 5, 2, 192, 4),
           (6, 3, 1, 320, 1)]

    def __init__(self, num_classes=100, dropout=0.2, drop_path_rate=0.1):
        super().__init__()
        stem, head = 32, 1280                       # fixed in the Lite family
        layers = [ConvBNAct(3, stem, 3, stride=1)]
        total = sum(c[4] for c in self.CFG)
        cin, idx = stem, 0
        for e, k, s, c, n in self.CFG:
            for i in range(n):
                layers.append(MBConv(cin, c, k, s if i == 0 else 1, e,
                                     drop_path=drop_path_rate * idx / total))
                cin, idx = c, idx + 1
        layers.append(ConvBNAct(cin, head, 1))
        self.body = nn.Sequential(*layers)
        self.feature_channels = head
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(head, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -1.0 / m.in_features ** 0.5, 1.0 / m.in_features ** 0.5)
                nn.init.zeros_(m.bias)

    def features(self, x):
        return self.body(x)

    def classifier(self, f):
        return self.head(f)

    def forward(self, x):
        return self.classifier(self.features(x))
