"""
MobileNetV2 (Sandler et al., 2018) implemented from scratch, adapted for 32x32 inputs.

Core idea: the *inverted residual* block
    1x1 expansion conv (t x channels) -> 3x3 depthwise conv -> 1x1 linear projection
with a residual connection when stride = 1 and in/out channels match. The final
projection has no activation ("linear bottleneck") so that information in the
low-dimensional manifold is not destroyed by ReLU.

CIFAR adaptation: the stem uses stride 1 (instead of 2), so a 32x32 image ends up as a
2x2 final feature map (total stride 16) instead of 1x1. This keeps the compute
(~49 MFLOPs) low enough for CPU training while preserving spatial structure; the
2x2 map also matches the spatial size of the teacher's last stage (feature KD).
"""
import torch.nn as nn


def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand_ratio):
        super().__init__()
        hidden = int(round(cin * expand_ratio))
        self.use_res = stride == 1 and cin == cout
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(cin, hidden, k=1))                       # pointwise expansion
        layers += [
            ConvBNReLU(hidden, hidden, k=3, stride=stride, groups=hidden),    # depthwise
            nn.Conv2d(hidden, cout, 1, bias=False),                            # linear projection
            nn.BatchNorm2d(cout)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2(nn.Module):
    # t (expansion), c (channels), n (repeats), s (stride of first block)
    CFG = [(1, 16, 1, 1),
           (6, 24, 2, 2),
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=100, width_mult=1.0, dropout=0.2):
        super().__init__()
        cin = _make_divisible(32 * width_mult)
        last = _make_divisible(1280 * max(1.0, width_mult))
        layers = [ConvBNReLU(3, cin, stride=1)]                                # CIFAR stem: stride 1
        for t, c, n, s in self.CFG:
            cout = _make_divisible(c * width_mult)
            for i in range(n):
                layers.append(InvertedResidual(cin, cout, s if i == 0 else 1, t))
                cin = cout
        layers.append(ConvBNReLU(cin, last, k=1))
        self.body = nn.Sequential(*layers)
        self.feature_channels = last
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(last, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def features(self, x):
        return self.body(x)

    def classifier(self, f):
        return self.head(f)

    def forward(self, x):
        return self.classifier(self.features(x))
