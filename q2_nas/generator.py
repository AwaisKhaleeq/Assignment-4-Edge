"""
Automated architecture generator: turns a sampled configuration (see search_space.py)
into a trainable PyTorch nn.Module.

Network template
    stem (3x3 conv, stride 1)
    -> stage_0 ... stage_{n-1}   (first block of every stage except stage_0 has stride 2)
    -> optional 1x1 head conv
    -> global average pool -> dropout -> linear classifier
Residual connections are added automatically whenever a block keeps shape.
"""
import torch.nn as nn

ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "relu6": lambda: nn.ReLU6(inplace=True),
    "hswish": lambda: nn.Hardswish(inplace=True),
    "silu": lambda: nn.SiLU(inplace=True),
    "leaky_relu": lambda: nn.LeakyReLU(0.1, inplace=True),
}


def conv_bn(cin, cout, k, stride=1, groups=1, act=None):
    layers = [nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(ACTIVATIONS[act]())
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """Standard KxK convolution -> BN -> activation."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.op = conv_bn(cin, cout, k, stride, act=act)
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        return x + self.op(x) if self.res else self.op(x)


class DSConvBlock(nn.Module):
    """Depthwise-separable convolution: KxK depthwise (per-channel) + 1x1 pointwise.
    Cost ~ (K^2 + C_out) / (K^2 * C_out) of a standard conv."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.op = nn.Sequential(conv_bn(cin, cin, k, stride, groups=cin, act=act),   # depthwise
                                conv_bn(cin, cout, 1, act=act))                      # pointwise
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        return x + self.op(x) if self.res else self.op(x)


class MBConvBlock(nn.Module):
    """Inverted residual: 1x1 expand (x4) -> KxK depthwise -> 1x1 linear projection."""

    def __init__(self, cin, cout, k, stride, act, expand=4):
        super().__init__()
        hid = cin * expand
        self.op = nn.Sequential(conv_bn(cin, hid, 1, act=act),
                                conv_bn(hid, hid, k, stride, groups=hid, act=act),
                                conv_bn(hid, cout, 1))
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        return x + self.op(x) if self.res else self.op(x)


BLOCKS = {"conv": ConvBlock, "dsconv": DSConvBlock, "mbconv": MBConvBlock}


class GeneratedNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super().__init__()
        self.cfg = cfg
        self.input_resolution = cfg["resolution"]
        first_act = cfg["stages"][0]["activation"]
        layers = [conv_bn(3, cfg["stem_channels"], 3, 1, act=first_act)]
        cin = cfg["stem_channels"]
        for i, st in enumerate(cfg["stages"][:cfg["num_stages"]]):
            for d in range(st["depth"]):
                stride = 2 if (i > 0 and d == 0) else 1
                layers.append(BLOCKS[st["block"]](cin, st["channels"], st["kernel"], stride, st["activation"]))
                cin = st["channels"]
        if cfg["head_channels"]:
            layers.append(conv_bn(cin, cfg["head_channels"], 1, act=cfg["stages"][cfg["num_stages"] - 1]["activation"]))
            cin = cfg["head_channels"]
        self.body = nn.Sequential(*layers)
        self.feature_channels = cin
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(cfg["dropout"]), nn.Linear(cin, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def features(self, x):
        return self.body(x)

    def classifier(self, f):
        return self.head(f)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg, num_classes=10):
    """Public entry point of the generator."""
    return GeneratedNet(cfg, num_classes)
