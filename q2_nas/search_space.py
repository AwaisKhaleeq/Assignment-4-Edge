"""
Searchable CNN architecture space for edge devices.

An architecture ("genotype") is a JSON-serialisable dict:

    {
      "resolution": 32,                 # input resolution     {24, 28, 32}
      "stem_channels": 24,              # stem width           {16, 24, 32}
      "num_stages": 3,                  # convolutional depth  {2, 3, 4} stages ...
      "stages": [                       # ... each with a variable number of blocks
          {"depth": 2,                  # blocks in the stage  {1, 2, 3}
           "channels": 32,              # filters              {16, 24, 32, 48, 64, 96, 128}
           "kernel": 3,                 # kernel size          {3, 5, 7}
           "block": "dsconv",           # block type           {conv, dsconv, mbconv}
           "activation": "hswish"},     # activation           {relu, relu6, hswish, silu, leaky_relu}
          ...                           # always MAX_STAGES genes; only the first num_stages are expressed
      ],
      "head_channels": 128,             # 1x1 conv before pooling (0 = none)
      "dropout": 0.1
    }

Block types
    conv   : standard KxK convolution
    dsconv : depthwise-separable convolution (KxK depthwise + 1x1 pointwise)
    mbconv : inverted residual (1x1 expand x4 -> KxK depthwise -> 1x1 linear projection)

Keeping MAX_STAGES genes and only *expressing* the first `num_stages` means that a
mutation of `num_stages` re-activates previously hidden (but still meaningful) genes,
which makes evolutionary search smoother.
"""
import copy
import json
import random

MAX_STAGES = 4

SPACE = {
    "resolution": [24, 28, 32],
    "stem_channels": [16, 24, 32],
    "num_stages": [2, 3, 4],
    "depth": [1, 2, 3],
    "channels": [16, 24, 32, 48, 64, 96, 128],
    "kernel": [3, 5, 7],
    "block": ["conv", "dsconv", "mbconv"],
    "activation": ["relu", "relu6", "hswish", "silu", "leaky_relu"],
    "head_channels": [0, 128, 256],
    "dropout": [0.0, 0.1, 0.2],
}
STAGE_KEYS = ["depth", "channels", "kernel", "block", "activation"]
GLOBAL_KEYS = ["resolution", "stem_channels", "num_stages", "head_channels", "dropout"]


def _sort_channels(cfg):
    """Channels are made non-decreasing across stages (standard CNN design prior)."""
    chans = sorted(s["channels"] for s in cfg["stages"])
    for s, c in zip(cfg["stages"], chans):
        s["channels"] = c
    return cfg


def sample_stage(rng):
    return {k: rng.choice(SPACE[k]) for k in STAGE_KEYS}


def sample_config(rng=random):
    """Uniformly sample a random architecture from the search space."""
    cfg = {k: rng.choice(SPACE[k]) for k in GLOBAL_KEYS}
    cfg["stages"] = [sample_stage(rng) for _ in range(MAX_STAGES)]
    return _sort_channels(cfg)


def mutate(cfg, rng=random):
    """Return a mutated copy of `cfg` (exactly one gene changes) and a description of the change."""
    child = copy.deepcopy(cfg)
    while True:
        if rng.random() < 0.3:                                  # mutate a global gene
            key = rng.choice(GLOBAL_KEYS)
            choices = [v for v in SPACE[key] if v != child[key]]
            old, child[key] = child[key], rng.choice(choices)
            desc = f"{key}: {old} -> {child[key]}"
        else:                                                   # mutate a gene of an expressed stage
            i = rng.randrange(child["num_stages"])
            key = rng.choice(STAGE_KEYS)
            choices = [v for v in SPACE[key] if v != child["stages"][i][key]]
            old = child["stages"][i][key]
            child["stages"][i][key] = rng.choice(choices)
            desc = f"stage{i}.{key}: {old} -> {child['stages'][i][key]}"
        _sort_channels(child)
        if arch_key(child) != arch_key(cfg):                    # make sure the phenotype changed
            return child, desc


def expressed(cfg):
    """The part of the genotype that actually defines the network (hidden genes removed)."""
    e = {k: cfg[k] for k in GLOBAL_KEYS}
    e["stages"] = cfg["stages"][:cfg["num_stages"]]
    return e


def arch_key(cfg):
    """Canonical string used to detect duplicate architectures."""
    return json.dumps(expressed(cfg), sort_keys=True)


def describe(cfg):
    """Compact human-readable summary, e.g. 'r32 s24 | 2x32 k3 dsconv hswish | ... | h128'."""
    parts = [f"r{cfg['resolution']} s{cfg['stem_channels']}"]
    for s in cfg["stages"][:cfg["num_stages"]]:
        parts.append(f"{s['depth']}x{s['channels']} k{s['kernel']} {s['block']} {s['activation']}")
    parts.append(f"h{cfg['head_channels']} d{cfg['dropout']}")
    return " | ".join(parts)


def space_size():
    stage = 1
    for k in STAGE_KEYS:
        stage *= len(SPACE[k])
    glob = len(SPACE["resolution"]) * len(SPACE["stem_channels"]) * len(SPACE["head_channels"]) * len(SPACE["dropout"])
    return glob * sum(stage ** n for n in SPACE["num_stages"])
