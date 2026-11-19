"""
Data utilities for CIFAR-10 / CIFAR-100.

Why a custom loader?
--------------------
On a CPU-only machine the standard torchvision pipeline (PIL image -> transforms
per sample -> DataLoader workers) is a bottleneck. Here the whole dataset is kept
in memory as a uint8 array and augmentation (random-crop with 4px padding and
random horizontal flip) is applied to whole batches as tensor operations.

Data sources (tried in this order):
  1. A cached ``<root>/<name>.npz`` file (created automatically on first use).
  2. HuggingFace parquet files in ``<root>/hf`` (fast mirror of CIFAR).
  3. The torchvision download (official Toronto mirror).
"""
import io
import os

import numpy as np
import torch
import torch.nn.functional as F

# Per-channel statistics computed on the respective training sets.
STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain",
    "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"]
CLASS_NAMES = {"cifar10": CIFAR10_CLASSES, "cifar100": CIFAR100_CLASSES}


# --------------------------------------------------------------------------- #
#  Raw data preparation
# --------------------------------------------------------------------------- #
def _from_parquet(path, label_col):
    """Decode a HuggingFace CIFAR parquet file into (N,32,32,3) uint8 + labels."""
    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(path)
    img_col = "img" if "img" in df.columns else df.columns[0]
    images = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
                       for d in df[img_col]])
    labels = df[label_col].to_numpy().astype(np.int64)
    return images, labels


def _from_torchvision(name, root):
    import torchvision
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    tr = cls(root, train=True, download=True)
    te = cls(root, train=False, download=True)
    return (tr.data, np.array(tr.targets, dtype=np.int64),
            te.data, np.array(te.targets, dtype=np.int64))


def prepare_dataset(name, root="data"):
    """Return (x_train, y_train, x_test, y_test) as numpy arrays (uint8 NHWC images)."""
    cache = os.path.join(root, f"{name}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]

    tag = "c10" if name == "cifar10" else "c100"
    label_col = "label" if name == "cifar10" else "fine_label"
    tr_pq = os.path.join(root, "hf", f"{tag}_train.parquet")
    te_pq = os.path.join(root, "hf", f"{tag}_test.parquet")
    if os.path.exists(tr_pq) and os.path.exists(te_pq):
        print(f"[data] decoding parquet files for {name} ...")
        xtr, ytr = _from_parquet(tr_pq, label_col)
        xte, yte = _from_parquet(te_pq, label_col)
    else:
        print(f"[data] downloading {name} with torchvision ...")
        xtr, ytr, xte, yte = _from_torchvision(name, root)

    os.makedirs(root, exist_ok=True)
    np.savez(cache, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
    print(f"[data] cached {name}: train={xtr.shape} test={xte.shape} -> {cache}")
    return xtr, ytr, xte, yte


# --------------------------------------------------------------------------- #
#  Fast in-memory batch loader
# --------------------------------------------------------------------------- #
class FastLoader:
    """Iterates over an in-memory uint8 dataset and yields normalised float batches.

    Args:
        images: uint8 array (N, H, W, 3)
        labels: int64 array (N,)
        batch_size: mini-batch size
        train: if True -> shuffle + random crop (pad 4) + horizontal flip
        mean, std: normalisation statistics
        resolution: optional output resolution (bilinear resize of the whole batch);
                    used by NAS to search over input resolutions.
    """

    def __init__(self, images, labels, batch_size, train, mean, std,
                 resolution=None, seed=0, drop_last=False):
        self.x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
        self.y = torch.from_numpy(labels.astype(np.int64))
        self.bs = batch_size
        self.train = train
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.resolution = resolution
        self.gen = torch.Generator().manual_seed(seed)
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.y)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def _augment(self, xb):
        b, _, h, w = xb.shape
        padded = F.pad(xb, (4, 4, 4, 4), mode="reflect")
        ox = torch.randint(0, 9, (b,), generator=self.gen)
        oy = torch.randint(0, 9, (b,), generator=self.gen)
        out = torch.empty_like(xb)
        for i in range(b):  # cheap slicing, no copies of the full image
            out[i] = padded[i, :, oy[i]:oy[i] + h, ox[i]:ox[i] + w]
        flip = torch.rand(b, generator=self.gen) < 0.5
        out[flip] = out[flip].flip(3)
        return out

    def __iter__(self):
        n = len(self.y)
        idx = torch.randperm(n, generator=self.gen) if self.train else torch.arange(n)
        for s in range(0, n, self.bs):
            bi = idx[s:s + self.bs]
            if self.drop_last and len(bi) < self.bs:
                break
            xb = self.x[bi].float().div_(255.0)
            if self.train:
                xb = self._augment(xb)
            xb = (xb - self.mean) / self.std
            if self.resolution is not None and self.resolution != xb.shape[-1]:
                xb = F.interpolate(xb, size=(self.resolution, self.resolution),
                                   mode="bilinear", align_corners=False)
            yield xb, self.y[bi]


def get_loaders(name, root="data", batch_size=128, val_size=0, train_subset=None,
                test_subset=None, resolution=None, seed=0):
    """Build train / validation / test loaders.

    val_size     : number of training images held out as a validation split (0 = none)
    train_subset : optional cap on the number of training images (for fast NAS proxies)
    test_subset  : optional cap on the number of test images
    """
    xtr, ytr, xte, yte = prepare_dataset(name, root)
    mean, std = STATS[name]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ytr))
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    if train_subset:
        tr_idx = tr_idx[:train_subset]

    train = FastLoader(xtr[tr_idx], ytr[tr_idx], batch_size, True, mean, std, resolution, seed, drop_last=True)
    val = (FastLoader(xtr[val_idx], ytr[val_idx], 256, False, mean, std, resolution)
           if val_size > 0 else None)
    if test_subset:
        xte, yte = xte[:test_subset], yte[:test_subset]
    test = FastLoader(xte, yte, 256, False, mean, std, resolution)
    return train, val, test


def preprocess_image(path, name, resolution=32):
    """Load an arbitrary image file and convert it to a normalised (1,3,R,R) tensor."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    mean, std = STATS[name]
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
