"""数据：预处理/增强、Dataset、DataLoader。

猫狗数据接口；CIFAR-10 数据加载见 cifar.py：
    build_transforms(img_size, augment) -> (train_tf, eval_tf)
    build_loaders(cfg) -> dict(train=..., val=..., classes=[...], n_train=..., n_val=...)
    build_test_loader(cfg) -> DataLoader，返回 (img, id)
猫狗数据用 ImageFolder，类别按字母序：cat=0, dog=1（与提交格式一致）。
"""
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from utils import seed_worker

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size=224, augment=True):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    # 验证/测试：短边缩放到 img_size*256/224 后中心裁剪（ImageNet 标准做法）
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def subsample_per_class(targets, per_class, seed):
    """每类随机取 per_class 个样本的索引（分层子采样）。"""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    idx = []
    for c in np.unique(targets):
        ci = np.where(targets == c)[0]
        idx.extend(rng.choice(ci, size=min(per_class, len(ci)), replace=False).tolist())
    return sorted(idx)


class TestFolder(Dataset):
    """无标签测试集：test/1.jpg ~ test/500.jpg，按数字 id 排序，返回 (img, id)。"""

    def __init__(self, root, transform):
        self.files = sorted(Path(root).glob("*.jpg"), key=lambda p: int(re.sub(r"\D", "", p.stem)))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        img = Image.open(p).convert("RGB")
        return self.transform(img), int(p.stem)


def _loader(ds, cfg, shuffle, gen=None, sampler=None):
    nw = cfg["num_workers"]
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=nw,
                      pin_memory=True, persistent_workers=nw > 0, worker_init_fn=seed_worker,
                      generator=gen, sampler=sampler, drop_last=False)


def build_loaders(cfg):
    root = Path(cfg["data_root"])
    train_tf, eval_tf = build_transforms(cfg["img_size"], cfg["augment"])
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
    assert train_ds.class_to_idx == val_ds.class_to_idx
    if cfg.get("train_per_class"):
        train_ds = Subset(train_ds, subsample_per_class(train_ds.targets, cfg["train_per_class"], cfg["seed"]))
    gen = torch.Generator().manual_seed(cfg["seed"])
    return {
        "train": _loader(train_ds, cfg, shuffle=True, gen=gen),
        "val": _loader(val_ds, cfg, shuffle=False),
        "classes": val_ds.classes,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "val_samples": val_ds.samples,  # [(path, label)]，供错例分析
    }


def build_test_loader(cfg):
    _, eval_tf = build_transforms(cfg["img_size"], augment=False)
    ds = TestFolder(Path(cfg["data_root"]) / "test", eval_tf)
    return _loader(ds, cfg, shuffle=False)
