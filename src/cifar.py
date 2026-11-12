"""CIFAR-10: fixed validation split and controlled class-imbalance experiments."""

import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from data import IMAGENET_MEAN, IMAGENET_STD, _loader


def split_indices(targets, val_per_class, minority_classes, minority_count, seed):
    """Split before thinning; keep the same validation set across treatments."""
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    train, val = [], []
    for label in range(10):
        ids = rng.permutation(np.flatnonzero(targets == label))
        if not 0 < val_per_class < len(ids):
            raise ValueError("val_per_class must leave at least one training image per class")
        val.extend(ids[:val_per_class].tolist())
        remaining = ids[val_per_class:]
        if label in minority_classes:
            if not 0 < minority_count < len(remaining):
                raise ValueError("minority_count must be smaller than the available training class")
            remaining = remaining[:minority_count]
        train.extend(remaining.tolist())
    return train, val


def build_cifar_loaders(cfg):
    minority = cfg["minority_classes"] or []
    if any(type(c) is not int or not 0 <= c < 10 for c in minority):
        raise ValueError("minority_classes must contain CIFAR-10 class indices 0–9")
    method = cfg["imbalance_method"]
    if method not in ("none", "weighted_loss", "oversample") or (method != "none" and not minority):
        raise ValueError("Choose none, weighted_loss or oversample; treatments require minority_classes")

    size = cfg["img_size"]
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    prep = [transforms.Resize((size, size)), transforms.ToTensor(), normalize]
    eval_tf = transforms.Compose(prep)
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if cfg["augment"] else []
    train_base = datasets.CIFAR10(cfg["data_root"], train=True, transform=transforms.Compose(aug + prep),
                                download=cfg["download"])
    # Use separate transforms on the same underlying training images.
    val_base = datasets.CIFAR10(cfg["data_root"], train=True, transform=eval_tf)
    train_idx, val_idx = split_indices(train_base.targets, cfg["val_per_class"], minority,
                                       cfg["minority_count"], cfg["split_seed"])
    train_ds, val_ds = Subset(train_base, train_idx), Subset(val_base, val_idx)
    labels = np.asarray(train_base.targets)[train_idx]
    counts = np.bincount(labels, minlength=10)
    gen = torch.Generator().manual_seed(cfg["seed"])
    sampler = None
    if method == "oversample":
        sampler = WeightedRandomSampler(1.0 / counts[labels], len(labels), replacement=True,
                                        generator=torch.Generator().manual_seed(cfg["seed"]))
    return dict(
        train=_loader(train_ds, cfg, sampler is None, gen, sampler=sampler),
        val=_loader(val_ds, cfg, False),
        classes=train_base.classes,
        n_train=len(train_ds), n_val=len(val_ds),
        val_samples=[(f"cifar10_train_{i}", val_base.targets[i]) for i in val_idx],
        train_counts=counts.tolist(),
    )


def build_cifar_test_loader(cfg):
    size = cfg["img_size"]
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    test = datasets.CIFAR10(cfg["data_root"], train=False, transform=tf, download=cfg["download"])
    return _loader(test, cfg, False)


def classification_metrics(cm, classes):
    """Macro F1 and class recall expose minority failures hidden by accuracy."""
    tp = cm.diagonal()
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros(len(classes)), where=support != 0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(classes)), where=denom != 0)
    return dict(accuracy=float(tp.sum() / cm.sum()), macro_f1=float(f1.mean()),
                balanced_accuracy=float(recall.mean()),
                per_class_recall=dict(zip(classes, recall.tolist())), confusion_matrix=cm.tolist())
