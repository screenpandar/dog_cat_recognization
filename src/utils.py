"""通用工具：随机种子、配置读取、指标、日志。"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = False):
    """固定 python / numpy / torch 的随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int):
    """DataLoader worker 的种子（配合 generator 使用）。"""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


DEFAULTS = {
    # 数据
    "data_root": "datasets",
    "dataset": "dogs_cats",
    "download": False,
    "val_per_class": 500,
    "split_seed": 42,
    "minority_classes": None,
    "minority_count": 450,
    "imbalance_method": "none",
    "num_classes": 2,
    "img_size": 224,
    "augment": True,
    "train_per_class": None,   # None = 全部；例如 1000 → 共 2k 张
    "batch_size": 64,
    "num_workers": 8,
    # 模型
    "model": "resnet18",       # simple_cnn / resnet18 / resnet50
    "pretrained": True,
    "freeze_epochs": 1,        # 前 N 个 epoch 冻结 backbone 只训线性头；-1 = 全程冻结
    # 优化
    "optimizer": "adamw",      # sgd / adamw
    "lr": 1e-3,                # 头部学习率
    "backbone_lr_mult": 0.1,   # 解冻后 backbone lr = lr * mult
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "epochs": 5,
    "scheduler": "cosine",     # cosine / none
    "warmup_epochs": 0,
    "label_smoothing": 0.0,
    "amp": True,
    # 其它
    "seed": 42,
    "deterministic": False,
    "exp_name": None,
    "output_dir": "outputs",
}


def load_config(path=None, overrides=None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    for k, v in (overrides or {}).items():
        if k not in DEFAULTS:
            raise KeyError(f"未知配置项: {k}")
        cfg[k] = v
    text_keys = {"exp_name"} | {k for k, d in DEFAULTS.items() if isinstance(d, str)}
    cfg = {k: v if k in text_keys else _to_number(v) for k, v in cfg.items()}
    if not cfg["exp_name"]:
        cfg["exp_name"] = Path(path).stem if path else cfg["model"]
    return cfg


def _to_number(v):
    """PyYAML 按 YAML 1.1 解析，'5e-4' 这类无小数点的科学计数法会变成字符串，这里转回数字。"""
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
    return v


def parse_overrides(items):
    """命令行 --set key=value，value 用 yaml 解析（自动识别数字/布尔/null）。"""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def get_device():
    """优先 NVIDIA GPU（CUDA），其次 Apple 芯片（MPS），否则 CPU。混合精度只在 CUDA 上开启。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
