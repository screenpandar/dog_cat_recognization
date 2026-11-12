"""test 推理 → submission.csv（id,label；1=dog，0=cat）。

用法：
    python src/predict.py --ckpt outputs/resnet18_ft_s42/best.pt --out submission.csv [--tta]
同时在 ckpt 同目录保存 test_preds.csv（含 dog 概率），供错例分析。
"""
import argparse
import csv
from pathlib import Path

import torch

from data import build_test_loader
from models import build_model
from utils import get_device


@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    ids, probs = [], []
    for x, i in loader:
        x = x.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p = torch.softmax(model(x).float(), 1)
            if tta:  # 水平翻转 TTA
                p = (p + torch.softmax(model(torch.flip(x, dims=[3])).float(), 1)) / 2
        probs.append(p.cpu())
        ids.extend(i.tolist())
    return ids, torch.cat(probs) if probs else torch.empty((0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--data-root", help="Override the dataset location recorded in the checkpoint")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = dict(ck["cfg"])
    if args.data_root:
        cfg["data_root"] = args.data_root
    if ck["classes"] != ["cat", "dog"] or cfg["num_classes"] != 2:
        ap.error("This command expects a binary cat/dog checkpoint")
    model = build_model(cfg["model"], cfg["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])

    ids, probs = predict(model, build_test_loader(cfg), device, args.tta)
    if sorted(ids) != list(range(1, 501)):
        ap.error("Expected exactly one image for each test ID 1–500")
    order = sorted(range(len(ids)), key=lambda k: ids[k])
    labels = probs.argmax(1)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for k in order:
            w.writerow([ids[k], int(labels[k])])
    detail = Path(args.ckpt).parent / "test_preds.csv"
    with open(detail, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "p_dog"])
        for k in order:
            w.writerow([ids[k], int(labels[k]), f"{probs[k, 1].item():.6f}"])
    n_dog = int(labels.sum())
    print(f"wrote {args.out}: {len(ids)} rows, dog={n_dog} cat={len(ids) - n_dog}; detail -> {detail}")


if __name__ == "__main__":
    main()
