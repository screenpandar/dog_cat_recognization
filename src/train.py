"""训练 + 验证。

用法（在 dog_vs_cat/ 目录下）：
    python src/train.py --config configs/resnet18_ft.yaml
    python src/train.py --config configs/resnet18_ft.yaml --set seed=1 lr=1e-4 epochs=3

输出 outputs/<exp_name>_s<seed>/：
    config.yaml  history.csv  curves.png  best.pt  metrics.json  val_preds.npz
"""
import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data import build_loaders
from cifar import build_cifar_loaders, build_cifar_test_loader, classification_metrics
from models import build_model, count_params, param_groups, set_backbone_trainable
from utils import confusion_matrix, get_device, load_config, parse_overrides, save_json, set_seed


def build_optimizer(model, cfg):
    groups = param_groups(model, cfg["lr"], cfg["backbone_lr_mult"])
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(groups, momentum=cfg["momentum"], weight_decay=cfg["weight_decay"], nesterov=True)
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    raise ValueError(cfg["optimizer"])


def build_scheduler(opt, cfg, steps_per_epoch):
    total = cfg["epochs"] * steps_per_epoch
    warm = cfg["warmup_epochs"] * steps_per_epoch

    def f(step):
        if warm and step < warm:
            return (step + 1) / warm
        if cfg["scheduler"] == "cosine":
            return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def run_epoch(model, loader, criterion, device, opt=None, sched=None, scaler=None, amp=True, frozen=False, desc=""):
    train = opt is not None
    if train:
        model.train()
        if frozen:  # 冻结阶段：backbone 保持 eval（BN 统计量不更新），只训练头
            model.eval()
            model.fc.train()
    else:
        model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    all_probs, all_labels = [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc=desc, leave=False, ncols=90):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            tot_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
            if not train:
                all_probs.append(torch.softmax(logits.float(), 1).cpu())
                all_labels.append(y.cpu())
    out = {"loss": tot_loss / n, "acc": correct / n}
    if not train:
        out["probs"] = torch.cat(all_probs).numpy()
        out["labels"] = torch.cat(all_labels).numpy()
    return out


def plot_curves(hist, path):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], "o-", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], "o-", label="val")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[1].plot(ep, [h["train_acc"] for h in hist], "o-", label="train")
    ax[1].plot(ep, [h["val_acc"] for h in hist], "o-", label="val")
    ax[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="覆盖配置，如 lr=1e-4 seed=1")
    ap.add_argument("--eval-test", action="store_true", help="Evaluate the selected CIFAR-10 checkpoint on test")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    if cfg["dataset"] not in ("dogs_cats", "cifar10"):
        ap.error("dataset must be dogs_cats or cifar10")
    is_cifar = cfg["dataset"] == "cifar10"
    if cfg["num_classes"] != (10 if is_cifar else 2):
        ap.error("num_classes does not match the selected dataset")
    if args.eval_test and not is_cifar:
        ap.error("--eval-test is for CIFAR-10; use predict.py for the unlabelled cat/dog test set")

    set_seed(cfg["seed"], cfg["deterministic"])
    device = get_device()
    out_dir = Path(cfg["output_dir"]) / f"{cfg['exp_name']}_s{cfg['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    data = build_cifar_loaders(cfg) if is_cifar else build_loaders(cfg)
    print(f"[{cfg['exp_name']}] device={device} classes={data['classes']} "
          f"train={data['n_train']} val={data['n_val']}")

    model = build_model(cfg["model"], cfg["num_classes"], cfg["pretrained"]).to(device)
    model = model.to(memory_format=torch.channels_last)
    print(f"model={cfg['model']} params={count_params(model) / 1e6:.2f}M")

    eval_criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    criterion = eval_criterion
    if is_cifar and cfg["imbalance_method"] == "weighted_loss":
        counts = torch.tensor(data["train_counts"], dtype=torch.float32, device=device)
        weights = counts.sum() / (len(counts) * counts)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["label_smoothing"])
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, len(data["train"]))
    scaler = torch.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")

    hist, best_acc, best_epoch = [], -1.0, -1
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        fe = cfg["freeze_epochs"]
        frozen = fe == -1 or epoch <= fe
        set_backbone_trainable(model, not frozen)
        te = time.time()
        tr = run_epoch(model, data["train"], criterion, device, opt, sched, scaler, cfg["amp"], frozen,
                       desc=f"ep{epoch} train")
        va = run_epoch(model, data["val"], eval_criterion, device, amp=cfg["amp"], desc=f"ep{epoch} val")
        row = {"epoch": epoch, "frozen": frozen, "lr": opt.param_groups[0]["lr"],
               "train_loss": tr["loss"], "train_acc": tr["acc"],
               "val_loss": va["loss"], "val_acc": va["acc"], "time_s": time.time() - te}
        hist.append(row)
        print(f"ep{epoch:02d} {'[frozen] ' if frozen else ''}train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.4f} | {row['time_s']:.0f}s")
        if va["acc"] > best_acc:
            best_acc, best_epoch = va["acc"], epoch
            torch.save({"model": model.state_dict(), "cfg": cfg, "classes": data["classes"], "epoch": epoch},
                        out_dir / "best.pt")
            np.savez(out_dir / "val_preds.npz", probs=va["probs"], labels=va["labels"],
                     paths=np.array([p for p, _ in data["val_samples"]]))

    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    plot_curves(hist, out_dir / "curves.png")

    vp = np.load(out_dir / "val_preds.npz")
    cm = confusion_matrix(vp["labels"], vp["probs"].argmax(1), cfg["num_classes"])
    metrics = {"exp_name": cfg["exp_name"], "seed": cfg["seed"], "model": cfg["model"],
               "n_train": data["n_train"], "n_val": data["n_val"],
               "best_val_acc": best_acc, "best_epoch": best_epoch,
               "final_val_acc": hist[-1]["val_acc"], "confusion_matrix": cm.tolist(),
               "classes": data["classes"], "train_time_min": (time.time() - t0) / 60}
    if is_cifar:
        metrics.update(dataset="cifar10", train_counts=dict(zip(data["classes"], data["train_counts"])),
                       imbalance_method=cfg["imbalance_method"],
                       validation=classification_metrics(cm, data["classes"]))
        if args.eval_test:
            checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            test = run_epoch(model, build_cifar_test_loader(cfg), eval_criterion, device,
                             amp=cfg["amp"], desc="test")
            test_cm = confusion_matrix(test["labels"], test["probs"].argmax(1), 10)
            metrics["test"] = classification_metrics(test_cm, data["classes"])
            metrics["test"]["n"] = len(test["labels"])
    save_json(metrics, out_dir / "metrics.json")
    print(f"best val acc {best_acc:.4f} @ epoch {best_epoch}; saved to {out_dir}")


if __name__ == "__main__":
    main()
