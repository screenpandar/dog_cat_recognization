"""Make report figures and review candidates from cat/dog prediction artifacts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--test-dir", type=Path, help="Optional original test images for a contact sheet")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    classes = metrics["classes"]
    if classes != ["cat", "dog"]:
        parser.error("Only cat/dog runs are supported")
    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=classes, yticklabels=classes,
           xlabel="Predicted", ylabel="True", title="Validation confusion matrix")
    for (i, j), n in np.ndenumerate(cm):
        ax.text(j, i, str(n), ha="center", va="center", color="white" if n > cm.max() / 2 else "black")
    fig.savefig(out / "val_confusion.png", dpi=160)
    plt.close(fig)
    with np.load(args.run / "val_preds.npz") as val:
        probs, labels, paths = val["probs"], val["labels"], val["paths"]
        pred = probs.argmax(axis=1)
        errors = sorted(np.flatnonzero(labels != pred), key=lambda i: probs[i, pred[i]], reverse=True)
        write_csv(out / "val_errors.csv", ["image_path", "true", "predicted", "p_dog"],
                  [(paths[i], classes[labels[i]], classes[pred[i]], f"{probs[i, 1]:.6f}") for i in errors])
    test_preds = args.run / "test_preds.csv"
    if not test_preds.is_file():
        if args.test_dir:
            parser.error("No test_preds.csv in this run; generate it with predict.py first")
        return
    with test_preds.open(encoding="utf-8", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: abs(float(r["p_dog"]) - 0.5))
    candidates = [("low confidence", r) for r in rows[:8]]
    for label in ("0", "1"):
        candidates.extend(("high confidence", r) for r in [x for x in reversed(rows[8:]) if x["label"] == label][:4])
    write_csv(out / "test_review.csv", ["selection", "id", "prediction", "p_dog", "human_label", "notes"],
              [(group, r["id"], r["label"], r["p_dog"], "", "") for group, r in candidates])
    if args.test_dir:
        fig, axes = plt.subplots(4, 4, figsize=(12, 12), layout="constrained")
        for ax in axes.flat:
            ax.axis("off")
        for ax, (group, r) in zip(axes.flat, candidates):
            with Image.open(args.test_dir / f"{int(r['id'])}.jpg") as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"#{r['id']} | {classes[int(r['label'])]} | P(dog)={float(r['p_dog']):.3f}\n{group}")
        fig.suptitle("Unlabelled test images: predictions for manual review")
        fig.savefig(out / "test_review.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
