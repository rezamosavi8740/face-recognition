"""
Evaluation & comparison script for CVLFace AdaFace models
========================================================

Place this file here:
/home/user1/face-recognition/codes/CVLface/cvlface/research/recognition/code/run_v1/eval_compare.py

Run it with e.g.:
    python eval_compare.py \
        --data_root /home/user1/newdata/eval_data \
        --model_pre /home/user1/face-recognition/codes/CVLface/cvlface/pretrained_models/adaface_ir101_webface12m/model.pt \
        --model_ft  /home/user3/face-recognition/codes/CVLface/cvlface/research/recognition/experiments/run_v1/default_05-22_0||/model.pt  \
        --batch_size 64

What it does
------------
1. Loads the evaluation dataset using the *same* transforms and class-index mapping that TRAIN.PY relies on (imported below).
2. Evaluates the **pre-trained** model – records softmax confidence, predicted label, and correctness for every sample.
3. Picks the 5 *best* correctly-classified samples (highest confidence) and the 10 *worst* mistakes (highest confidence but wrong).
4. Saves a PNG grid for the good samples and one for the bad samples.
5. Feeds only the 10 bad samples to the **fine-tuned** model, records its predictions, and saves a side-by-side comparison grid (pre vs. fine-tuned) plus a small CSV with the raw numbers.

All outputs are written to ./eval_results/YYYY-MM-DD_HH-MM-SS/ so they never collide with earlier runs.

Dependencies
------------
Assumes you already created the CVLFace conda/venv with Torch, torchvision, PIL, matplotlib, pandas, and (of course) your repo itself in PYTHONPATH.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, utils as vutils
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------
# Import helpers from the repo so we share the exact preprocessing pipeline that
# TRAIN.PY uses.  The relative path makes the script position-independent as
# long as it lives somewhere inside `cvlface/research/recognition`.
# -----------------------------------------------------------------------------
try:
    from ..train import build_transforms  # type: ignore
except ImportError:
    # Fallback: TRAIN.PY may keep transforms in a utils module.
    try:
        from ..utils.transforms import build_transforms  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Cannot import TRAIN.PY transforms. Adjust the import in\n"
            "eval_compare.py to point at whatever build_transforms(...) function\n"
            "or transform config your repo actually exposes."
        )

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def load_model(model_path: Path | str, device: torch.device):
    """Load an AdaFace model checkpoint *without* changing its state-dict names.
    The repo's `AdaFace.load_model()` helper already does the heavy lifting;
    we only need to supply the file-system path.
    """
    from adaface.models import get_model  # repo helper – adjust if path differs

    ckpt = torch.load(model_path, map_location="cpu")
    # ckpt is assumed to be either (1) plain state_dict or (2) dict with key
    # "state_dict" – TRAIN.PY usually saves the latter.
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Architecture name should be stored inside ckpt*; fallback to ir_101.
    arch = ckpt.get("arch", "ir_101") if isinstance(ckpt, dict) else "ir_101"

    model = get_model(arch, fp16=False)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def prepare_dataloader(data_root: Path | str, batch_size: int):
    """Use exactly the transforms TRAIN.PY applies for val/test."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(root)

    normalize, _ = build_transforms()  # returns (test_tfms, train_tfms)

    dataset = datasets.ImageFolder(root, transform=normalize)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader, dataset.classes


def evaluate(model, loader, device):
    """Return list[dict] with path, gt, pred, conf, correct."""
    results = []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="eval", leave=False):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = softmax(logits)
            confs, preds = probs.max(dim=1)

            for i in range(len(imgs)):
                results.append(
                    {
                        "idx": len(results),
                        "path": loader.dataset.samples[i + results[0]["idx"] if results else i][0],
                        "gt": int(labels[i]),
                        "pred": int(preds[i]),
                        "conf": float(confs[i]),
                        "correct": bool(preds[i] == labels[i]),
                    }
                )
    return results


def topk_samples(results, k_good=5, k_bad=10):
    """Pick the *k_good* most confident correct predictions and the *k_bad*
    most confident *wrong* predictions.
    """
    correct = [r for r in results if r["correct"]]
    wrong = [r for r in results if not r["correct"]]

    best = sorted(correct, key=lambda r: -r["conf"])[:k_good]
    worst = sorted(wrong, key=lambda r: -r["conf"])[:k_bad]
    return best, worst


def make_grid(sample_list, out_path: Path, title: str, class_names):
    imgs = [Image.open(r["path"]).convert("RGB") for r in sample_list]

    tfms = transforms.Compose([
        transforms.Resize((224, 224)),  # small grid only for visualisation
        transforms.ToTensor(),
    ])
    tensor_stack = torch.stack([tfms(img) for img in imgs])
    grid = vutils.make_grid(tensor_stack, nrow=len(sample_list), padding=4)
    plt.figure(figsize=(3 * len(sample_list), 3))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_csv(rows, path: Path):
    keys = rows[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser("Compare pre-trained vs. fine-tuned AdaFace models on your eval set")
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--model_pre", required=True, type=Path)
    parser.add_argument("--model_ft", required=True, type=Path)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = Path("./eval_results") / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1) Dataset
    # ---------------------------------------------------------------------
    loader, class_names = prepare_dataloader(args.data_root, args.batch_size)

    # ---------------------------------------------------------------------
    # 2) Pre-trained evaluation
    # ---------------------------------------------------------------------
    device = torch.device(args.device)
    print("\n>>> Evaluating PRE-TRAINED model …")
    model_pre = load_model(args.model_pre, device)
    res_pre = evaluate(model_pre, loader, device)
    save_csv(res_pre, outdir / "results_pre.csv")

    best, worst = topk_samples(res_pre)

    make_grid(best, outdir / "good_pre.png", "5 best - PRE", class_names)
    make_grid(worst, outdir / "bad_pre.png", "10 worst - PRE", class_names)

    print(f"Saved good/bad sample grids to {outdir}")

    # ---------------------------------------------------------------------
    # 3) Fine-tuned model on bad samples only
    # ---------------------------------------------------------------------
    print("\n>>> Re-evaluating the 10 *bad* samples with FINE-TUNED model …")
    model_ft = load_model(args.model_ft, device)

    # build a mini loader only for the worst samples
    tfms_test, _ = build_transforms()
    mini_imgs = [tfms_test(Image.open(r["path"]).convert("RGB")) for r in worst]
    mini_batch = torch.stack(mini_imgs).to(device)

    with torch.no_grad():
        logits = model_ft(mini_batch)
        probs = F.softmax(logits, dim=1)
        conf_ft, preds_ft = probs.max(dim=1)

    for i, r in enumerate(worst):
        r["pred_ft"] = int(preds_ft[i])
        r["conf_ft"] = float(conf_ft[i])
        r["correct_ft"] = int(preds_ft[i]) == r["gt"]

    # ------------------------------------------------------------------
    # 4) Visualise comparison grid (pre vs ft)
    # ------------------------------------------------------------------
    from PIL import Image

    side_by_side = []
    label_tmpl = "GT:{gt}\nPRE:{pred}({conf:.2f})\nFT :{pred_ft}({conf_ft:.2f})"

    for r in worst:
        img = Image.open(r["path"]).convert("RGB").resize((224, 224))
        caption = label_tmpl.format(
            gt=class_names[r["gt"]],
            pred=class_names[r["pred"]],
            conf=r["conf"],
            pred_ft=class_names[r["pred_ft"]],
            conf_ft=r["conf_ft"],
        )
        # put caption underneath using matplotlib
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(caption, fontsize=6)
        fig.canvas.draw()
        # convert figure to numpy array
        fig_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        fig_arr = fig_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        side_by_side.append(transforms.ToTensor()(Image.fromarray(fig_arr)))
        plt.close(fig)

    grid_cmp = vutils.make_grid(side_by_side, nrow=5, padding=4)
    plt.figure(figsize=(20, 8))
    plt.imshow(grid_cmp.permute(1, 2, 0))
    plt.axis("off")
    plt.title("10 worst samples – PRE vs. FINE-TUNED")
    plt.tight_layout()
    cmp_path = outdir / "compare_pre_ft.png"
    plt.savefig(cmp_path)
    plt.close()

    save_csv(worst, outdir / "results_cmp.csv")

    print(f"All done – comparison figure saved to {cmp_path}")


if __name__ == "__main__":
    main()
