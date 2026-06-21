from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uaat.cifar_model import CIFAR10_MEAN, CIFAR10_STD, build_cifar_resnet18  # noqa: E402
from uaat.image_quality import batch_quality_from_uint8  # noqa: E402
from uaat.utils import ensure_dir, get_device, set_seed  # noqa: E402

PAPER5 = ["gaussian_noise", "motion_blur", "jpeg_compression", "brightness", "contrast"]


class ArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images.astype(np.uint8)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.labels[idx]), int(idx)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract UAAT feature CSV from CIFAR-10-C.")
    p.add_argument("--cifar10c_dir", required=True, help="Folder containing *.npy and labels.npy")
    p.add_argument("--ckpt", required=True, help="Checkpoint from e1_train_cifar.py")
    p.add_argument("--out_csv", default="runs/e1_cifar10c/features.csv")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib_frac", type=float, default=0.20)
    p.add_argument("--max_per_severity", type=int, default=0, help="0 means use all 10,000 per severity")
    p.add_argument("--corruptions", default="paper5", help="paper5, all, or comma-separated names without .npy")
    p.add_argument("--severities", default="1,2,3,4,5")
    p.add_argument("--tta", type=int, default=4, help="Number of stochastic forward passes. 1 disables instability.")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def normalize_tensor(x: torch.Tensor, device: str) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def make_input(batch_uint8: torch.Tensor, device: str, augment: bool) -> torch.Tensor:
    # batch_uint8: B,H,W,C from DataLoader, uint8
    x = batch_uint8.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    if augment:
        b = x.shape[0]
        # Horizontal flip.
        flip = torch.rand((b,), device=device) < 0.5
        x[flip] = torch.flip(x[flip], dims=[3])
        # Mild brightness jitter.
        factor = 0.90 + 0.20 * torch.rand((b, 1, 1, 1), device=device)
        x = torch.clamp(x * factor, 0.0, 1.0)
    return normalize_tensor(x, device)


@torch.no_grad()
def predict_batch(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, device: str, tta: int) -> dict[str, np.ndarray]:
    probs_list = []
    tta = max(1, int(tta))
    for t in range(tta):
        x = make_input(images, device=device, augment=(t > 0))
        logits = model(x)
        probs_list.append(F.softmax(logits, dim=1))
    probs = torch.stack(probs_list, dim=0)  # T,B,K
    mean_probs = probs.mean(dim=0)
    score, pred = mean_probs.max(dim=1)
    eps = 1e-8
    entropy = -(mean_probs.clamp_min(eps) * mean_probs.clamp_min(eps).log()).sum(dim=1) / math.log(mean_probs.shape[1])
    instability = probs.var(dim=0).mean(dim=1)
    uncertainty = torch.clamp(entropy + 10.0 * instability, 0.0, 1.0)
    error = (pred.cpu() != labels.cpu()).long()
    return {
        "score": score.detach().cpu().numpy(),
        "uncertainty": uncertainty.detach().cpu().numpy(),
        "pred": pred.detach().cpu().numpy(),
        "error": error.numpy(),
    }


def resolve_corruptions(root: Path, value: str) -> list[str]:
    if value == "paper5":
        return PAPER5
    if value == "all":
        return sorted([p.stem for p in root.glob("*.npy") if p.stem != "labels"])
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = Path(args.cifar10c_dir)
    out_csv = Path(args.out_csv)
    ensure_dir(out_csv.parent)
    device = "cpu" if args.cpu else get_device()

    labels_path = root / "labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.npy not found in {root}")
    labels_all = np.load(labels_path)
    corruptions = resolve_corruptions(root, args.corruptions)
    severities = [int(x) for x in args.severities.split(",")]

    model = build_cifar_resnet18(num_classes=10).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    corruption_set = set(corruptions)

    for corr in corruptions:
        npy_path = root / f"{corr}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Missing corruption file: {npy_path}")
        arr = np.load(npy_path)
        if len(arr) < 50000:
            raise ValueError(f"Expected 50,000 images in {npy_path}, got {len(arr)}")
        for sev in severities:
            start, end = (sev - 1) * 10000, sev * 10000
            images = arr[start:end]
            if len(labels_all) == 10000:
                labels = labels_all.copy()
            elif len(labels_all) >= end:
                labels = labels_all[start:end]
            else:
                raise ValueError("labels.npy must have length 10,000 or 50,000")

            indices = np.arange(len(images))
            rng.shuffle(indices)
            if args.max_per_severity and args.max_per_severity > 0:
                indices = indices[: args.max_per_severity]
            split_boundary = int(round(len(indices) * args.calib_frac))
            calib_ids = set(indices[:split_boundary].tolist())
            use_images = images[indices]
            use_labels = labels[indices]
            ds = ArrayDataset(use_images, use_labels)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

            for batch_images, batch_labels, batch_local_idx in tqdm(
                loader, desc=f"{corr} severity {sev}", ncols=100
            ):
                pred_info = predict_batch(model, batch_images, batch_labels, device=device, tta=args.tta)
                quality = batch_quality_from_uint8(batch_images.numpy())
                for j in range(len(batch_labels)):
                    original_idx = int(indices[int(batch_local_idx[j])])
                    row = {
                        "dataset": "cifar10c",
                        "corruption": corr,
                        "severity": sev,
                        "sample_index": original_idx,
                        "split": "calib" if original_idx in calib_ids else "test",
                        "label": int(batch_labels[j]),
                        "pred": int(pred_info["pred"][j]),
                        "score": float(pred_info["score"][j]),
                        "uncertainty": float(pred_info["uncertainty"][j]),
                        "error": int(pred_info["error"][j]),
                        "ctx_severity": (sev - 1) / 4.0,
                    }
                    for c in corruption_set:
                        row[f"ctx_corr_{c}"] = 1.0 if c == corr else 0.0
                    row.update(quality[j])
                    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"saved: {out_csv} rows={len(df)} calib={(df.split == 'calib').sum()} test={(df.split == 'test').sum()}")


if __name__ == "__main__":
    main()
