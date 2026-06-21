from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uaat.utils import ensure_dir, get_device, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/extract UAAT features for iWildCam-WILDS.")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="runs/e3_iwildcam")
    p.add_argument("--out_csv", default="runs/e3_iwildcam/features.csv")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib_split", default="auto", help="auto, id_val, val, or test")
    p.add_argument("--test_split", default="auto", help="auto, test, val, or id_val")
    p.add_argument("--max_train_batches", type=int, default=0, help="0 means full train split")
    p.add_argument("--max_extract", type=int, default=0, help="0 means full split extraction")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_num_classes(dataset: Any) -> int:
    for attr in ["n_classes", "num_classes"]:
        if hasattr(dataset, attr):
            return int(getattr(dataset, attr))
    y = getattr(dataset, "y_array", None)
    if y is not None:
        return int(torch.as_tensor(y).max().item() + 1)
    raise ValueError("Could not infer number of classes from WILDS dataset.")


def split_exists(dataset: Any, split: str) -> bool:
    return hasattr(dataset, "split_dict") and split in dataset.split_dict


def choose_split(dataset: Any, requested: str, candidates: list[str]) -> str:
    if requested != "auto":
        if not split_exists(dataset, requested):
            raise ValueError(f"Requested split '{requested}' not in dataset.split_dict={dataset.split_dict}")
        return requested
    for c in candidates:
        if split_exists(dataset, c):
            return c
    raise ValueError(f"None of candidate splits exist: {candidates}; available={dataset.split_dict}")


def get_loader(subset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)


def extract_y_meta(batch: tuple) -> tuple[torch.Tensor, Any]:
    # WILDS subsets usually return (x, y, metadata). Some wrappers return (x, y).
    if len(batch) >= 3:
        return batch[1], batch[2]
    return batch[1], None


def metadata_to_context(metadata: Any, metadata_fields: list[str] | None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if metadata is None or metadata_fields is None:
        return out
    meta = torch.as_tensor(metadata).detach().cpu().numpy()
    if meta.ndim == 1:
        meta = meta.reshape(-1, 1)
    for j, name in enumerate(metadata_fields[: min(len(metadata_fields), meta.shape[1], 6)]):
        values = meta[:, j].astype(np.float32)
        # Stable numeric compression. Exact one-hot can be too large for camera IDs.
        denom = float(np.nanmax(np.abs(values)) + 1.0)
        out[f"ctx_meta_{name}"] = values / denom
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    out_csv = Path(args.out_csv)
    ensure_dir(out_csv.parent)
    device = "cpu" if args.cpu else get_device()

    from wilds import get_dataset

    dataset = get_dataset(dataset="iwildcam", root_dir=args.data_dir, download=True)
    print(f"WILDS split_dict: {dataset.split_dict}")
    num_classes = get_num_classes(dataset)
    calib_split = choose_split(dataset, args.calib_split, ["id_val", "val"])
    test_split = choose_split(dataset, args.test_split, ["test", "val"])
    metadata_fields = list(getattr(dataset, "metadata_fields", []))

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_data = dataset.get_subset("train", transform=train_tf)
    calib_data = dataset.get_subset(calib_split, transform=test_tf)
    test_data = dataset.get_subset(test_split, transform=test_tf)

    train_loader = get_loader(train_data, args.batch_size, args.workers, shuffle=True)
    calib_loader = get_loader(calib_data, args.batch_size, args.workers, shuffle=False)
    test_loader = get_loader(test_data, args.batch_size, args.workers, shuffle=False)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for bi, batch in enumerate(tqdm(train_loader, desc=f"iwildcam epoch {epoch}/{args.epochs}", ncols=100)):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            x = batch[0].to(device, non_blocking=True)
            y, _meta = extract_y_meta(batch)
            y = y.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
            correct += int((logits.argmax(dim=1) == y).sum().item())
        print({"epoch": epoch, "train_loss": total_loss / max(1, total), "train_acc": correct / max(1, total)})

    ckpt_path = out_dir / "iwildcam_resnet18.pt"
    torch.save({"model": model.state_dict(), "num_classes": num_classes, "args": vars(args)}, ckpt_path)

    @torch.no_grad()
    def extract_split(loader: DataLoader, split_name: str) -> list[dict]:
        model.eval()
        rows: list[dict] = []
        seen = 0
        for batch in tqdm(loader, desc=f"extract {split_name}", ncols=100):
            x = batch[0].to(device, non_blocking=True)
            y, metadata = extract_y_meta(batch)
            y = y.to(device, non_blocking=True).long()
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            score, pred = probs.max(dim=1)
            eps = 1e-8
            entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1) / math.log(num_classes)
            # Context: simple confidence uncertainty plus compressed metadata.
            uncertainty = torch.clamp(entropy + (1.0 - score) * 0.25, 0.0, 1.0)
            meta_context = metadata_to_context(metadata, metadata_fields)
            bsz = int(y.numel())
            for i in range(bsz):
                row = {
                    "dataset": "iwildcam_wilds",
                    "split": "calib" if split_name == calib_split else "test",
                    "wilds_split": split_name,
                    "sample_index": seen + i,
                    "label": int(y[i].detach().cpu()),
                    "pred": int(pred[i].detach().cpu()),
                    "score": float(score[i].detach().cpu()),
                    "uncertainty": float(uncertainty[i].detach().cpu()),
                    "error": int(pred[i].detach().cpu() != y[i].detach().cpu()),
                }
                for key, values in meta_context.items():
                    row[key] = float(values[i])
                rows.append(row)
            seen += bsz
            if args.max_extract and seen >= args.max_extract:
                break
        return rows

    rows = []
    rows += extract_split(calib_loader, calib_split)
    rows += extract_split(test_loader, test_split)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    save_json(
        {
            "checkpoint": str(ckpt_path),
            "csv": str(out_csv),
            "num_classes": num_classes,
            "split_dict": dataset.split_dict,
            "metadata_fields": metadata_fields,
            "calib_split": calib_split,
            "test_split": test_split,
        },
        out_dir / "run_config.json",
    )
    print(f"saved: {out_csv} rows={len(df)} calib={(df.split == 'calib').sum()} test={(df.split == 'test').sum()}")


if __name__ == "__main__":
    main()
