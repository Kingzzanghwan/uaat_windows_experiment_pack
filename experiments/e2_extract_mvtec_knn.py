from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uaat.image_quality import image_quality_from_path  # noqa: E402
from uaat.utils import ensure_dir, get_device, set_seed  # noqa: E402


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as im:
            x = self.transform(im.convert("RGB"))
        return x, str(path), idx


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        base = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        return torch.nn.functional.normalize(feat, dim=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract UAAT feature CSV from MVTec AD/LOCO using ResNet18 kNN anomaly score.")
    p.add_argument("--root", required=True, help="Dataset root, e.g. data/mvtec_ad or data/mvtec_loco_ad")
    p.add_argument("--out_csv", default="runs/e2_mvtec/features.csv")
    p.add_argument("--categories", default="all", help="all or comma-separated category folder names")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib_frac", type=float, default=0.20)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def list_categories(root: Path, value: str) -> list[str]:
    if value == "all":
        return sorted([p.name for p in root.iterdir() if p.is_dir()])
    return [v.strip() for v in value.split(",") if v.strip()]


def list_train_good(cat_dir: Path) -> list[Path]:
    folder = cat_dir / "train" / "good"
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])


def list_test(cat_dir: Path) -> tuple[list[Path], list[int], list[str]]:
    test_dir = cat_dir / "test"
    paths, labels, defect_types = [], [], []
    for sub in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
        for path in sorted([p for p in sub.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]):
            defect = sub.name
            paths.append(path)
            labels.append(0 if defect == "good" else 1)
            defect_types.append(defect)
    return paths, labels, defect_types


@torch.no_grad()
def extract_features(paths: list[Path], model: nn.Module, transform, device: str, batch_size: int, workers: int) -> np.ndarray:
    ds = ImagePathDataset(paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=(device == "cuda"))
    feats = []
    for x, _path, _idx in tqdm(loader, desc="features", ncols=100):
        x = x.to(device, non_blocking=True)
        feats.append(model(x).detach().cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = Path(args.root)
    out_csv = Path(args.out_csv)
    ensure_dir(out_csv.parent)
    device = "cpu" if args.cpu else get_device()

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model = FeatureExtractor().to(device).eval()
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    categories = list_categories(root, args.categories)
    category_set = set(categories)

    for cat in categories:
        cat_dir = root / cat
        train_paths = list_train_good(cat_dir)
        test_paths, labels, defect_types = list_test(cat_dir)
        if not train_paths or not test_paths:
            print(f"skip {cat}: missing train/test images")
            continue
        print(f"\ncategory={cat} train_good={len(train_paths)} test={len(test_paths)}")
        train_feat = extract_features(train_paths, model, transform, device, args.batch_size, args.workers)
        test_feat = extract_features(test_paths, model, transform, device, args.batch_size, args.workers)

        # Train reference distribution. For each test image, anomaly distance = mean distance to k nearest normal features.
        nn_test = NearestNeighbors(n_neighbors=min(args.k, len(train_feat)), metric="euclidean")
        nn_test.fit(train_feat)
        test_dist, _ = nn_test.kneighbors(test_feat)
        anomaly_dist = test_dist.mean(axis=1)

        # Estimate normal distance scale using leave-neighbor distances in normal train set.
        nn_train = NearestNeighbors(n_neighbors=min(args.k + 1, len(train_feat)), metric="euclidean")
        nn_train.fit(train_feat)
        train_dist, _ = nn_train.kneighbors(train_feat)
        if train_dist.shape[1] > 1:
            ref = train_dist[:, 1:].mean(axis=1)
        else:
            ref = train_dist[:, 0]
        mu = float(ref.mean())
        sigma = float(ref.std() + 1e-6)
        z = (anomaly_dist - mu) / sigma
        # Higher score means safer to auto-pass as normal.
        score = sigmoid(-z)
        uncertainty = np.clip(1.0 - score, 0.0, 1.0)

        # Per-category stratified calibration split.
        labels_arr = np.asarray(labels)
        split = np.array(["test"] * len(test_paths), dtype=object)
        for lab in [0, 1]:
            idx = np.where(labels_arr == lab)[0]
            rng.shuffle(idx)
            n_cal = int(round(len(idx) * args.calib_frac))
            split[idx[:n_cal]] = "calib"

        for i, path in enumerate(test_paths):
            row = {
                "dataset": root.name,
                "category": cat,
                "defect_type": defect_types[i],
                "path": str(path),
                "split": split[i],
                "label_anomaly": int(labels[i]),
                # In this task auto-accept means pass as normal; it is wrong iff actual image is anomalous.
                "error": int(labels[i] == 1),
                "score": float(score[i]),
                "uncertainty": float(uncertainty[i]),
                "anomaly_distance": float(anomaly_dist[i]),
            }
            for c in category_set:
                row[f"ctx_cat_{c}"] = 1.0 if c == cat else 0.0
            row.update(image_quality_from_path(path))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"saved: {out_csv} rows={len(df)} calib={(df.split == 'calib').sum()} test={(df.split == 'test').sum()}")


if __name__ == "__main__":
    main()
