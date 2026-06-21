from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uaat.cifar_model import CIFAR10_MEAN, CIFAR10_STD, build_cifar_resnet18  # noqa: E402
from uaat.utils import ensure_dir, get_device, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CIFAR-10 ResNet18 base predictor.")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="runs/e1_cifar_base")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(1, total)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cpu" if args.cpu else get_device()
    out_dir = ensure_dir(args.out_dir)

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )

    full_for_indices = datasets.CIFAR10(args.data_dir, train=True, download=True)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(full_for_indices), generator=g).tolist()
    train_idx = perm[:45000]
    val_idx = perm[45000:]

    train_ds = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=train_tf)
    val_ds = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=test_tf)
    test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )

    model = build_cifar_resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val = -1.0
    best_path = out_dir / "cifar10_resnet18.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        loop = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", ncols=100)
        for x, y in loop:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
            loop.set_postfix(loss=total_loss / max(1, total))
        scheduler.step()
        val_acc = accuracy(model, val_loader, device)
        test_acc = accuracy(model, test_loader, device)
        row = {"epoch": epoch, "val_acc": val_acc, "test_acc": test_acc}
        history.append(row)
        print(row)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"saved best checkpoint: {best_path}")

    save_json({"history": history, "best_val": best_val, "checkpoint": str(best_path)}, out_dir / "train_log.json")


if __name__ == "__main__":
    main()
