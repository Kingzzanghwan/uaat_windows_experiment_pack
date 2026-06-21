from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


@torch.no_grad()
def softmax_entropy(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    ent = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1)
    return ent
