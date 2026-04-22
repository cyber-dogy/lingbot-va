from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from lingbot_va.config import ExperimentConfig
from lingbot_va.data.registry import build_dataset as build_dataset_from_registry
from lingbot_va.model.registry import build_backbone, build_head, build_text_encoder
from lingbot_va.policy.registry import build_policy as build_policy_from_registry


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(split: str, cfg: ExperimentConfig):
    return build_dataset_from_registry(split, cfg)


def build_dataloaders(cfg: ExperimentConfig):
    dataset_train = build_dataset("train", cfg)
    dataset_valid = build_dataset("valid", cfg)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=bool(cfg.num_workers > 0),
    )
    dataloader_valid = DataLoader(
        dataset_valid,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        persistent_workers=bool(cfg.num_workers > 0),
    )
    return dataset_train, dataset_valid, dataloader_train, dataloader_valid


def build_model(cfg: ExperimentConfig):
    text_encoder = build_text_encoder(cfg)
    backbone = build_backbone(cfg)
    head = build_head(cfg)
    return build_policy_from_registry(
        cfg=cfg,
        text_encoder=text_encoder,
        backbone=backbone,
        head=head,
    )


def build_optimizer(model, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    return model.get_optimizer(cfg)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
    train_loader_len: int,
):
    total_training_steps = max(
        1,
        (train_loader_len * cfg.train_epochs) // max(1, cfg.grad_accum_steps),
    )

    def lr_lambda(current_step: int) -> float:
        if current_step < cfg.lr_warmup_steps:
            return float(current_step) / float(max(1, cfg.lr_warmup_steps))
        progress = float(current_step - cfg.lr_warmup_steps) / float(
            max(1, total_training_steps - cfg.lr_warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    if cfg.lr_scheduler_name != "cosine":
        raise ValueError(f"不支持的 lr_scheduler_name: {cfg.lr_scheduler_name}")
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
