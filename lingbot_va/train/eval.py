from __future__ import annotations

import json
from pathlib import Path

import torch

from .builders import move_batch_to_device


@torch.no_grad()
def evaluate_model_on_loader(model, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_latent_loss = 0.0
    total_action_loss = 0.0
    num_batches = 0

    for batch_cpu in dataloader:
        batch = move_batch_to_device(batch_cpu, device=device)
        loss_dict = model.compute_loss_dict(batch)
        total_loss += float(loss_dict["loss_total"].detach().cpu())
        total_latent_loss += float(loss_dict["loss_latent"].detach().cpu())
        total_action_loss += float(loss_dict["loss_action"].detach().cpu())
        num_batches += 1

    if num_batches == 0:
        raise ValueError("评估 dataloader 为空，无法计算指标。")
    return {
        "loss_total": total_loss / num_batches,
        "loss_latent": total_latent_loss / num_batches,
        "loss_action": total_action_loss / num_batches,
        "num_batches": float(num_batches),
    }


def write_summary_json(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
