from __future__ import annotations

from typing import Any

import torch

from lingbot_va.config import ExperimentConfig, config_to_dict


def _default_resume_state() -> dict[str, Any]:
    return {
        "start_epoch": 0,
        "global_step": 0,
        "best_metric": None,
        "history": [],
    }


def save_checkpoint(
    path,
    cfg: ExperimentConfig,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    history: list[dict[str, Any]],
) -> None:
    checkpoint_path = cfg.latest_ckpt_path if path is None else path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config_to_dict(cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "history": history,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint_payload(path):
    return torch.load(path, map_location="cpu")


def load_resume_state(
    cfg: ExperimentConfig,
    model,
    optimizer,
    scheduler,
) -> dict[str, Any]:
    if not cfg.resume_from_latest:
        return _default_resume_state()
    if not cfg.latest_ckpt_path.exists():
        raise FileNotFoundError(f"resume_from_latest=True，但未找到：{cfg.latest_ckpt_path}")

    payload = load_checkpoint_payload(cfg.latest_ckpt_path)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return {
        "start_epoch": int(payload["epoch"]) + 1,
        "global_step": int(payload["global_step"]),
        "best_metric": payload["best_metric"],
        "history": list(payload["history"]),
    }
