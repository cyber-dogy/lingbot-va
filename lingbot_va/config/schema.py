from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def default_device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class ExperimentConfig:
    run_name: str = "synthetic_smoke"
    dataset_name: str = "synthetic_latent"
    train_manifest_path: Path | None = None
    valid_manifest_path: Path | None = None
    output_root: Path = Path("train_out") / "lingbot_va"
    seed: int = 20260422
    device: str = default_device_str()
    resume_from_latest: bool = False

    context_frames: int = 6
    prediction_frames: int = 6
    latent_dim: int = 48
    action_dim: int = 30
    text_dim: int = 256

    encoder_name: str = "task_text"
    backbone_name: str = "latent_action_transformer"
    head_name: str = "latent_action"
    policy_name: str = "lingbot_joint"

    hidden_dim: int = 192
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1

    train_size: int = 128
    valid_size: int = 32
    synthetic_num_tasks: int = 12
    synthetic_noise_std: float = 0.03

    batch_size: int = 8
    num_workers: int = 0
    train_epochs: int = 3
    grad_accum_steps: int = 1
    grad_clip_norm: float | None = 1.0

    learning_rate: float = 3.0e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1.0e-2
    lr_scheduler_name: str = "cosine"
    lr_warmup_steps: int = 10

    latent_loss_weight: float = 1.0
    action_loss_weight: float = 1.0

    val_every_epochs: int = 1
    checkpoint_every_epochs: int = 1
    print_every: int = 10

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        if self.train_manifest_path is not None:
            self.train_manifest_path = Path(self.train_manifest_path)
        if self.valid_manifest_path is not None:
            self.valid_manifest_path = Path(self.valid_manifest_path)
        self.betas = tuple(float(beta) for beta in self.betas)

    def validate(self) -> None:
        if self.context_frames <= 0:
            raise ValueError("context_frames 必须大于 0。")
        if self.prediction_frames <= 0:
            raise ValueError("prediction_frames 必须大于 0。")
        if self.latent_dim <= 0 or self.action_dim <= 0 or self.text_dim <= 0:
            raise ValueError("latent_dim / action_dim / text_dim 必须为正数。")
        if self.hidden_dim <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_dim / num_layers / num_heads 必须为正数。")
        if self.train_size <= 0 or self.valid_size <= 0:
            raise ValueError("train_size / valid_size 必须为正数。")
        if self.batch_size <= 0 or self.train_epochs <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch_size / train_epochs / grad_accum_steps 必须为正数。")
        if self.synthetic_num_tasks <= 0:
            raise ValueError("synthetic_num_tasks 必须大于 0。")
        if self.dataset_name != "synthetic_latent":
            if self.train_manifest_path is None or not self.train_manifest_path.exists():
                raise FileNotFoundError(
                    f"dataset_name={self.dataset_name} 时，train_manifest_path 必须存在："
                    f"{self.train_manifest_path}"
                )
            if self.valid_manifest_path is None or not self.valid_manifest_path.exists():
                raise FileNotFoundError(
                    f"dataset_name={self.dataset_name} 时，valid_manifest_path 必须存在："
                    f"{self.valid_manifest_path}"
                )

    @property
    def ckpt_dir(self) -> Path:
        return self.output_root / self.run_name

    @property
    def periodic_ckpt_dir(self) -> Path:
        return self.ckpt_dir / "epochs"

    @property
    def latest_ckpt_path(self) -> Path:
        return self.ckpt_dir / "latest.pt"

    @property
    def best_ckpt_path(self) -> Path:
        return self.ckpt_dir / "best.pt"

    @property
    def config_path(self) -> Path:
        return self.ckpt_dir / "config.json"

    @property
    def summary_path(self) -> Path:
        return self.ckpt_dir / "summary.json"
