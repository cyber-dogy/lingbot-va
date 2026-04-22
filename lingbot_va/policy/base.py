from __future__ import annotations

import torch
from torch import nn

from lingbot_va.config import ExperimentConfig


class BaseLingBotPolicy(nn.Module):
    def get_optimizer(self, cfg: ExperimentConfig):
        return torch.optim.AdamW(
            self.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
