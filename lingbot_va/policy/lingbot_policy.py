from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .base import BaseLingBotPolicy


@dataclass
class LingBotJointPolicyConfig:
    latent_loss_weight: float
    action_loss_weight: float


class LingBotJointPolicy(BaseLingBotPolicy):
    def __init__(
        self,
        policy_cfg: LingBotJointPolicyConfig,
        text_encoder,
        backbone,
        head,
    ) -> None:
        super().__init__()
        self.policy_cfg = policy_cfg
        self.text_encoder = text_encoder
        self.backbone = backbone
        self.head = head

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text_token = self.text_encoder(batch["text_context"])
        features = self.backbone(
            latent_context=batch["latent_context"],
            action_context=batch["action_context"],
            text_token=text_token,
        )
        return self.head(features)

    def compute_loss_dict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        predictions = self.forward(batch)
        latent_loss = F.mse_loss(predictions["latent_pred"], batch["latent_target"])
        action_loss = F.mse_loss(predictions["action_pred"], batch["action_target"])
        loss_total = (
            self.policy_cfg.latent_loss_weight * latent_loss
            + self.policy_cfg.action_loss_weight * action_loss
        )
        return {
            "loss_total": loss_total,
            "loss_latent": latent_loss,
            "loss_action": action_loss,
        }
