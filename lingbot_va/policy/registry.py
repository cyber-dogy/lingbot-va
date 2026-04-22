from __future__ import annotations

from lingbot_va.config import ExperimentConfig

from .lingbot_policy import LingBotJointPolicy, LingBotJointPolicyConfig


def build_policy(
    cfg: ExperimentConfig,
    text_encoder,
    backbone,
    head,
):
    policy_name = str(cfg.policy_name).lower()
    if policy_name != "lingbot_joint":
        raise ValueError(f"不支持的 policy_name: {cfg.policy_name}")
    policy_cfg = LingBotJointPolicyConfig(
        latent_loss_weight=cfg.latent_loss_weight,
        action_loss_weight=cfg.action_loss_weight,
    )
    return LingBotJointPolicy(
        policy_cfg=policy_cfg,
        text_encoder=text_encoder,
        backbone=backbone,
        head=head,
    )


def list_policies() -> list[str]:
    return ["lingbot_joint"]
