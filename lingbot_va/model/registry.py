from __future__ import annotations

from collections.abc import Callable

from lingbot_va.config import ExperimentConfig

from .backbones import LatentActionTransformerBackbone
from .encoders import TaskTextEncoder
from .heads import LatentActionHead


EncoderBuilder = Callable[[ExperimentConfig], object]
BackboneBuilder = Callable[[ExperimentConfig], object]
HeadBuilder = Callable[[ExperimentConfig], object]


def _build_task_text_encoder(cfg: ExperimentConfig) -> TaskTextEncoder:
    return TaskTextEncoder(
        text_dim=cfg.text_dim,
        hidden_dim=cfg.hidden_dim,
    )


def _build_latent_action_transformer(cfg: ExperimentConfig) -> LatentActionTransformerBackbone:
    return LatentActionTransformerBackbone(
        latent_dim=cfg.latent_dim,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        context_frames=cfg.context_frames,
    )


def _build_latent_action_head(cfg: ExperimentConfig) -> LatentActionHead:
    return LatentActionHead(
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        action_dim=cfg.action_dim,
        prediction_frames=cfg.prediction_frames,
    )


ENCODER_REGISTRY: dict[str, EncoderBuilder] = {
    "task_text": _build_task_text_encoder,
}

BACKBONE_REGISTRY: dict[str, BackboneBuilder] = {
    "latent_action_transformer": _build_latent_action_transformer,
}

HEAD_REGISTRY: dict[str, HeadBuilder] = {
    "latent_action": _build_latent_action_head,
}


def build_text_encoder(cfg: ExperimentConfig):
    try:
        builder = ENCODER_REGISTRY[str(cfg.encoder_name)]
    except KeyError as exc:
        raise ValueError(f"不支持的 encoder_name: {cfg.encoder_name}") from exc
    return builder(cfg)


def build_backbone(cfg: ExperimentConfig):
    try:
        builder = BACKBONE_REGISTRY[str(cfg.backbone_name)]
    except KeyError as exc:
        raise ValueError(f"不支持的 backbone_name: {cfg.backbone_name}") from exc
    return builder(cfg)


def build_head(cfg: ExperimentConfig):
    try:
        builder = HEAD_REGISTRY[str(cfg.head_name)]
    except KeyError as exc:
        raise ValueError(f"不支持的 head_name: {cfg.head_name}") from exc
    return builder(cfg)


def list_encoders() -> list[str]:
    return sorted(ENCODER_REGISTRY)


def list_backbones() -> list[str]:
    return sorted(BACKBONE_REGISTRY)


def list_heads() -> list[str]:
    return sorted(HEAD_REGISTRY)
