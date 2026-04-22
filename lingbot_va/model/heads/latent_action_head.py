from __future__ import annotations

import torch
from torch import nn


class LatentActionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        action_dim: int,
        prediction_frames: int,
    ) -> None:
        super().__init__()
        self.prediction_frames = int(prediction_frames)
        self.frame_embedding = nn.Embedding(prediction_frames, hidden_dim)
        self.latent_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        joint_state = (
            features["summary"] + features["latent_summary"] + features["action_summary"]
        ) / 3.0
        batch_size = int(joint_state.size(0))
        frame_ids = torch.arange(self.prediction_frames, device=joint_state.device)
        frame_tokens = self.frame_embedding(frame_ids).unsqueeze(0).expand(batch_size, -1, -1)

        # 输出头不直接复用整段序列 hidden，而是只拿 joint summary 和每一帧的位置编码，
        # 这样这层后续迁到真实 LingBot-VA 版本时更容易替换成更复杂的 decoder。
        latent_context = (joint_state + features["latent_summary"]).unsqueeze(1).expand(
            -1, self.prediction_frames, -1
        )
        action_context = (joint_state + features["action_summary"]).unsqueeze(1).expand(
            -1, self.prediction_frames, -1
        )

        latent_pred = self.latent_mlp(torch.cat([frame_tokens, latent_context], dim=-1))
        action_pred = self.action_mlp(torch.cat([frame_tokens, action_context], dim=-1))
        return {
            "latent_pred": latent_pred,
            "action_pred": action_pred,
        }
