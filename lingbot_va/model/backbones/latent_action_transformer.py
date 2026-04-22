from __future__ import annotations

import torch
from torch import nn


class LatentActionTransformerBackbone(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        context_frames: int,
    ) -> None:
        super().__init__()
        self.context_frames = int(context_frames)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.position_embedding = nn.Embedding(1 + 2 * context_frames, hidden_dim)
        self.text_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.latent_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        latent_context: torch.Tensor,
        action_context: torch.Tensor,
        text_token: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent_tokens = self.latent_proj(latent_context) + self.latent_type_embedding
        action_tokens = self.action_proj(action_context) + self.action_type_embedding
        text_tokens = text_token.unsqueeze(1) + self.text_type_embedding

        # 文本 token 放在最前面，后面再拼接 latent / action token，
        # 这样 backbone 内部的注意力上下文天然就是“任务语义 + 历史观测 + 历史动作”。
        tokens = torch.cat([text_tokens, latent_tokens, action_tokens], dim=1)
        position_ids = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        tokens = tokens + self.position_embedding(position_ids)

        hidden = self.out_norm(self.encoder(tokens))
        latent_hidden = hidden[:, 1 : 1 + self.context_frames]
        action_hidden = hidden[:, 1 + self.context_frames :]
        return {
            "summary": hidden[:, 0],
            "latent_summary": latent_hidden.mean(dim=1),
            "action_summary": action_hidden.mean(dim=1),
        }
