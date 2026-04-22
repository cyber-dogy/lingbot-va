from __future__ import annotations

from torch import nn


class TaskTextEncoder(nn.Module):
    def __init__(self, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, text_context):
        return self.network(text_context)
