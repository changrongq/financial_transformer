from __future__ import annotations

import torch
from torch import nn


class SharedTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_dim = hidden_dim * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.proj(x))
        hidden = self.encoder(hidden)
        last_token = hidden[:, -1, :]
        mean_pool = hidden.mean(dim=1)
        max_pool = hidden.max(dim=1).values
        return torch.cat([last_token, mean_pool, max_pool], dim=-1)


class TaskTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        mid_dim = max(hidden_dim // 2, 32)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
        )
        self.output_dim = mid_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskForecastModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.encoder = SharedTemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        encoder_dim = self.encoder.output_dim
        self.shared = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.short_up_tower = TaskTower(hidden_dim, hidden_dim, dropout)
        self.take_profit_tower = TaskTower(hidden_dim, hidden_dim, dropout)
        self.stop_loss_tower = TaskTower(hidden_dim, hidden_dim, dropout)
        self.max_gain_tower = TaskTower(hidden_dim, hidden_dim, dropout)
        self.max_drawdown_tower = TaskTower(hidden_dim, hidden_dim, dropout)
        self.short_up_head = nn.Linear(self.short_up_tower.output_dim, 1)
        self.take_profit_head = nn.Linear(self.take_profit_tower.output_dim, 1)
        self.stop_loss_head = nn.Linear(self.stop_loss_tower.output_dim, 1)
        self.max_gain_head = nn.Linear(self.max_gain_tower.output_dim, 1)
        self.max_drawdown_head = nn.Linear(self.max_drawdown_tower.output_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.shared(self.encoder(x))
        short_features = self.short_up_tower(features)
        take_profit_features = self.take_profit_tower(features)
        stop_loss_features = self.stop_loss_tower(features)
        max_gain_features = self.max_gain_tower(features)
        max_drawdown_features = self.max_drawdown_tower(features)
        return {
            "short_up_logits": self.short_up_head(short_features).squeeze(-1),
            "take_profit_logits": self.take_profit_head(take_profit_features).squeeze(-1),
            "stop_loss_logits": self.stop_loss_head(stop_loss_features).squeeze(-1),
            "max_gain": self.max_gain_head(max_gain_features).squeeze(-1),
            "max_drawdown": self.max_drawdown_head(max_drawdown_features).squeeze(-1),
        }