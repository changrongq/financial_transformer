from __future__ import annotations

import torch
from torch import nn


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return (weights * x).sum(dim=1)


class MicroLSTMModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.act(self.fc1(out))
        return self.fc2(out).squeeze(-1)


class SmallBiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True, bidirectional=True)
        self.attn = AttentionPool(128)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.attn(out)
        out = self.dropout(out)
        out = self.act(self.fc1(out))
        return self.fc2(out).squeeze(-1)


class MidCNNBiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(64)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.attn = AttentionPool(128)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        x = self.norm(x)
        out, _ = self.lstm(x)
        out = self.attn(out)
        out = self.dropout(out)
        out = self.act(self.fc1(out))
        return self.fc2(out).squeeze(-1)


class LargeTransformerModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        out = self.encoder(out)
        out = self.norm(out[:, -1, :])
        out = self.dropout(out)
        out = self.act(self.fc1(out))
        return self.fc2(out).squeeze(-1)


def build_model(layer: str, input_dim: int) -> nn.Module:
    mapping = {
        "micro": MicroLSTMModel,
        "small": SmallBiLSTMAttentionModel,
        "mid": MidCNNBiLSTMAttentionModel,
        "large": LargeTransformerModel,
    }
    if layer not in mapping:
        raise ValueError(f"未知分层: {layer}")
    return mapping[layer](input_dim)
