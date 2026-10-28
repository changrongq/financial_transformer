from __future__ import annotations

import math

import torch
from torch import nn


def _get_alibi_slopes(num_heads: int) -> torch.Tensor:
    def slopes_power_of_two(power_of_two: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(power_of_two) - 3)))
        ratio = start
        return [start * (ratio ** index) for index in range(power_of_two)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(slopes_power_of_two(num_heads), dtype=torch.float32)

    closest_power_of_two = 2 ** math.floor(math.log2(num_heads))
    slopes = slopes_power_of_two(closest_power_of_two)
    extra_slopes = slopes_power_of_two(closest_power_of_two * 2)[0::2]
    slopes.extend(extra_slopes[: num_heads - closest_power_of_two])
    return torch.tensor(slopes, dtype=torch.float32)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TimeAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        if padding_mask is not None:
            weights = weights.masked_fill(padding_mask, 0.0)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights.unsqueeze(-1)
        return (weights * x).sum(dim=1)


class TaskTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        mid_dim = max(hidden_dim // 2, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
        )
        self.output_dim = mid_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(d_model, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        x = value * torch.nn.functional.silu(gate)
        x = self.dropout(x)
        return self.out_proj(x)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, ffn_type: str, position_encoding: str):
        super().__init__()
        self.nhead = nhead
        self.use_alibi = position_encoding == "alibi"
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if ffn_type == "swiglu":
            self.ffn = SwiGLUFeedForward(d_model=d_model, hidden_dim=dim_feedforward, dropout=dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
            )
        if self.use_alibi:
            self.register_buffer("alibi_slopes", _get_alibi_slopes(nhead), persistent=False)
        else:
            self.alibi_slopes = None

    def _build_attention_bias(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.use_alibi or self.alibi_slopes is None:
            return None
        _, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        relative_distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        bias = -self.alibi_slopes.to(device=x.device, dtype=x.dtype).view(self.nhead, 1, 1) * relative_distance.unsqueeze(0)
        return bias.repeat(x.size(0), 1, 1)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None, return_attention: bool = False):
        attn_input = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=self._build_attention_bias(attn_input),
            key_padding_mask=padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.dropout(attn_out)
        ffn_input = self.norm2(x)
        x = x + self.dropout(self.ffn(ffn_input))
        if return_attention:
            return x, attn_weights
        return x


def _split_embedding_dims(total_dim: int, group_count: int) -> list[int]:
    base_dim = total_dim // group_count
    remainder = total_dim % group_count
    return [base_dim + (1 if index < remainder else 0) for index in range(group_count)]


class GroupedFeatureProjector(nn.Module):
    def __init__(self, feature_cols: list[str], grouped_feature_groups: dict[str, list[str]], d_model: int, dropout: float, use_group_gates: bool = False):
        super().__init__()
        if not feature_cols:
            raise ValueError("grouped input projection requires feature_cols")
        if not grouped_feature_groups:
            raise ValueError("grouped input projection requires grouped_feature_groups")

        assigned_features: set[str] = set()
        group_features: list[list[str]] = []
        for group_name, group_cols in grouped_feature_groups.items():
            if not group_cols:
                continue
            missing = [feature for feature in group_cols if feature not in feature_cols]
            if missing:
                raise ValueError(f"group {group_name} contains unknown features: {missing}")
            group_features.append(group_cols)
            assigned_features.update(group_cols)

        residual_features = [feature for feature in feature_cols if feature not in assigned_features]
        if residual_features:
            group_features.append(residual_features)
        if not group_features:
            raise ValueError("grouped input projection resolved no feature groups")

        projection_dims = _split_embedding_dims(d_model, len(group_features))
        self.use_group_gates = use_group_gates
        self.group_indices = [torch.tensor([feature_cols.index(feature) for feature in group], dtype=torch.long) for group in group_features]
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(len(indices)),
                    nn.Linear(len(indices), proj_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for indices, proj_dim in zip(self.group_indices, projection_dims)
            ]
        )
        self.group_gates = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(len(indices)), nn.Linear(len(indices), 1)) for indices in self.group_indices]
        ) if use_group_gates else None
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected_groups = []
        for group_index, (indices, projection) in enumerate(zip(self.group_indices, self.projections)):
            group_x = x.index_select(dim=2, index=indices.to(device=x.device))
            projected = projection(group_x)
            if self.group_gates is not None:
                gate = torch.sigmoid(self.group_gates[group_index](group_x))
                projected = projected * gate
            projected_groups.append(projected)
        return self.fusion(torch.cat(projected_groups, dim=-1))


class FinancialTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        ffn_type: str = "mlp",
        position_encoding: str = "sinusoidal",
        model_variant: str = "flat",
        feature_cols: list[str] | None = None,
        grouped_feature_groups: dict[str, list[str]] | None = None,
        use_group_gates: bool = False,
    ):
        super().__init__()
        self.model_variant = model_variant
        self.grouped_input = None
        self.input_proj = None
        if model_variant == "grouped":
            self.grouped_input = GroupedFeatureProjector(
                feature_cols=feature_cols or [],
                grouped_feature_groups=grouped_feature_groups or {},
                d_model=d_model,
                dropout=dropout,
                use_group_gates=use_group_gates,
            )
        else:
            self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.position = SinusoidalPositionEncoding(d_model) if position_encoding == "sinusoidal" else None
        self.ffn_type = ffn_type
        self.position_encoding = position_encoding
        use_custom_encoder = ffn_type != "mlp" or position_encoding == "alibi"
        if not use_custom_encoder:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.encoder = nn.ModuleList(
                [
                    CustomTransformerEncoderLayer(
                        d_model,
                        nhead,
                        dim_feedforward,
                        dropout,
                        ffn_type=ffn_type,
                        position_encoding=position_encoding,
                    )
                    for _ in range(num_layers)
                ]
            )
        self.pool = TimeAttentionPool(d_model)
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.short_up_tower = TaskTower(d_model, d_model, dropout)
        self.take_profit_tower = TaskTower(d_model, d_model, dropout)
        self.stop_loss_tower = TaskTower(d_model, d_model, dropout)
        self.max_gain_tower = TaskTower(d_model, d_model, dropout)
        self.max_drawdown_tower = TaskTower(d_model, d_model, dropout)
        self.short_up_head = nn.Linear(self.short_up_tower.output_dim, 1)
        self.take_profit_head = nn.Linear(self.take_profit_tower.output_dim, 1)
        self.stop_loss_head = nn.Linear(self.stop_loss_tower.output_dim, 1)
        self.max_gain_head = nn.Linear(self.max_gain_tower.output_dim, 1)
        self.max_drawdown_head = nn.Linear(self.max_drawdown_tower.output_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        if self.grouped_input is not None:
            x = self.grouped_input(x)
            x = self.input_norm(x)
        else:
            x = self.input_norm(self.input_proj(x))
        if self.position is not None:
            x = self.position(x)
        attentions: list[torch.Tensor] = []
        if isinstance(self.encoder, nn.ModuleList):
            for layer in self.encoder:
                if return_attention:
                    x, attn_weights = layer(x, padding_mask=padding_mask, return_attention=True)
                    attentions.append(attn_weights)
                else:
                    x = layer(x, padding_mask=padding_mask)
        else:
            if return_attention:
                for layer in self.encoder.layers:
                    if layer.norm_first:
                        normed_x = layer.norm1(x)
                        attn_out, attn_weights = layer.self_attn(
                            normed_x,
                            normed_x,
                            normed_x,
                            attn_mask=None,
                            key_padding_mask=padding_mask,
                            need_weights=True,
                            average_attn_weights=False,
                        )
                        x = x + layer.dropout1(attn_out)
                        ffn_input = layer.norm2(x)
                        x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(ffn_input)))))
                    else:
                        attn_input = layer.norm1(x)
                        attn_out, attn_weights = layer.self_attn(
                            attn_input,
                            attn_input,
                            attn_input,
                            attn_mask=None,
                            key_padding_mask=padding_mask,
                            need_weights=True,
                            average_attn_weights=False,
                        )
                        x = x + layer.dropout1(attn_out)
                        x = layer.norm1(x)
                        ffn_input = layer.norm2(x)
                        x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(ffn_input)))))
                        x = layer.norm2(x)
                    attentions.append(attn_weights)
            else:
                x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.pool(x, padding_mask=padding_mask)
        x = self.shared(x)
        short_features = self.short_up_tower(x)
        take_profit_features = self.take_profit_tower(x)
        stop_loss_features = self.stop_loss_tower(x)
        max_gain_features = self.max_gain_tower(x)
        max_drawdown_features = self.max_drawdown_tower(x)
        return {
            "short_up_logits": self.short_up_head(short_features).squeeze(-1),
            "take_profit_logits": self.take_profit_head(take_profit_features).squeeze(-1),
            "stop_loss_logits": self.stop_loss_head(stop_loss_features).squeeze(-1),
            "max_gain": self.max_gain_head(max_gain_features).squeeze(-1),
            "max_drawdown": self.max_drawdown_head(max_drawdown_features).squeeze(-1),
        } if not return_attention else ({
            "short_up_logits": self.short_up_head(short_features).squeeze(-1),
            "take_profit_logits": self.take_profit_head(take_profit_features).squeeze(-1),
            "stop_loss_logits": self.stop_loss_head(stop_loss_features).squeeze(-1),
            "max_gain": self.max_gain_head(max_gain_features).squeeze(-1),
            "max_drawdown": self.max_drawdown_head(max_drawdown_features).squeeze(-1),
        }, attentions)


def create_model(
    input_dim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    ffn_type: str = "mlp",
    position_encoding: str = "sinusoidal",
    model_variant: str = "flat",
    feature_cols: list[str] | None = None,
    grouped_feature_groups: dict[str, list[str]] | None = None,
    use_group_gates: bool = False,
) -> FinancialTransformer:
    return FinancialTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        ffn_type=ffn_type,
        position_encoding=position_encoding,
        model_variant=model_variant,
        feature_cols=feature_cols,
        grouped_feature_groups=grouped_feature_groups,
        use_group_gates=use_group_gates,
    )