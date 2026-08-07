"""Conditioned blocks for the low-dimensional diffusers."""

from __future__ import annotations

import torch
from torch import nn

from .adaln import AdaLayerNorm


def _ffn(hidden_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 4, hidden_dim),
    )


class GoalConditionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_norm = AdaLayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = AdaLayerNorm(hidden_dim)
        self.ffn = _ffn(hidden_dim, dropout)

    def forward(
        self,
        token: torch.Tensor,
        memory: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        query = self.cross_norm(token, timestep)
        normalized_memory = self.memory_norm(memory)
        update = self.cross_attention(
            query, normalized_memory, normalized_memory, need_weights=False
        )[0]
        token = token + update
        return token + self.ffn(self.ffn_norm(token, timestep))


class TrajectoryConditionBlock(nn.Module):
    ATTENTION_MODES = {"full", "local"}

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        *,
        attention_mode: str = "full",
        local_window: int = 7,
        use_phase_attention: bool = False,
        phase_residual_gating: bool = False,
        residual_gating: bool = False,
        residual_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention_mode = str(attention_mode)
        if self.attention_mode not in self.ATTENTION_MODES:
            raise ValueError(f"Unknown temporal attention mode: {attention_mode}")
        self.local_window = int(local_window)
        if self.local_window < 1 or self.local_window % 2 == 0:
            raise ValueError("local_window must be a positive odd integer")
        self.residual_gating = bool(residual_gating)
        self.conv_norm = AdaLayerNorm(hidden_dim)
        self.temporal_conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1
        )
        self.self_norm = AdaLayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.phase_norm = AdaLayerNorm(hidden_dim) if use_phase_attention else None
        self.phase_memory_norm = (
            nn.LayerNorm(hidden_dim) if use_phase_attention else None
        )
        self.phase_attention = (
            nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            if use_phase_attention
            else None
        )
        self.cross_norm = AdaLayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = AdaLayerNorm(hidden_dim)
        self.ffn = _ffn(hidden_dim, dropout)
        if self.residual_gating:
            initial = torch.full((hidden_dim,), float(residual_gate_init))
            self.conv_scale = nn.Parameter(initial.clone())
            self.self_scale = nn.Parameter(initial.clone())
            self.phase_scale = (
                nn.Parameter(initial.clone()) if use_phase_attention else None
            )
            self.cross_scale = nn.Parameter(initial.clone())
            self.ffn_scale = nn.Parameter(initial.clone())
        else:
            self.register_parameter("conv_scale", None)
            self.register_parameter("self_scale", None)
            if phase_residual_gating and use_phase_attention:
                self.phase_scale = nn.Parameter(
                    torch.full((hidden_dim,), float(residual_gate_init))
                )
            else:
                self.register_parameter("phase_scale", None)
            self.register_parameter("cross_scale", None)
            self.register_parameter("ffn_scale", None)

    @staticmethod
    def _scale(update: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
        return update if scale is None else update * scale[None, None]

    def _self_attention_mask(self, tokens: torch.Tensor) -> torch.Tensor | None:
        if self.attention_mode == "full":
            return None
        length = tokens.shape[1]
        indices = torch.arange(length, device=tokens.device)
        radius = self.local_window // 2
        return (indices[:, None] - indices[None, :]).abs() > radius

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        timestep: torch.Tensor,
        phase_tokens: torch.Tensor | None = None,
        phase_attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conv = self.conv_norm(tokens, timestep).transpose(1, 2)
        tokens = tokens + self._scale(
            self.temporal_conv(conv).transpose(1, 2), self.conv_scale
        )
        normalized = self.self_norm(tokens, timestep)
        self_update = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=self._self_attention_mask(tokens),
            need_weights=False,
        )[0]
        tokens = tokens + self._scale(self_update, self.self_scale)
        if self.phase_attention is not None:
            if phase_tokens is None:
                raise ValueError("phase_tokens are required when phase attention is enabled")
            phase_memory = self.phase_memory_norm(phase_tokens)
            phase_update = self.phase_attention(
                self.phase_norm(tokens, timestep),
                phase_memory,
                phase_memory,
                attn_mask=phase_attention_bias,
                need_weights=False,
            )[0]
            tokens = tokens + self._scale(phase_update, self.phase_scale)
        normalized_memory = self.memory_norm(memory)
        cross_update = self.cross_attention(
            self.cross_norm(tokens, timestep),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )[0]
        tokens = tokens + self._scale(cross_update, self.cross_scale)
        return tokens + self._scale(
            self.ffn(self.ffn_norm(tokens, timestep)), self.ffn_scale
        )
