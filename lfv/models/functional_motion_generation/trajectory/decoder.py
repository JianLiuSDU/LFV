"""Non-causal trajectory diffusion transformer."""

from __future__ import annotations

import torch
from torch import nn

from ..blocks import (
    GoalConditionedContextMixer,
    LatentPhaseTokenGenerator,
    SinusoidalEmbedding,
    TimestepEmbedding,
    TrajectoryConditionBlock,
)


class TrajectoryDecoder(nn.Module):
    POSITION_ENCODING_MODES = {
        "discrete_sinusoidal",
        "legacy_normalized_sinusoidal",
    }

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        horizon_without_start: int = 63,
        use_hard_start_token: bool = False,
        position_encoding: str = "discrete_sinusoidal",
        goal_context_layers: int = 0,
        goal_context_residual_gating: bool = False,
        num_phase_tokens: int = 0,
        temporal_attention_mode: str = "full",
        temporal_local_window: int = 7,
        phase_residual_gating: bool = False,
        residual_gating: bool = False,
        residual_gate_init: float = 0.1,
        phase_attention_sigma: float = 0.22,
    ) -> None:
        super().__init__()
        self.horizon_without_start = int(horizon_without_start)
        self.use_hard_start_token = bool(use_hard_start_token)
        self.position_encoding = str(position_encoding)
        self.goal_context_layers = int(goal_context_layers)
        self.goal_context_residual_gating = bool(goal_context_residual_gating)
        self.num_phase_tokens = int(num_phase_tokens)
        self.temporal_attention_mode = str(temporal_attention_mode)
        self.phase_attention_sigma = float(phase_attention_sigma)
        if self.position_encoding not in self.POSITION_ENCODING_MODES:
            raise ValueError(
                f"Unknown trajectory position encoding: {self.position_encoding}. "
                f"Expected one of {sorted(self.POSITION_ENCODING_MODES)}"
            )
        self.pose_embedding = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.goal_embedding = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.timestep_embedding = TimestepEmbedding(hidden_dim)
        self.progress_embedding = SinusoidalEmbedding(hidden_dim)
        self.context_mixer = (
            GoalConditionedContextMixer(
                hidden_dim,
                num_heads,
                self.goal_context_layers,
                dropout,
                residual_gating=self.goal_context_residual_gating,
                residual_gate_init=residual_gate_init,
            )
            if self.goal_context_layers > 0
            else None
        )
        self.phase_generator = (
            LatentPhaseTokenGenerator(
                hidden_dim, num_heads, self.num_phase_tokens, dropout
            )
            if self.num_phase_tokens > 0
            else None
        )
        if self.temporal_attention_mode not in {
            "full",
            "alternating_local_global",
        }:
            raise ValueError(
                f"Unknown trajectory temporal attention mode: {temporal_attention_mode}"
            )
        self.blocks = nn.ModuleList(
            [
                TrajectoryConditionBlock(
                    hidden_dim,
                    num_heads,
                    dropout,
                    attention_mode=(
                        "local"
                        if self.temporal_attention_mode
                        == "alternating_local_global"
                        and layer_index % 2 == 0
                        else "full"
                    ),
                    local_window=temporal_local_window,
                    use_phase_attention=self.num_phase_tokens > 0,
                    phase_residual_gating=phase_residual_gating,
                    residual_gating=residual_gating,
                    residual_gate_init=residual_gate_init,
                )
                for layer_index in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 9)
        with torch.no_grad():
            self.output.bias.zero_()
            self.output.bias[3] = 1.0
            self.output.bias[7] = 1.0

    def _frame_positions(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the absolute trajectory-frame indices used by self-attention."""

        sequence_length = self.horizon_without_start + int(self.use_hard_start_token)
        if self.position_encoding == "discrete_sinusoidal":
            first_frame = 0 if self.use_hard_start_token else 1
            return torch.arange(
                first_frame,
                first_frame + sequence_length,
                device=device,
                dtype=dtype,
            )
        return torch.linspace(
            0.0 if self.use_hard_start_token else 1.0 / self.horizon_without_start,
            1.0,
            sequence_length,
            device=device,
            dtype=dtype,
        )

    def _phase_attention_bias(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return a soft monotonic prior from frames to ordered phase tokens."""

        if self.num_phase_tokens <= 0:
            return None
        frame = torch.linspace(0.0, 1.0, sequence_length, device=device, dtype=dtype)
        center = torch.linspace(
            0.0, 1.0, self.num_phase_tokens, device=device, dtype=dtype
        )
        return -0.5 * (
            (frame[:, None] - center[None, :]) / self.phase_attention_sigma
        ).square()

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        normalized_goal: torch.Tensor,
        normalized_start: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noisy_tokens = self.pose_embedding(noisy_trajectory)
        frame_positions = self._frame_positions(
            device=noisy_tokens.device,
            dtype=noisy_tokens.dtype,
        )
        progress_tokens = self.progress_embedding(frame_positions)[None].to(
            dtype=noisy_tokens.dtype
        )
        fixed_start = None
        if self.use_hard_start_token:
            if normalized_start is None:
                raise ValueError("normalized_start is required for hard start token")
            fixed_start = self.pose_embedding(normalized_start)[:, None]
            fixed_start = fixed_start + progress_tokens[:, :1]
            tokens = torch.cat(
                (fixed_start, noisy_tokens + progress_tokens[:, 1:]), dim=1
            )
        else:
            tokens = noisy_tokens + progress_tokens
        goal_token = self.goal_embedding(normalized_goal)[:, None]
        memory = (
            self.context_mixer(context, goal_token)
            if self.context_mixer is not None
            else torch.cat((context, goal_token), dim=1)
        )
        phase_tokens = (
            self.phase_generator(memory) if self.phase_generator is not None else None
        )
        phase_attention_bias = self._phase_attention_bias(
            tokens.shape[1], device=tokens.device, dtype=tokens.dtype
        )
        time = self.timestep_embedding(timestep)
        for block in self.blocks:
            tokens = block(
                tokens,
                memory,
                time,
                phase_tokens=phase_tokens,
                phase_attention_bias=phase_attention_bias,
            )
            if fixed_start is not None:
                # The clean identity token is an inpainting boundary. Reset it
                # after every block so temporal convolution/self-attention can
                # read it, but denoising can never move the boundary itself.
                tokens = torch.cat((fixed_start, tokens[:, 1:]), dim=1)
        if fixed_start is not None:
            tokens = tokens[:, 1:]
        return self.output(self.output_norm(tokens))
