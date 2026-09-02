"""Source-canonical, field-gated encoder used by Stage 2 V7.

The V7 encoder deliberately separates three operations:

1. :class:`LocalPointEncoder` creates a per-point XYZ--DINO payload and has
   no object-level pooling;
2. :class:`FieldSelector` may inspect the complete two-object relation, but
   returns only scalar gates;
3. :class:`GatedRelationEncoder` rebuilds the relation from the raw payload,
   using the scalar fields as both key bias and output gates.

This boundary is important: selector hidden states never become a generator
condition.  The only generator input is the field-gated functional context
produced by :class:`FunctionalPooling`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..interfaces import ContextEncoding


def _as_mask(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(
            reference.shape[:2], device=reference.device, dtype=reference.dtype
        )
    if mask.ndim != 2 or mask.shape != reference.shape[:2]:
        raise ValueError(
            f"visibility mask must be [B,N]={reference.shape[:2]}, got {tuple(mask.shape)}"
        )
    return mask.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)


def _masked_center(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (points * mask.unsqueeze(-1)).sum(dim=1) / denominator


def _key_bias(gate: torch.Tensor, query_length: int, num_heads: int) -> torch.Tensor:
    """Build a per-batch additive attention mask for key/value gates."""

    if gate.ndim != 2:
        raise ValueError(f"gate must be [B,N], got {tuple(gate.shape)}")
    bias = gate.clamp_min(1e-6).log()[:, None, None, :]
    bias = bias.expand(gate.shape[0], num_heads, query_length, gate.shape[1])
    return bias.reshape(gate.shape[0] * num_heads, query_length, gate.shape[1])


def _intervene_gate(
    gate: torch.Tensor,
    mask: torch.Tensor,
    intervention: str | None,
) -> torch.Tensor:
    """Apply paired diagnostic interventions without changing point order."""

    if intervention is None:
        return gate * mask
    valid = mask > 0.5
    output = gate.clone()
    if intervention == "uniform":
        output = torch.full_like(output, 0.5)
    elif intervention in {"roll", "shuffled"}:
        output = torch.roll(output, shifts=max(1, output.shape[1] // 2), dims=1)
    elif intervention == "complement":
        output = 1.0 - output
    elif intervention.startswith("drop_top") or intervention.startswith("keep_top"):
        keep = intervention.startswith("keep_top")
        fraction = 0.10
        if "_" in intervention:
            try:
                fraction = float(intervention.rsplit("_", 1)[-1]) / 100.0
            except ValueError as exc:
                raise ValueError(f"Invalid V7 field intervention: {intervention}") from exc
        fraction = min(max(fraction, 1.0 / max(output.shape[1], 1)), 1.0)
        count = max(1, int(round(fraction * output.shape[1])))
        masked_values = output.masked_fill(~valid, float("-inf"))
        indices = torch.topk(masked_values, k=min(count, output.shape[1]), dim=1).indices
        selected = torch.zeros_like(output).scatter(1, indices, 1.0)
        output = output * selected if keep else output * (1.0 - selected)
    elif intervention == "bottom20":
        masked_values = output.masked_fill(~valid, float("inf"))
        count = max(1, int(round(0.20 * output.shape[1])))
        indices = torch.topk(-masked_values, k=min(count, output.shape[1]), dim=1).indices
        selected = torch.zeros_like(output).scatter(1, indices, 1.0)
        output = output * selected
    else:
        raise ValueError(
            "V7 field intervention must be None, uniform, roll, shuffled, "
            "complement, drop_top[_PERCENT], keep_top_PERCENT, or bottom20"
        )
    return output * mask


class LocalPointEncoder(nn.Module):
    """Point-wise XYZ--DINO encoder with no global pooling or broadcast."""

    def __init__(
        self,
        dino_dim: int,
        hidden_dim: int = 128,
        dino_proj_dim: int = 64,
        object_xyz_dim: int = 32,
        relational_xyz_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dino_proj_dim + object_xyz_dim + relational_xyz_dim != hidden_dim:
            raise ValueError(
                "V7 local projection dimensions must sum to hidden_dim: "
                f"{dino_proj_dim}+{object_xyz_dim}+{relational_xyz_dim}!={hidden_dim}"
            )
        self.dino = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, 128),
            nn.GELU(),
            nn.Linear(128, dino_proj_dim),
            nn.LayerNorm(dino_proj_dim),
        )
        self.object_xyz = nn.Sequential(
            nn.Linear(3, object_xyz_dim),
            nn.GELU(),
            nn.Linear(object_xyz_dim, object_xyz_dim),
        )
        self.relation_xyz = nn.Sequential(
            nn.Linear(3, relational_xyz_dim),
            nn.GELU(),
            nn.Linear(relational_xyz_dim, relational_xyz_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        points_object: torch.Tensor,
        points_relation: torch.Tensor,
        dino: torch.Tensor,
    ) -> torch.Tensor:
        payload = torch.cat(
            (
                self.dino(dino),
                self.object_xyz(points_object),
                self.relation_xyz(points_relation),
            ),
            dim=-1,
        )
        return self.fusion(payload)


class _SelectorBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended = self.cross_attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )[0]
        query = query + attended
        return query + self.ffn(self.ffn_norm(query))


class FieldSelector(nn.Module):
    """Bidirectional relation observer whose public output is scalar fields only."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("field selector temperature must be positive")
        self.temperature = nn.Parameter(torch.tensor(float(temperature)).log())
        self.m_to_r = nn.ModuleList(
            [_SelectorBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(layers)]
        )
        self.r_to_m = nn.ModuleList(
            [_SelectorBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(layers)]
        )
        self.manipulated_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.reference_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        manipulated: torch.Tensor,
        reference: torch.Tensor,
        manipulated_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        selected_m = manipulated
        selected_r = reference
        for block_m, block_r in zip(self.m_to_r, self.r_to_m):
            selected_m = block_m(selected_m, selected_r)
            selected_r = block_r(selected_r, selected_m)
        temperature = self.temperature.exp().clamp_min(1e-3)
        logits_m = self.manipulated_head(selected_m).squeeze(-1) / temperature
        logits_r = self.reference_head(selected_r).squeeze(-1) / temperature
        gate_m = torch.sigmoid(logits_m) * manipulated_mask
        gate_r = torch.sigmoid(logits_r) * reference_mask
        # Only gates/logits leave this module in the generator path.  The
        # optional detached-style diagnostic return is consumed only by
        # ``V7SceneEncoding`` and never by Goal/Trajectory decoders.
        if return_features:
            return gate_m, gate_r, logits_m, logits_r, selected_m, selected_r
        return gate_m, gate_r, logits_m, logits_r


class _GatedRelationBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.num_heads = int(num_heads)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_gate: torch.Tensor,
        memory_gate: torch.Tensor,
    ) -> torch.Tensor:
        attention_bias = _key_bias(memory_gate, query.shape[1], self.num_heads)
        attended = self.cross_attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            attn_mask=attention_bias,
            need_weights=False,
        )[0]
        # Gate the whole residual update.  An ungated ``query + gate * update``
        # would leave the raw local payload as a bypass.
        output = query_gate.unsqueeze(-1) * (query + attended)
        output = query_gate.unsqueeze(-1) * (
            output + self.ffn(self.ffn_norm(output))
        )
        return self.output_norm(output) * query_gate.unsqueeze(-1)


class GatedRelationEncoder(nn.Module):
    """Rebuilds cross-object relations from local payloads under scalar gates."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.m_to_r = nn.ModuleList(
            [_GatedRelationBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(layers)]
        )
        self.r_to_m = nn.ModuleList(
            [_GatedRelationBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(layers)]
        )

    def forward(
        self,
        manipulated: torch.Tensor,
        reference: torch.Tensor,
        manipulated_gate: torch.Tensor,
        reference_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation_m = manipulated * manipulated_gate.unsqueeze(-1)
        relation_r = reference * reference_gate.unsqueeze(-1)
        for block_m, block_r in zip(self.m_to_r, self.r_to_m):
            relation_m = block_m(relation_m, relation_r, manipulated_gate, reference_gate)
            relation_r = block_r(relation_r, relation_m, reference_gate, manipulated_gate)
        return relation_m, relation_r


class FunctionalPooling(nn.Module):
    """Field-biased learned pooling producing four tokens per object."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        queries_per_role: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.queries_per_role = int(queries_per_role)
        self.manipulated_queries = nn.Parameter(torch.empty(queries_per_role, hidden_dim))
        self.reference_queries = nn.Parameter(torch.empty(queries_per_role, hidden_dim))
        nn.init.trunc_normal_(self.manipulated_queries, std=0.02)
        nn.init.trunc_normal_(self.reference_queries, std=0.02)
        self.m_query_norm = nn.LayerNorm(hidden_dim)
        self.r_query_norm = nn.LayerNorm(hidden_dim)
        self.m_memory_norm = nn.LayerNorm(hidden_dim)
        self.r_memory_norm = nn.LayerNorm(hidden_dim)
        self.m_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.r_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.num_heads = int(num_heads)
        self.joint_token = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def _pool_one(
        self,
        queries: torch.Tensor,
        relation: torch.Tensor,
        gate: torch.Tensor,
        query_norm: nn.LayerNorm,
        memory_norm: nn.LayerNorm,
        attention: nn.MultiheadAttention,
    ) -> torch.Tensor:
        batch = relation.shape[0]
        expanded_queries = queries[None].expand(batch, -1, -1)
        bias = _key_bias(gate, expanded_queries.shape[1], self.num_heads)
        result = attention(
            query_norm(expanded_queries),
            memory_norm(relation),
            memory_norm(relation),
            attn_mask=bias,
            need_weights=False,
        )[0]
        # Make a fully disabled field an explicit zero context rather than a
        # hidden, ungated summary from LayerNorm/bias terms.
        result = result * gate.mean(dim=1, keepdim=True).unsqueeze(-1)
        return result

    def forward(
        self,
        manipulated: torch.Tensor,
        reference: torch.Tensor,
        manipulated_gate: torch.Tensor,
        reference_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled_m = self._pool_one(
            self.manipulated_queries,
            manipulated,
            manipulated_gate,
            self.m_query_norm,
            self.m_memory_norm,
            self.m_attention,
        )
        pooled_r = self._pool_one(
            self.reference_queries,
            reference,
            reference_gate,
            self.r_query_norm,
            self.r_memory_norm,
            self.r_attention,
        )
        global_token = self.joint_token(
            torch.cat((pooled_m.mean(dim=1), pooled_r.mean(dim=1)), dim=-1)
        )[:, None]
        global_mass = 0.5 * (
            manipulated_gate.mean(dim=1) + reference_gate.mean(dim=1)
        )
        global_token = global_token * global_mass[:, None, None]
        return global_token, pooled_m, pooled_r


@dataclass
class V7SceneEncoding:
    """Debug-only payload for V7; generator still receives only ``tokens``."""

    context: ContextEncoding
    manipulated_local: torch.Tensor
    reference_local: torch.Tensor
    manipulated_relation: torch.Tensor
    reference_relation: torch.Tensor
    selector_feature_m: torch.Tensor | None = None
    selector_feature_r: torch.Tensor | None = None
    functional_tokens_m: torch.Tensor | None = None
    functional_tokens_r: torch.Tensor | None = None
    joint_token: torch.Tensor | None = None


class V7SceneEncoder(nn.Module):
    """Full source-canonical V7 scene encoder."""

    def __init__(
        self,
        dino_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        local_dino_proj_dim: int = 64,
        local_object_xyz_dim: int = 32,
        local_relation_xyz_dim: int = 32,
        selector_layers: int = 2,
        selector_ffn_dim: int = 256,
        selector_dropout: float | None = None,
        relation_layers: int = 2,
        relation_ffn_dim: int = 256,
        pooling_queries: int = 4,
        dropout: float = 0.1,
        field_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.manipulated_local_encoder = LocalPointEncoder(
            dino_dim,
            hidden_dim,
            local_dino_proj_dim,
            local_object_xyz_dim,
            local_relation_xyz_dim,
            dropout,
        )
        self.reference_local_encoder = LocalPointEncoder(
            dino_dim,
            hidden_dim,
            local_dino_proj_dim,
            local_object_xyz_dim,
            local_relation_xyz_dim,
            dropout,
        )
        self.manipulated_role = nn.Parameter(torch.zeros(hidden_dim))
        self.reference_role = nn.Parameter(torch.zeros(hidden_dim))
        self.selector = FieldSelector(
            hidden_dim,
            num_heads,
            selector_ffn_dim,
            selector_layers,
            dropout if selector_dropout is None else float(selector_dropout),
            temperature=field_temperature,
        )
        self.relation = GatedRelationEncoder(
            hidden_dim,
            num_heads,
            relation_ffn_dim,
            relation_layers,
            dropout,
        )
        self.pooling = FunctionalPooling(
            hidden_dim,
            num_heads,
            pooling_queries,
            dropout,
        )

    @staticmethod
    def _override_or_select(
        selected: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        override: tuple[torch.Tensor, torch.Tensor] | None,
        masks: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_m, gate_r, logits_m, logits_r = selected
        if override is None:
            return gate_m, gate_r, logits_m, logits_r
        override_m, override_r = override
        if override_m.shape != gate_m.shape or override_r.shape != gate_r.shape:
            raise ValueError(
                "V7 field override must match selected gate shapes: "
                f"{tuple(gate_m.shape)}, {tuple(gate_r.shape)}"
            )
        gate_m = override_m.to(gate_m).clamp(0.0, 1.0) * masks[0]
        gate_r = override_r.to(gate_r).clamp(0.0, 1.0) * masks[1]
        return gate_m, gate_r, logits_m, logits_r

    def forward(
        self,
        manipulated_points: torch.Tensor,
        manipulated_dino: torch.Tensor,
        reference_points: torch.Tensor,
        reference_dino: torch.Tensor,
        *,
        manipulated_mask: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
        scene_scale: torch.Tensor | None = None,
        field_override: tuple[torch.Tensor, torch.Tensor] | None = None,
        field_intervention: str | None = None,
        return_debug: bool = False,
    ) -> ContextEncoding | V7SceneEncoding:
        mask_m = _as_mask(manipulated_mask, manipulated_points)
        mask_r = _as_mask(reference_mask, reference_points)
        center_m = _masked_center(manipulated_points, mask_m)
        center_r = _masked_center(reference_points, mask_r)
        if scene_scale is None:
            scale = torch.ones(
                manipulated_points.shape[0], 1, device=manipulated_points.device,
                dtype=manipulated_points.dtype,
            )
        else:
            scale = scene_scale.to(manipulated_points).reshape(-1, 1).clamp_min(1e-6)
        object_m = (manipulated_points - center_m[:, None]) / scale[:, None]
        object_r = (reference_points - center_r[:, None]) / scale[:, None]
        relation_m = (manipulated_points - center_r[:, None]) / scale[:, None]
        relation_r = (reference_points - center_r[:, None]) / scale[:, None]
        local_m = self.manipulated_local_encoder(object_m, relation_m, manipulated_dino)
        local_r = self.reference_local_encoder(object_r, relation_r, reference_dino)
        local_m = local_m + self.manipulated_role[None, None]
        local_r = local_r + self.reference_role[None, None]
        selected_output = self.selector(
            local_m,
            local_r,
            mask_m,
            mask_r,
            return_features=return_debug,
        )
        selected = selected_output[:4]
        selector_feature_m = (
            selected_output[4] if return_debug else None
        )
        selector_feature_r = (
            selected_output[5] if return_debug else None
        )
        gate_m, gate_r, logits_m, logits_r = self._override_or_select(
            selected, field_override, (mask_m, mask_r)
        )
        gate_m = _intervene_gate(gate_m, mask_m, field_intervention)
        gate_r = _intervene_gate(gate_r, mask_r, field_intervention)
        gated_m, gated_r = self.relation(local_m, local_r, gate_m, gate_r)
        global_token, pooled_m, pooled_r = self.pooling(
            gated_m, gated_r, gate_m, gate_r
        )
        tokens = torch.cat((global_token, pooled_m, pooled_r), dim=1)
        context = ContextEncoding(
            tokens=tokens,
            manipulated_motion_field=gate_m,
            reference_motion_field=gate_r,
            manipulated_motion_logits=logits_m,
            reference_motion_logits=logits_r,
            functional_tokens=tokens,
            manipulated_field_mask=mask_m,
            reference_field_mask=mask_r,
        )
        if not return_debug:
            return context
        return V7SceneEncoding(
            context=context,
            manipulated_local=local_m,
            reference_local=local_r,
            manipulated_relation=gated_m,
            reference_relation=gated_r,
            selector_feature_m=selector_feature_m,
            selector_feature_r=selector_feature_r,
            functional_tokens_m=pooled_m,
            functional_tokens_r=pooled_r,
            joint_token=global_token,
        )
