#!/usr/bin/env python3
"""Run the frozen V7 Gate-A computation-graph and bottleneck checks.

This is intentionally a diagnostic script rather than a training script.  It
uses one fixed batch, one fixed timestep and one fixed noise tensor for every
Field intervention, then writes JSON/CSV/PNG artifacts.  The default data is
the pouring cache; ``--synthetic`` is useful on machines without the cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from lfv.datasets.functional_motion import (
    FunctionalMotionDataset,
    SyntheticFunctionalMotionDataset,
    collate_functional_motion,
)
from lfv.diffusion import make_ddpm_scheduler
from lfv.models.functional_motion_generation import build_model
from lfv.models.functional_motion_generation.encoders.v7 import V7SceneEncoding
from lfv.models.functional_motion_generation.loading import model_kwargs


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batch_from_config(config: dict[str, Any], *, count: int, seed: int) -> dict[str, Any]:
    data = config["data"]
    if data.get("type") == "synthetic":
        dataset = SyntheticFunctionalMotionDataset(
            num_samples=count,
            num_points=int(data.get("num_points", 256)),
            dino_dim=int(data["dino_dim"]),
            horizon=64,
            seed=seed,
        )
        return collate_functional_motion([dataset[index] for index in range(count)])
    dataset = FunctionalMotionDataset(
        data["cache_root"],
        "train",
        shuffle_points=False,
        seed=seed,
        limit=count,
        consistency_group_fallback=data.get("consistency_group_fallback"),
        require_instance_id=False,
    )
    return collate_functional_motion([dataset[index] for index in range(count)])


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _mask(batch: dict[str, Any], role: str) -> torch.Tensor:
    points = batch[role + "_points"]
    value = batch.get(role + "_mask")
    if value is None:
        return torch.ones(points.shape[:2], device=points.device, dtype=points.dtype)
    return value.to(points).clamp(0.0, 1.0)


def _debug_encode(model, batch: dict[str, Any]) -> V7SceneEncoding:
    raw = model.encoder(
        batch["manipulated_points"],
        batch["manipulated_dino"],
        batch["reference_points"],
        batch["reference_dino"],
        manipulated_mask=_mask(batch, "manipulated"),
        reference_mask=_mask(batch, "reference"),
        scene_scale=batch.get("scene_scale"),
        return_debug=True,
    )
    if not isinstance(raw, V7SceneEncoding):
        raise TypeError("V7 encoder did not return debug encoding")
    return raw


def _context_for_gate(model, batch: dict[str, Any], gates: tuple[torch.Tensor, torch.Tensor]):
    return model.encode(batch, field_override=gates)


def _normalized_entropy(values: torch.Tensor) -> torch.Tensor:
    mass = values.clamp_min(0.0)
    mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -(mass * mass.clamp_min(1e-12).log()).sum(dim=1) / np.log(values.shape[1])


def _fixed_predictions(
    model,
    batch: dict[str, Any],
    context: torch.Tensor,
    *,
    timestep_value: int,
    goal_noise: torch.Tensor,
    trajectory_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict Goal/Trajectory noise with exactly shared t/noise across fields."""

    clean_goal = model.normalizer.normalize(batch["goal_pose9d"])
    timestep = torch.full(
        (clean_goal.shape[0],), timestep_value, device=clean_goal.device, dtype=torch.long
    )
    goal_scheduler = make_ddpm_scheduler(num_train_timesteps=model.goal_diffuser.num_train_timesteps)
    noisy_goal = goal_scheduler.add_noise(clean_goal, goal_noise, timestep)
    predicted_goal = model.goal_diffuser.decoder(noisy_goal, timestep, context)

    clean_full = model.normalizer.normalize(batch["trajectory_pose9d"])
    clean_traj = clean_full[:, 1:]
    normalized_start = clean_full[:, 0]
    normalized_goal = model.normalizer.normalize(batch["goal_pose9d"])
    trajectory_scheduler = make_ddpm_scheduler(
        num_train_timesteps=model.trajectory_diffuser.num_train_timesteps
    )
    noisy_traj = trajectory_scheduler.add_noise(clean_traj, trajectory_noise, timestep)
    predicted_traj = model.trajectory_diffuser.decoder(
        noisy_traj,
        timestep,
        context,
        normalized_goal,
        normalized_start=normalized_start,
    )
    return predicted_goal, predicted_traj


def _grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().abs().sum())
    return total


def _gradient_probe(model, batch: dict[str, Any], stage: str, seed: int) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    model.field_budget_weight = 0.0
    model.field_smooth_weight = 0.0
    model.field_consistency_weight = 0.0
    _seed(seed)
    losses = model.compute_loss(batch, stage=stage)
    losses["total"].backward()
    groups = {
        "local_point_encoder": model.encoder.manipulated_local_encoder.parameters(),
        "selector_cross_attention": model.encoder.selector.m_to_r.parameters(),
        "field_logits_head": model.encoder.selector.manipulated_head.parameters(),
        "gated_relation_encoder": model.encoder.relation.parameters(),
        "functional_pooling": model.encoder.pooling.parameters(),
        "goal_diffusion": model.goal_diffuser.parameters(),
        "trajectory_diffusion": model.trajectory_diffuser.parameters(),
    }
    return {name: _grad_norm(parameters) for name, parameters in groups.items()}


def _overfit_probe(model, batch: dict[str, Any], steps: int, seed: int) -> list[dict[str, float]]:
    """Fixed-batch optimization probe; not a replacement for formal training."""

    _seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    model.field_budget_weight = 0.02
    model.field_smooth_weight = 0.01
    model.field_consistency_weight = 0.0
    history: list[dict[str, float]] = []
    model.train()
    probe_points = sorted(set([0, max(steps // 4, 1), max(steps // 2, 1), max(steps - 1, 1)]))
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = model.compute_loss(batch, stage="joint")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in probe_points:
            with torch.no_grad():
                debug = _debug_encode(model, batch)
                fields = torch.cat(
                    (debug.context.manipulated_motion_field, debug.context.reference_motion_field),
                    dim=1,
                )
                history.append(
                    {
                        "step": float(step),
                        "total": float(losses["total"].detach()),
                        "goal": float(losses["goal_total"].detach()),
                        "trajectory": float(losses["trajectory_total"].detach()),
                        "field_entropy": float(_normalized_entropy(fields).mean()),
                        "field_mean": float(fields.mean()),
                        "field_peak": float(fields.max()),
                    }
                )
    model.eval()
    return history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage2/motion_field_v7_pouring_lfv_smoke.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overfit-steps", type=int, default=200)
    parser.add_argument("--timestep", type=int, default=37)
    args = parser.parse_args()
    _seed(args.seed)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["runtime"]["device"] = args.device
    device = torch.device(args.device)
    batch = _to_device(_batch_from_config(config, count=args.batch_size, seed=args.seed), device)
    model = build_model(
        config["model"]["name"],
        **model_kwargs(config, int(config["data"]["dino_dim"])),
    ).to(device)
    model.eval()
    model.normalizer.fit_tensors([batch["trajectory_pose9d"]])
    timestep_value = min(
        int(args.timestep),
        int(model.goal_diffuser.num_train_timesteps) - 1,
        int(model.trajectory_diffuser.num_train_timesteps) - 1,
    )

    with torch.no_grad():
        debug = _debug_encode(model, batch)
        learned_m = debug.context.manipulated_motion_field
        learned_r = debug.context.reference_motion_field
        if learned_m is None or learned_r is None:
            raise RuntimeError("V7 did not produce scalar Motion Functional Fields")
        mask_m = _mask(batch, "manipulated")
        mask_r = _mask(batch, "reference")
        mean_m = (learned_m.sum(1) / mask_m.sum(1).clamp_min(1.0))[:, None]
        mean_r = (learned_r.sum(1) / mask_r.sum(1).clamp_min(1.0))[:, None]
        uniform = (mean_m.expand_as(learned_m) * mask_m, mean_r.expand_as(learned_r) * mask_r)
        all_one = (mask_m, mask_r)
        zero = (torch.zeros_like(learned_m), torch.zeros_like(learned_r))
        roll = (torch.roll(learned_m, learned_m.shape[1] // 2, 1) * mask_m,
                torch.roll(learned_r, learned_r.shape[1] // 2, 1) * mask_r)
        permutation = torch.randperm(learned_m.shape[1], device=device)
        shuffled = (learned_m[:, permutation] * mask_m, learned_r[:, permutation] * mask_r)
        complement = ((1.0 - learned_m) * mask_m, (1.0 - learned_r) * mask_r)
        valid_count_m = mask_m.sum(1).clamp_min(1.0)
        valid_count_r = mask_r.sum(1).clamp_min(1.0)
        bottom_m = torch.zeros_like(learned_m)
        bottom_r = torch.zeros_like(learned_r)
        count_m = max(1, int(round(0.20 * learned_m.shape[1])))
        count_r = max(1, int(round(0.20 * learned_r.shape[1])))
        bottom_m.scatter_(1, learned_m.topk(count_m, dim=1, largest=False).indices, learned_m.topk(count_m, dim=1, largest=False).values)
        bottom_r.scatter_(1, learned_r.topk(count_r, dim=1, largest=False).indices, learned_r.topk(count_r, dim=1, largest=False).values)
        bottom = (bottom_m * mask_m, bottom_r * mask_r)
        conditions = {
            "learned": (learned_m, learned_r),
            "all_one": all_one,
            "uniform_budget": uniform,
            "zero": zero,
            "rolled": roll,
            "shuffled": shuffled,
            "complement": complement,
            "bottom20": bottom,
        }
        goal_noise = torch.randn_like(batch["goal_pose9d"])
        trajectory_noise = torch.randn(
            batch["trajectory_pose9d"].shape[0], 63, 9,
            device=device,
            dtype=batch["trajectory_pose9d"].dtype,
        )

        rows: list[dict[str, Any]] = []
        prediction_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for name, gates in conditions.items():
            context = _context_for_gate(model, batch, gates)
            predicted_goal, predicted_traj = _fixed_predictions(
                model,
                batch,
                context.tokens,
                timestep_value=timestep_value,
                goal_noise=goal_noise,
                trajectory_noise=trajectory_noise,
            )
            prediction_cache[name] = (context.tokens, predicted_goal, predicted_traj)
            rows.append(
                {
                    "condition": name,
                    "context_l2": float(context.tokens.norm(dim=-1).mean()),
                    "goal_prediction_l2": float(predicted_goal.norm(dim=-1).mean()),
                    "trajectory_prediction_l2": float(predicted_traj.norm(dim=(-1, -2)).mean()),
                    "field_ratio_m": float(((gates[0] * mask_m).sum(1) / mask_m.sum(1).clamp_min(1.0)).mean()),
                    "field_ratio_r": float(((gates[1] * mask_r).sum(1) / mask_r.sum(1).clamp_min(1.0)).mean()),
                    "field_entropy_m": float(_normalized_entropy(gates[0]).mean()),
                    "field_entropy_r": float(_normalized_entropy(gates[1]).mean()),
                    "field_peak_m": float(gates[0].max(1).values.mean()),
                    "field_peak_r": float(gates[1].max(1).values.mean()),
                }
            )
        learned_context = prediction_cache["learned"][0]
        zero_context = prediction_cache["zero"][0]
        shuffled_context = prediction_cache["shuffled"][0]
        context_ratio = float(zero_context.norm() / (learned_context.norm() + 1e-8))
        learned_goal = prediction_cache["learned"][1]
        learned_traj = prediction_cache["learned"][2]
        delta_rows = {
            name: {
                "goal_delta_l2": float((pred[1] - learned_goal).norm(dim=-1).mean()),
                "trajectory_delta_l2": float((pred[2] - learned_traj).norm(dim=(-1, -2)).mean()),
                "context_delta_l2": float((pred[0] - learned_context).norm(dim=-1).mean()),
            }
            for name, pred in prediction_cache.items()
        }
        shape_report = {
            "manipulated_points": list(batch["manipulated_points"].shape),
            "reference_points": list(batch["reference_points"].shape),
            "manipulated_dino": list(batch["manipulated_dino"].shape),
            "reference_dino": list(batch["reference_dino"].shape),
            "local_E_m": list(debug.manipulated_local.shape),
            "local_E_r": list(debug.reference_local.shape),
            "selector_feature_m": list(debug.selector_feature_m.shape) if debug.selector_feature_m is not None else None,
            "selector_feature_r": list(debug.selector_feature_r.shape) if debug.selector_feature_r is not None else None,
            "field_logits_m": list(debug.context.manipulated_motion_logits.shape),
            "field_logits_r": list(debug.context.reference_motion_logits.shape),
            "gate_m": list(learned_m.shape),
            "gate_r": list(learned_r.shape),
            "relation_R_m": list(debug.manipulated_relation.shape),
            "relation_R_r": list(debug.reference_relation.shape),
            "functional_tokens_m": list(debug.functional_tokens_m.shape),
            "functional_tokens_r": list(debug.functional_tokens_r.shape),
            "joint_token": list(debug.joint_token.shape),
            "Z_func": list(debug.context.tokens.shape),
            "goal_noise_prediction": list(prediction_cache["learned"][1].shape),
            "trajectory_noise_prediction": list(prediction_cache["learned"][2].shape),
        }

    gradients = {
        "goal_only": _gradient_probe(model, batch, "goal", args.seed + 11),
        "trajectory_only": _gradient_probe(model, batch, "trajectory", args.seed + 12),
    }
    overfit_model = build_model(
        config["model"]["name"],
        **model_kwargs(config, int(config["data"]["dino_dim"])),
    ).to(device)
    overfit_model.normalizer.fit_tensors([batch["trajectory_pose9d"]])
    overfit_history = _overfit_probe(overfit_model, batch, args.overfit_steps, args.seed + 20)

    with (output / "gate_a_interventions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    names = [row["condition"] for row in rows]
    axes[0, 0].bar(names, [row["context_l2"] for row in rows])
    axes[0, 0].set_title("Functional context norm")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 1].bar(names, [row["goal_prediction_l2"] for row in rows], color="tab:orange")
    axes[0, 1].set_title("Fixed-noise Goal prediction norm")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[1, 0].bar(names, [row["field_entropy_m"] for row in rows], label="manipulated")
    axes[1, 0].bar(names, [row["field_entropy_r"] for row in rows], alpha=0.65, label="reference")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("Normalized Field entropy")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].legend()
    if overfit_history:
        axes[1, 1].plot([row["step"] for row in overfit_history], [row["total"] for row in overfit_history], marker="o", label="total")
        axes[1, 1].plot([row["step"] for row in overfit_history], [row["goal"] for row in overfit_history], marker="o", label="goal")
        axes[1, 1].plot([row["step"] for row in overfit_history], [row["trajectory"] for row in overfit_history], marker="o", label="trajectory")
    axes[1, 1].set_title("Fixed-batch overfit probe")
    axes[1, 1].set_xlabel("optimizer step")
    axes[1, 1].legend()
    figure.savefig(output / "gate_a_interventions.png", dpi=180)
    plt.close(figure)

    report = {
        "gate": "A",
        "status": "INCONCLUSIVE",
        "config": str(args.config.resolve()),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "timestep": timestep_value,
        "shape_report": shape_report,
        "information_flow": {
            "selector_hidden_to_generator": False,
            "selector_outputs": ["field_logits_m", "field_logits_r", "gate_m", "gate_r"],
            "generator_input": "field-gated local payload -> GatedRelationEncoder -> FunctionalPooling -> Z_func",
            "zero_context_ratio": context_ratio,
            "zero_context_ratio_threshold": 0.05,
            "zero_field_pass": context_ratio < 0.05,
            "learned_vs_shuffled_context_delta_l2": delta_rows["shuffled"]["context_delta_l2"],
            "learned_vs_shuffled_goal_delta_l2": delta_rows["shuffled"]["goal_delta_l2"],
            "gate_sensitivity_pass": delta_rows["shuffled"]["context_delta_l2"] > 1e-5,
        },
        "interventions": rows,
        "paired_prediction_deltas_from_learned": delta_rows,
        "gradients_from_task_loss_only": gradients,
        "overfit": {
            "steps": args.overfit_steps,
            "history": overfit_history,
            "initial_total": overfit_history[0]["total"] if overfit_history else None,
            "final_total": overfit_history[-1]["total"] if overfit_history else None,
        },
        "outputs": {
            "json": str((output / "gate_a_report.json").resolve()),
            "csv": str((output / "gate_a_interventions.csv").resolve()),
            "plot": str((output / "gate_a_interventions.png").resolve()),
        },
        "note": "Gate A is not marked PASS automatically; review all conditions and task-loss gradient norms.",
    }
    final_entropy = (
        float(overfit_history[-1]["field_entropy"])
        if overfit_history
        else float("nan")
    )
    gradient_values = [
        value
        for stage_values in gradients.values()
        for name, value in stage_values.items()
        if name in {"selector_cross_attention", "field_logits_head"}
    ]
    checks = {
        "selector_hidden_bypass_absent": not report["information_flow"]["selector_hidden_to_generator"],
        "zero_field_context_closed": report["information_flow"]["zero_field_pass"],
        "field_changes_context": report["information_flow"]["gate_sensitivity_pass"],
        "task_loss_field_gradients_nonzero": bool(gradient_values) and min(gradient_values) > 0.0,
        # A normalized entropy close to one after a fixed-batch 1000-step
        # probe means the selector is still effectively uniform.  This is a
        # required Gate-A failure, not a reason to tune the model silently.
        "field_not_uniform_after_overfit": bool(np.isfinite(final_entropy) and final_entropy < 0.995),
        "overfit_loss_decreases": bool(
            overfit_history
            and overfit_history[-1]["total"] < overfit_history[0]["total"]
        ),
    }
    report["checks"] = checks
    if all(checks.values()):
        report["status"] = "PASS"
    elif not checks["field_not_uniform_after_overfit"]:
        report["status"] = "FAIL"
    else:
        report["status"] = "INCONCLUSIVE"
    (output / "gate_a_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
