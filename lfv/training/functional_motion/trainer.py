"""AMP/EMA/checkpoint trainer for the hierarchical Stage 2 model."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lfv.diffusion import ExponentialMovingAverage

from .checkpoint import (
    load_checkpoint,
    restore_rng_state,
    rng_state,
    save_checkpoint,
)
from .logger import ScalarLogger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class FunctionalMotionTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config: dict,
        output_dir: str | Path,
        *,
        collate_fn=None,
    ) -> None:
        self.config = config
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config["runtime"]["device"])
        self.model = model.to(self.device)
        self.stage = str(config["training"].get("stage", "joint"))
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            num_workers=int(config["training"].get("num_workers", 0)),
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_fn,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["training"].get("num_workers", 0)),
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_fn,
            drop_last=False,
        )
        optimization = config["optimization"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(optimization["learning_rate"]),
            weight_decay=float(optimization.get("weight_decay", 1e-4)),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(config["training"]["epochs"])),
            eta_min=float(optimization.get("min_learning_rate", 1e-6)),
        )
        amp_enabled = bool(optimization.get("amp", True) and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.grad_clip = float(optimization.get("grad_clip_norm", 1.0))
        self.ema = ExponentialMovingAverage(
            self.model, decay=float(optimization.get("ema_decay", 0.999))
        )
        self.logger = ScalarLogger(self.output)
        self.start_epoch = 0
        self.global_step = 0
        self.best_value = float("inf")
        (self.output / "resolved_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def fit_normalizer(self) -> None:
        trajectories = [
            self.train_dataset[index]["trajectory_pose9d"]
            for index in range(len(self.train_dataset))
        ]
        self.model.normalizer.fit_tensors(trajectories)
        self.model.normalizer.save_json(self.output / "normalizer.json")

    def resume(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(path)
        self.model.load_state_dict(checkpoint["model"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_value = float(checkpoint.get("best_value", float("inf")))
        restore_rng_state(checkpoint["rng_state"])

    def _checkpoint_payload(self, epoch: int) -> dict:
        return {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "best_value": float(self.best_value),
            "config": self.config,
            "normalizer": self.model.normalizer.to_dict(),
            "rng_state": rng_state(),
            "registry_name": self.config["model"]["name"],
        }

    def _run_loader(self, loader, training: bool) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        count = 0
        self.model.train(training)
        for batch in loader:
            batch = _to_device(batch, self.device)
            batch_size = int(batch["goal_pose9d"].shape[0])
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    losses = self.model.compute_loss(batch, stage=self.stage)
                    loss = losses["total"]
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.ema.update(self.model)
                    self.global_step += 1
            for key, value in losses.items():
                totals[key] += float(value.detach()) * batch_size
            count += batch_size
        return {key: value / max(count, 1) for key, value in totals.items()}

    def train(self, resume: str | Path | None = None) -> dict:
        if resume:
            self.resume(resume)
        else:
            self.fit_normalizer()
        epochs = int(self.config["training"]["epochs"])
        patience = int(self.config["training"].get("early_stopping_patience", epochs))
        stale = 0
        history = []
        for epoch in range(self.start_epoch, epochs):
            if hasattr(self.train_dataset, "set_epoch"):
                self.train_dataset.set_epoch(epoch)
            if hasattr(self.model, "set_training_progress"):
                self.model.set_training_progress(
                    epoch / max(epochs - 1, 1)
                )
            train_metrics = self._run_loader(self.train_loader, True)
            # Diffusion validation also samples timesteps and Gaussian noise.
            # Evaluate every epoch against the same stochastic probe while
            # restoring the training RNG afterwards, so best-checkpoint
            # selection is comparable and does not perturb training.
            training_rng = rng_state()
            seed_everything(int(self.config["runtime"]["seed"]) + 1_000_003)
            try:
                with torch.no_grad():
                    val_metrics = self._run_loader(self.val_loader, False)
            finally:
                restore_rng_state(training_rng)
            self.lr_scheduler.step()
            self.logger.log("train", train_metrics, epoch)
            self.logger.log("val", val_metrics, epoch)
            current = float(val_metrics["total"])
            record = {
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            save_checkpoint(
                self.output / "checkpoints" / "last.pt",
                self._checkpoint_payload(epoch),
            )
            if current < self.best_value:
                self.best_value = current
                stale = 0
                save_checkpoint(
                    self.output / "checkpoints" / "best.pt",
                    self._checkpoint_payload(epoch),
                )
            else:
                stale += 1
            (self.output / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            if stale >= patience:
                print(f"[stage2] early stopping at epoch {epoch}")
                break
        self.logger.close()
        return {"best_value": self.best_value, "epochs": len(history)}
