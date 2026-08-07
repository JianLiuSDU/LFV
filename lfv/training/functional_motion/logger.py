"""TensorBoard logging with a no-op fallback."""

from __future__ import annotations

from pathlib import Path


class ScalarLogger:
    def __init__(self, output_dir: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(Path(output_dir) / "tensorboard"))
        except Exception as exc:
            print(f"[stage2] TensorBoard disabled: {exc}")
            self.writer = None

    def log(self, prefix: str, values: dict[str, float], step: int) -> None:
        if self.writer is None:
            return
        for key, value in values.items():
            self.writer.add_scalar(f"{prefix}/{key}", float(value), step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
