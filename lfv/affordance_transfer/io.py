from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import TransferResult


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "stored_in_npz": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_transfer_result(
    result: TransferResult,
    output_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    array_diagnostics = {
        key: value
        for key, value in result.diagnostics.items()
        if isinstance(value, np.ndarray)
    }
    npz_path = output_dir / "transfer_result.npz"
    np.savez_compressed(
        npz_path,
        target_heatmap=result.target_heatmap.astype(np.float32),
        target_heatmap_raw=result.target_heatmap_raw.astype(np.float32),
        **array_diagnostics,
    )
    method = str(result.diagnostics.get("method", "soft_heatmap_affcorrs"))
    uses_fgw = method == "affcorrs_fgw"
    report = {
        "schema_version": 2 if uses_fgw else 1,
        "stage": (
            "rgbd_affcorrs_fgw_contact_field_transport"
            if uses_fgw
            else "2d_soft_heatmap_affcorrs"
        ),
        "accepted": result.accepted,
        "rejection_reasons": result.rejection_reasons,
        "confidence": result.confidence,
        "diagnostics": result.diagnostics,
        "config": config,
        "scope": {
            "uses_target_rgb": True,
            "uses_target_mask": True,
            "uses_target_depth": uses_fgw,
            "uses_point_cloud": uses_fgw,
            "uses_graspnet": False,
        },
    }
    report_path = output_dir / "transfer_report.json"
    report_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"npz": npz_path, "report": report_path}
