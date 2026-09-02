#!/usr/bin/env python3
"""Create machine-readable Gate A--D status artifacts.

Gate B--D are intentionally marked NOT_RUN when Gate A fails or is
inconclusive.  This prevents a convenient target result from being reported
before the local functional bottleneck has been established.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-a-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        default="<configured pouring cache>",
        help="Optional label stored in the portable report; no host path is required.",
    )
    args = parser.parse_args()
    gate_a_path = args.gate_a_report.expanduser().resolve()
    gate_a = json.loads(gate_a_path.read_text(encoding="utf-8"))
    gate_a_status = str(gate_a.get("status", "INCONCLUSIVE"))
    downstream = "NOT_RUN" if gate_a_status != "PASS" else "PENDING"
    rows = [
        {
            "gate": "A",
            "status": gate_a_status,
            "reason": "computation graph, task-loss gradients, interventions and fixed-batch overfit",
        },
        {
            "gate": "B",
            "status": downstream,
            "reason": "blocked until Gate A passes; no formal source training launched",
        },
        {
            "gate": "C",
            "status": downstream,
            "reason": "blocked until Gate B passes; canonical consistency not evaluated",
        },
        {
            "gate": "D",
            "status": downstream,
            "reason": "blocked until Gate C passes; strict target-instance transfer not evaluated",
        },
    ]
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "gate_a_report": gate_a_path.name,
        "gates": rows,
        "decision_rule": "Do not start downstream gates unless the previous gate is PASS.",
        "source_data": {
            "cache_root": str(args.cache_root),
            "records": 179,
            "split": {"train": 143, "val": 18, "test": 18},
            "object_instance_id": "empty in legacy cache; strict cross-instance evaluation unavailable",
        },
        "artifacts": {
            "gate_a_report": gate_a_path.name,
            "gate_a_csv": "gate_a_interventions.csv",
            "gate_a_plot": "gate_a_interventions.png",
        },
    }
    (output / "gate_status.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output / "gate_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("gate", "status", "reason"))
        writer.writeheader()
        writer.writerows(rows)
    colors = {"PASS": "tab:green", "FAIL": "tab:red", "INCONCLUSIVE": "tab:orange", "NOT_RUN": "tab:gray", "PENDING": "tab:blue"}
    figure, axis = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    for index, row in enumerate(rows):
        axis.bar(index, 1, color=colors.get(row["status"], "tab:gray"))
        axis.text(index, 0.5, row["status"], ha="center", va="center", color="white", fontweight="bold")
    axis.set_xticks(range(len(rows)), [row["gate"] for row in rows])
    axis.set_ylim(0, 1.2)
    axis.set_yticks([])
    axis.set_title("LFV Stage 2 V7 validation gates")
    figure.savefig(output / "gate_status.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
