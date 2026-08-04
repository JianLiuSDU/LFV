# Tools

This directory is for one-off inspection, validation, and debugging scripts.

Scripts here may be episode-specific or experimental. Once a tool becomes a
stable pipeline or model workflow, move the implementation into `lfv/` and keep
only a thin wrapper under `scripts/`.

Current important tools:

- `check_hand_pouring_contact_batch.py`
- `check_contact_field_outputs.py`
- `check_pipeline_outputs.py`

Stable Soft Heatmap AffCorrs commands live under
`scripts/affordance_transfer/`, not in this one-off tools directory.
