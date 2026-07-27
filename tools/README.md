# Tools

This directory is for one-off inspection, validation, and debugging scripts.

Scripts here may be episode-specific or experimental. Once a tool becomes a
stable pipeline or model workflow, move the implementation into `lfv/` and keep
only a thin wrapper under `scripts/`.

Current important tools:

- `check_hand_pouring_contact_batch.py`
- `check_hand_pouring_grasp_batch.py`
- `verify_episode0_graspnet_contact_roi.py`
- `visualize_hamer_thumb_index_grasp_open3d.py`

