# Hand Pouring DINO/SAM Processing

## Scope

This note records the preprocessing run for the cup-to-bowl pouring data:

- Raw data: `/media/ljian/lj/hand_data/pouring`
- Processed data: `/media/ljian/lj/data_3d/hand_pouring_lfv`
- Config: `configs/pipeline/hand_pouring.yaml`
- Episode count: 149

The processed directory is a separate LFV data folder under `/media/ljian/lj`. `prepare` creates episode directories and symlinks `rgb`, `depth`, `camera_0.mp4`, `meta.json`, and `timestamps.npy` back to the raw data, so the raw dataset is not modified and generated data is not mixed into the code repository.

## Prompts

For this pouring task the object roles are:

- Manipulated / affordance object: `cup .`
- Target object: `bowl .`

These prompts are configured under `objects.affordance.prompt` and `objects.target.prompt`.

## Completed Stages

The following stages were completed for all 149 episodes:

```bash
python scripts/run_pipeline.py --config configs/pipeline/hand_pouring.yaml --steps prepare
HF_ENDPOINT=https://hf-mirror.com python scripts/run_pipeline.py --config configs/pipeline/hand_pouring.yaml --steps dino
python scripts/run_pipeline.py --config configs/pipeline/hand_pouring.yaml --steps sam2
python scripts/run_pipeline.py --config configs/pipeline/hand_pouring.yaml --steps sample
```

Runtime environments used during this run:

- DINO: `/home/users1/ljian/anaconda3/envs/tapip3d/bin/python`
- SAM2: `/home/users1/ljian/anaconda3/envs/sam2/bin/python`
- Sampling/checks: `/home/users1/ljian/anaconda3/envs/tapip3d/bin/python`

The DINO run required `HF_ENDPOINT=https://hf-mirror.com` and network access for Hugging Face model files. DINO and SAM2 were run with GPU access.

## Output Files

Each episode now contains:

- `bbox/affordance_bbox.npy`
- `target_bbox/target_bbox.npy`
- `sam_mask/affordance_mask.npy`
- `target_sam_mask/target_mask.npy`
- `sample_points/sampled_2d_uniform.npy`
- `target_sample_points/target_sampled_2d_uniform.npy`
- Visual checks under `viz/`

Sampling is configured as:

```yaml
sampling:
  num_points: 512
```

The saved sample files contain `{"query_points_2d": points}` with shape `(512, 2)`.

## Verification

Final file counts:

- processed episodes: 149
- affordance DINO bbox: 149
- target DINO bbox: 149
- affordance SAM2 mask: 149
- target SAM2 mask: 149
- affordance sampled points: 149
- target sampled points: 149

Representative shape checks:

- `episode_0`: both point sets are `(512, 2)`, both have 512 unique points.
- `episode_61`: both point sets are `(512, 2)`, both have 512 unique points.
- `episode_149`: both point sets are `(512, 2)`, both have 512 unique points.

## Quality Notes

- `episode_0` DINO/SAM/sample visualizations were manually checked and look correct.
- `episode_61` has nearly identical DINO boxes for `cup .` and `bowl .`; this should be reviewed before using it for contact labeling or trajectory learning.
- The raw episode numbering is not strictly continuous: there is no `episode_147`, but there is `episode_149`. The processed folder mirrors the raw episode set exactly.

## Code Changes Made For This Run

- Added `configs/pipeline/hand_pouring.yaml`.
- Updated `lfv/pipeline/dino_bbox.py` to support both old and new `transformers` GroundingDINO post-process parameter names:
  - old API: `box_threshold`
  - new API: `threshold`

No training dataset or model code was modified for this preprocessing run.

## Hand, Contact Timing, And DINOv2 Update

Additional data-processing stages were added after the DINO/SAM object preprocessing:

- `hand_bbox`: DINO-based hand/manipulator bbox detection over sampled video frames.
- `hand_mask`: SAM2 hand/manipulator mask extraction from the saved hand bboxes.
- `timing`: automatic anchor-frame and first stable contact-window selection from hand/object mask geometry.
- `dinov2`: DINOv2 patch-grid and point-level semantic feature extraction for the anchor frame.

The new pipeline entries are registered in `scripts/run_pipeline.py`. Generated data is still written under:

```text
/media/ljian/lj/data_3d/hand_pouring_lfv
```

For this pouring data, the prompt `hand .` can falsely detect the cup or bowl in frames before the human hand enters. The `hand_bbox` stage therefore rejects hand boxes that mostly overlap the known first-frame cup or bowl masks:

```yaml
hand:
  reject_object_box_overlap_ratio: 0.45
  reject_object_mask_paths:
    - sam_mask/affordance_mask.npy
    - target_sam_mask/target_mask.npy
```

The DINOv2 feature stage uses the official DINOv2 small patch-14 weights downloaded to:

```text
/home/users1/ljian/LFV/third_party/dinov2_weights/dinov2_vits14_pretrain.pth
```

No third-party conda environment was modified. DINO bbox was run with the existing `tapip3d` environment, and SAM2 masks were run with the existing `sam2` environment. The combined `hand` stage remains available for an environment that has both dependency sets, but the practical current route is:

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python scripts/run_pipeline.py --config configs/pipeline/hand_pouring.yaml --steps hand_bbox --episodes 0 --overwrite
/home/users1/ljian/anaconda3/envs/sam2/bin/python scripts/run_pipeline.py --config configs/pipeline/hand_pouring.yaml --steps hand_mask --episodes 0 --overwrite
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python scripts/run_pipeline.py --config configs/pipeline/hand_pouring.yaml --steps timing --episodes 0 --overwrite
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python scripts/run_pipeline.py --config configs/pipeline/hand_pouring.yaml --steps dinov2 --episodes 0 --overwrite
```

### Episode 0 Verification

For `episode_0`, the current single-video validation outputs are:

- `hand_bbox`: 58 saved bbox frames out of 69 sampled frames.
- `hand_mask`: 58 saved mask frames out of 69 sampled frames.
- Contact timing: `anchor_frame=39`, `contact_start=45`, `contact_end=66`.
- Contact frames: `[45, 48, 51, 54, 57, 60, 63, 66]`.
- DINOv2 point features: `(512, 384)` float32, all finite.
- DINOv2 patch grid: `(35, 46, 384)` float32.

Main outputs:

```text
episode_0/
  hand_bbox/frame_*.npy
  hand_mask/frame_*.npy
  hand_contact/hand_bbox_meta.json
  hand_contact/hand_mask_meta.json
  contact_timing/contact_timing.json
  contact_timing/contact_timing_overlay.png
  dinov2_features/anchor_dinov2_grid.npz
  dinov2_features/point_dinov2_features.npy
  dinov2_features/point_pixels_uv.npy
  dinov2_features/dinov2_meta.json
```

The timing overlay was manually checked. The selected anchor frame is before contact, and the contact window covers the initial hand-cup interaction near the cup handle.

## Contact Heatmap MVP

Added a `contact_heatmap` stage for constructing a task-relevant contact heat label from a short contact window. This stage is intentionally separate from training dataset/model code.

The implementation reuses existing LFV utilities:

- `load_episode_camera_params` and `project_to_2d` from `lfv.pipeline.tracking`.
- `build_anchor_point_cloud`, `_load_se3_relative`, `_transform_anchor_points`, `_connected_components_high_heat`, and 3D visualization helpers from `lfv.pipeline.contact_field`.
- Existing `hand_mask`, `sam_mask`, `contact_timing`, RGB zarr and depth zarr outputs.

Current config uses only four frames around first contact:

```yaml
contact_heatmap:
  frame_offsets: [-3, 0, 3, 6]
  num_frames: 4
```

For `episode_0`, `contact_start=45`, so the selected frames are:

```text
[42, 45, 48, 51]
```

The computation is:

1. For each selected frame, compute hand distance transform.
2. Convert distance to object-pixel evidence with sigma scaled by the object bbox size.
3. Build anchor object point cloud with pixel correspondence.
4. Transform/project anchor points to each contact frame if SE(3) exists; otherwise use identity for the short four-frame MVP.
5. Query per-frame contact evidence at projected points.
6. Aggregate per-point evidence by Top-K mean instead of long-window averaging.
7. Keep seeds inside the anchor object mask and retain the main connected seed component.
8. Fit one weighted anisotropic Gaussian ellipse.
9. Mask the heatmap by the anchor object mask and normalize to `[0, 1]`.
10. Assign `contact_heat_i = H(pixel_i)` to anchor object points.
11. Apply KNN/normal constrained 3D high-heat connected-component correction.

Main outputs:

```text
episode_0/contact_heatmap/contact_heatmap.npz
episode_0/contact_heatmap/contact_heatmap_meta.json
episode_0/contact_heatmap/viz/frame_*_distance_evidence.png
episode_0/contact_heatmap/viz/frame_*_aligned_point_evidence.png
episode_0/contact_heatmap/viz/aggregated_point_evidence.png
episode_0/contact_heatmap/viz/contact_seeds.png
episode_0/contact_heatmap/viz/final_2d_elliptical_heatmap.png
episode_0/contact_heatmap/viz/raw_point_heat_3d.png
episode_0/contact_heatmap/viz/corrected_point_heat_3d.png
```

`episode_0` verification:

- Used contact frames: `[42, 45, 48, 51]`
- Anchor frame: `39`
- Contact seeds: `279`
- Heat area ratio: `0.000628`
- Ellipse center: `[161.65, 232.42]`
- Ellipse std axes: `[8.57, 5.37]` px
- Ellipse angle: `128.03` deg
- Seed connected components: `1`
- 3D high-heat components: `1`
- Point count: `4096`
- `per_frame_point_evidence`: `(4, 4096)`
- `contact_heat`: `(4096,)`, range `[0, 1]`

Manual visual check: the final 2D heatmap and corrected 3D point heat are concentrated on the cup handle side, matching the observed initial hand-cup interaction.

## Batch Script

Added a resumable batch script:

```bash
scripts/run_hand_pouring_contact_batch.sh
```

Default behavior processes all episodes and skips outputs that already exist:

```bash
scripts/run_hand_pouring_contact_batch.sh
```

Run selected episodes:

```bash
scripts/run_hand_pouring_contact_batch.sh --episodes 0 1 2
```

Force recomputation:

```bash
scripts/run_hand_pouring_contact_batch.sh --overwrite
```

The script runs stages in order:

```text
hand_bbox -> hand_mask -> timing -> dinov2 -> contact_heatmap
```

It uses:

- `/home/users1/ljian/anaconda3/envs/tapip3d/bin/python` for DINO hand bbox, timing, DINOv2, and contact heatmap.
- `/home/users1/ljian/anaconda3/envs/sam2/bin/python` for SAM2 hand masks.

Final summary is produced by:

```bash
tools/check_hand_pouring_contact_batch.py --config configs/pipeline/hand_pouring.yaml --show-bad
```

The check reports episode count, timing quality distribution, DINOv2 completion, contact heatmap completion, hand bbox/mask frame counts, and bad or incomplete episodes.
