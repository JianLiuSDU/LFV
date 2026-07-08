# LFV

LFV collects the real-world data processing pipeline and model code used for
object-centric manipulation experiments.

The repository keeps source code, configuration, and lightweight metadata here.
Large datasets and checkpoints stay outside the repo and are referenced through
configuration files.

## Data Pipeline

Default pick-and-place paths are configured in:

```bash
configs/pipeline/picknplace.yaml
```

Run individual stages:

```bash
python scripts/run_pipeline.py --config configs/pipeline/picknplace.yaml --steps prepare dino sam2 sample track se3
```

The default raw dataset is:

```text
/media/ljian/lj/new_data/pickNplace
```

The default processed output is:

```text
/media/ljian/lj/data_3d/pickNplace_lfv
```

## Third-Party Code

Third-party code is stored under `third_party/`. LFV pipeline modules should
wrap those packages instead of editing them directly.

The point-cloud tracking stage uses TAPIP3D:

```text
third_party/tapip3d
```

The large TAPIP3D checkpoint is not copied into this repository. It is referenced
from `configs/pipeline/picknplace.yaml`.
