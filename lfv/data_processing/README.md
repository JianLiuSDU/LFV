# Data Processing

This package contains shared episode I/O used by the legacy RGB-D source-data
preparation pipeline.

Current working processors remain in `lfv/pipeline/` and should not be moved
until the new interfaces are stable. The migration target is:

- video/RGB-D IO adapters;
- object and hand segmentation stages;
- contact timing and anchor selection;
- contact heat field generation;
- DINO feature extraction;
- reusable visualization exporters.

The active image-transfer algorithm has its own schema and preprocessing under
`lfv/affordance_transfer/`.
