# Data Processing

This package is reserved for the cleaned data-processing API for the new
two-stage framework.

Current working processors remain in `lfv/pipeline/` and should not be moved
until the new interfaces are stable. The migration target is:

- video/RGB-D IO adapters;
- object and hand segmentation stages;
- contact timing and anchor selection;
- contact heat field generation;
- HaMeR hand keypoint extraction;
- thumb-index grasp pseudo-label generation;
- DINO feature extraction;
- reusable visualization exporters.

The rule is: `lfv/pipeline/` remains the proven implementation; this package
will expose stable, model-facing preprocessing APIs after the labels are stable.

