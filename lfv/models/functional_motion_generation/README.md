# Functional Motion Generation Network

This package will implement the second-stage model:

```text
manipulated object point cloud/features
+ reference target object point cloud/features
+ object-object geometry
-> object-centric multimodal SE(3) motion trajectory
```

The model should generate functional object motion in the fixed manipulated
object coordinate system so that different contact/grasp candidates can be
combined with different motion trajectories during robot feasibility selection.

Older trajectory-diffusion references live in:

- `scripts/model/`
- `diffusion_policy_3d/`
- `lfv/pipeline/se3_trajectory.py`

Implementation is intentionally left empty until the architecture is finalized.

