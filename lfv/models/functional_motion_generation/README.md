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

The package now contains two compatible model variants.  The original
`ThreeTokenHierarchicalDiffusion` remains available for V2/V6 checkpoints;
`V7FunctionalAlignmentDiffusion` is selected with the
`v7_functional_alignment` registry name and uses the source-canonical,
field-gated encoder in `encoders/v7.py`.  Both variants expose the same
`compute_loss` and `sample` interfaces.
