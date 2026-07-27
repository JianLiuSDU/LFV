# Common Model Components

Reusable neural-network modules should live here:

- point cloud encoders;
- point-DINO feature fusion;
- diffusion schedulers and denoisers shared across stages;
- SE(3), SO(3), rotation-6D, and trajectory utilities;
- conditioning and cross-attention blocks;
- losses and normalization helpers.

Older useful references are in `diffusion_policy_3d/model/` and
`diffusion_policy_3d/policy/`. Do not copy code blindly; migrate only the parts
that fit the new two-stage design.

