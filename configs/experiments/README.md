# Experiment Configs

`functional_motion/pouring_cup_far_execution.yaml` is the fixed learned-motion
rollout configuration. It joins saved affordance/GraspNet outputs to the
trained pouring GoalPose and Full64 checkpoints, then records ManiSkill Panda
execution. Training-free image transfer remains in `configs/affordance_transfer/`.

`functional_motion/drawer_open_episode60_front_topdown_execution.yaml` is the
authoritative drawer v2 configuration.  It fixes the real-data-aligned front
camera/yaw=0 scene, episode-60 transfer, geometry-verified top-down GraspNet
grasp, trained drawer GoalPose/Full64 checkpoints, drawer-specific pre-shaped
long fingers, prismatic-axis safety projection, and synchronized front/oblique
recording under `lfv_runs/drawer_open_v2/front_seed_0`.
