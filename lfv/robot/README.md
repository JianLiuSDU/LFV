# Robot Execution

`panda_grasp_execution.py` now provides the pure geometry used to combine a
GraspNet object-frame grasp and generated object motion into a Panda TCP path.
The ManiSkill rollout/MP4 adapter is
`scripts/robot/execute_pouring_motion_maniskill.py`.
The repeatable state-only TCP offset search is
`scripts/robot/calibrate_panda_grasp_offset.py`; it closes and lifts the mug for
each candidate, then writes every trial and the selected local correction to
JSON.

Current responsibilities:

- convert object-centric trajectories and grasp transforms into TCP
  trajectories;
- use ManiSkill absolute EE-pose IK;
- respect the mixed Panda action contract (absolute arm pose, normalized
  `+1/-1` open/close gripper scalar);
- preserve the rigid object-to-TCP transform along all 64 model waypoints;
- record grasp state, actual finger qpos, TCP tracking error, simulator task
  state, and synchronized oblique/front-view videos.

The execution config expresses `grasp_offset_local` in Panda TCP-local
`[orthogonal, closing, approach]` axes. The current Cole mug calibration is
`[0.005, -0.005, 0.0]` metres. This adapter is deliberately separate from the
predicted GraspNet row so geometry generation remains reproducible and robot
mount/contact calibration can be changed independently.

`gripper_extension.py` defines the simulator-independent geometry contract for
the optional `panda_long_finger` robot registered in
`lfv_sim/maniskill/robots/panda_long_finger.py`. It keeps the stock Panda
kinematics, 80 mm opening and TCP, but adds a 30 x 70 x 8 mm high-friction
plate to each moving finger. Visual and collision geometry are added to the
same articulation links, so `Panda.is_grasping` measures the extension's real
bilateral contact forces. The contact area is 6.49 times the stock
17.5 x 18.5 mm pad. This is a rigid first-pass proxy for a TPU/Fin-Ray finger,
not a deformable-body simulation.

`panda_drawer_finger` uses the same articulation/controller/TCP interface but
with a 16 x 70 x 4.5 mm plate shifted 30 mm farther down the finger axis.  It
retains 3.46x the stock pad area while fitting between a drawer handle and its
end supports.  Drawer execution additionally uses a configurable normalized
`approach_gripper_action` to pre-shape the jaw before top-down descent; the
full-close `-1` command is still sent only after reaching the grasp pose.

Multi-candidate feasibility ranking, explicit robot collision planning and
joint-limit scoring remain future work.
