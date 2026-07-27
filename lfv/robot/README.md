# Robot Feasibility Selection

This package is reserved for combining generated grasps and generated object
motions into executable robot end-effector trajectories.

Planned responsibilities:

- convert object-centric trajectories and grasp transforms into TCP
  trajectories;
- run continuous IK;
- score joint-limit margins, joint deltas, smoothness, and collisions;
- select the best feasible grasp-motion pair.

Simulation and GraspNet references currently live in `scripts/sim/` and
`lfv_sim/`.

