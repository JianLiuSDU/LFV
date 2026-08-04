# Visualization

Active helpers are deliberately output-only:

- `affordance_transfer.py` saves the fixed 2x3 source/target/V-Q-H diagnostic
  image for the pure 2D transfer stage, without Open3D or a GUI.
- `topdown_grasp_report.py` composes the fixed four-panel downstream report from
  already-rendered 2D heat, complete-surface heat, RGB grasp, and Open3D grasp
  images. It also creates the fixed two-instance affordance/grasp comparison
  used by the mug generalization regression.

Visualization consumes arrays or saved artifacts. It must not load DINO, read
experiment configuration, run lifting/GraspNet, or recompute matching.
