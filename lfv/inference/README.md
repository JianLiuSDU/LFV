# Learned Functional-Motion Inference

`functional_motion/two_stage_pouring.py` owns the saved-data contract and frame
conversion for the historical trained GoalPose and Full64 pouring models. The
heavy checkpoint adapter is `scripts/inference/infer_pouring_motion.py` so LFV
does not copy the old model implementation. Its fixed outputs are model-local
poses, absolute ManiSkill world object poses, an overlay and a JSON report.
