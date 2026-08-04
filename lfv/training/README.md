# Training

There is no active training loop in the current Soft Heatmap AffCorrs stage.
The method uses a frozen DINOv2 encoder and deterministic matching.

Future learned modules may add trainers here, but they must not be coupled to
the current `lfv.affordance_transfer` interface.
