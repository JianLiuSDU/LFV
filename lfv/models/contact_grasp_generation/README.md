# Contact-Grasp Generation Network

This package will implement the first-stage model:

```text
manipulated object point cloud
+ point DINO features
+ normals / visibility / scale
-> contact heat field generation
-> task-related grasp pose generation
```

Planned outputs:

- point-wise contact heat field on the manipulated object;
- one or more robot parallel gripper candidates;
- grasp width, contact points, confidence, and quality scores.

Current data labels already available for this model:

- `contact_heatmap/contact_heatmap.npz`
- `dinov2_features/point_dinov2_features.npy`
- `hamer_grasp_pseudo_label/grasp_pseudo_label.npz`

Implementation is intentionally left empty until the network architecture is
finalized.

