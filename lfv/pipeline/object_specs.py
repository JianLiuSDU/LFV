from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    prompt: str
    bbox_dir: str
    bbox_file: str
    mask_dir: str
    mask_file: str
    sample_dir: str
    sample_file: str
    viz_prefix: str


def iter_object_specs(cfg) -> list[ObjectSpec]:
    if "objects" not in cfg:
        obj = cfg.object
        return [
            ObjectSpec(
                name="target",
                prompt=str(obj.name),
                bbox_dir="target_bbox",
                bbox_file="target_bbox.npy",
                mask_dir="target_sam_mask",
                mask_file="target_mask.npy",
                sample_dir="target_sample_points",
                sample_file="target_sampled_2d_uniform.npy",
                viz_prefix="target",
            )
        ]

    specs = []
    for name, obj in cfg.objects.items():
        specs.append(
            ObjectSpec(
                name=name,
                prompt=str(obj.prompt),
                bbox_dir=str(obj.bbox_dir),
                bbox_file=str(obj.bbox_file),
                mask_dir=str(obj.mask_dir),
                mask_file=str(obj.mask_file),
                sample_dir=str(obj.sample_dir),
                sample_file=str(obj.sample_file),
                viz_prefix=str(obj.viz_prefix),
            )
        )
    return specs
