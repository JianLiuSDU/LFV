import os
import glob
import traceback

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ================= 配置区 =================
SCENE_ROOT = os.path.expanduser("~/object_centric_diffusion/env_data/pouring_v1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAM2_CHECKPOINT = "/home/users1/ljian/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

IMAGE_NAME = "0000.png"
BBOX_FILENAME = "target_bbox.npy"
# ==========================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_rgb(scene_dir: str, image_name: str = IMAGE_NAME) -> np.ndarray:
    rgb_png = os.path.join(scene_dir, "rgb", image_name)
    rgb_npy = os.path.join(scene_dir, "rgb", image_name.replace(".png", ".npy"))

    if os.path.exists(rgb_png):
        return np.array(Image.open(rgb_png).convert("RGB"))
    if os.path.exists(rgb_npy):
        arr = np.load(rgb_npy)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    raise FileNotFoundError(f"未找到首帧图像: {rgb_png} 或 {rgb_npy}")


def show_box(box, ax):
    x0, y0, x1, y1 = box
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            edgecolor="red",
            facecolor=(0, 0, 0, 0),
            lw=1.5,
            linestyle="--",
        )
    )


def show_mask_and_border(mask, ax, color=(1, 1, 0, 0.5)):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)

    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        poly = cnt.reshape(-1, 2)
        if len(poly) > 2:
            ax.add_patch(
                plt.Polygon(poly, facecolor="none", edgecolor="yellow", linewidth=2)
            )


def main():
    print("[*] 加载 SAM2 模型")
    if DEVICE == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    with autocast_ctx:
        sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
        predictor = SAM2ImagePredictor(sam2_model)
    print("[✔] SAM2 加载完成")

    scene_dirs = sorted(
        [d for d in glob.glob(os.path.join(SCENE_ROOT, "scene_*")) if os.path.isdir(d)]
    )
    print(f"[*] 发现 {len(scene_dirs)} 个场景")

    failed = []

    for idx, scene_dir in enumerate(scene_dirs, start=1):
        scene_name = os.path.basename(scene_dir)
        print(f"\n--- [{idx}/{len(scene_dirs)}] 处理 {scene_name} ---")

        bbox_path = os.path.join(scene_dir, "target_bbox", BBOX_FILENAME)
        if not os.path.exists(bbox_path):
            print(f"[!] 跳过: 未找到 bbox -> {bbox_path}")
            failed.append((scene_name, "missing bbox"))
            continue

        mask_dir = os.path.join(scene_dir, "target_mask")
        ensure_dir(mask_dir)

        try:
            rgb = load_rgb(scene_dir)
            input_box = np.load(bbox_path).astype(np.float32)

            if DEVICE == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)

            with autocast_ctx:
                predictor.set_image(rgb)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=True,
                )

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])

            mask_npy_path = os.path.join(mask_dir, "target_mask.npy")
            mask_png_path = os.path.join(mask_dir, "target_mask.png")
            mask_overlay_path = os.path.join(mask_dir, "target_mask_overlay.png")

            np.save(mask_npy_path, best_mask)

            binary_img = (best_mask.astype(np.uint8)) * 255
            Image.fromarray(binary_img).save(mask_png_path)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(rgb)
            show_mask_and_border(best_mask, ax)
            show_box(input_box, ax)
            ax.set_title(f"{scene_name} | SAM2 score={best_score:.4f}")
            ax.axis("off")
            fig.savefig(mask_overlay_path, bbox_inches="tight", pad_inches=0, dpi=120)
            plt.close(fig)

            print(f"[✔] 成功: mask_score={best_score:.4f}")

        except Exception as e:
            print(f"[✘] 失败: {e}")
            failed.append((scene_name, str(e)))
            traceback.print_exc()

    print("\n================ 批量 SAM2 分割完成 ================")
    print(f"总数: {len(scene_dirs)} | 成功: {len(scene_dirs) - len(failed)} | 失败: {len(failed)}")

    if failed:
        log_path = os.path.join(SCENE_ROOT, "sam_failed_logs.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            for name, err in failed:
                f.write(f"{name}: {err}\n")
        print(f"[!] 失败日志已保存: {log_path}")


if __name__ == "__main__":
    main()