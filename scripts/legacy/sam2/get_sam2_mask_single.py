import os
import torch
import numpy as np
import cv2  # 用于提取轮廓
import matplotlib.pyplot as plt
from PIL import Image

# --- 核心：处理 Zarr 编解码器问题 ---
import numcodecs
try:
    from imagecodecs.numcodecs import register_codecs
    register_codecs()
    print("[*] 成功注册 imagecodecs 编解码器 (处理 JPEGXL)")
except ImportError:
    print("[!] 警告: 未能导入 imagecodecs，请确保已执行 pip install imagecodecs")

import zarr

# 导入 SAM 2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ================= 配置区 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPISODE_ROOT = "/home/users1/ljian/LFV/test/episode_0"

# 模型路径 (请确保路径正确)
SAM2_CHECKPOINT = "/home/users1/ljian/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml" 

# 输出文件名
MASK_NAME = "affordance_mask"  # 在 camera_0 下创建的 zarr 数组名
VIS_OVERLAY_PATH = os.path.join(EPISODE_ROOT, "camera_0/affordance_mask_overlay.png")
VIS_BINARY_PATH = os.path.join(EPISODE_ROOT, "camera_0/afffordance_mask_binary.png")
# ==========================================

def show_mask_and_border(mask, ax, color=(1, 1, 0, 0.5)):
    """在 plt ax 上绘制半透明掩码和明亮的轮廓线"""
    h, w = mask.shape[-2:]
    
    # 1. 绘制半透明填充
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    # 2. 绘制边界线 (使用 OpenCV 提取轮廓)
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        poly = cnt.reshape(-1, 2)
        if len(poly) > 2:
            ax.add_patch(plt.Polygon(poly, facecolor='none', edgecolor='yellow', linewidth=2))

def show_box(box, ax):
    """画出 DINO 的原始检测框"""
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='red', facecolor=(0,0,0,0), lw=1.5, linestyle='--'))

def main():
    # --- 1. 数据准备 ---
    print(f"[*] 正在读取 Zarr 数据: {EPISODE_ROOT}")
    episode_data = zarr.open(EPISODE_ROOT, mode="a")
    
    # 读取第一帧和 Bbox
    initial_frame = episode_data["camera_0/rgb"][0]
    bbox_data = episode_data["camera_0/affordance_bbox"][:]
    # 确保 bbox 格式为 [x1, y1, x2, y2]
    input_box = bbox_data[0] if bbox_data.ndim > 1 else bbox_data
    
    print(f"[*] 图像尺寸: {initial_frame.shape}, 输入 Box: {input_box}")

    # --- 2. 初始化 SAM 2.1 ---
    print(f"[*] 正在加载 SAM 2.1 模型...")
    # 使用 bfloat16 优化显存
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
        predictor = SAM2ImagePredictor(sam2_model)

        # --- 3. 执行推理 ---
        print("[*] 正在执行分割推理 (Multimask Mode)...")
        predictor.set_image(initial_frame)
        
        # 开启 multimask_output 以获得更稳健的结果
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True, 
        )
        
        # 挑选置信度最高的掩码
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        print(f"[*] 最高分割得分: {best_score:.4f}")

    # --- 4. 存储数据到 Zarr ---
    # 我们直接存入原 episode 的 Group 中，保持结构整齐
    if f"camera_0/{MASK_NAME}" in episode_data:
        print("[*] 正在覆盖旧的 Mask 数据...")
        episode_data[f"camera_0/{MASK_NAME}"][:] = best_mask
    else:
        print("[*] 正在创建新的 Mask 数据集...")
        episode_data.create_dataset(f"camera_0/{MASK_NAME}", data=best_mask, dtype='bool')

    # --- 5. 双重可视化保存 ---
    
    # A. 保存叠加图 (Overlay)
    plt.figure(figsize=(10, 8))
    plt.imshow(initial_frame)
    show_mask_and_border(best_mask, plt.gca()) # 黄色遮罩 + 黄色轮廓
    show_box(input_box, plt.gca())             # 红色虚线框
    plt.title(f"SAM 2.1: {best_score:.4f}")
    plt.axis('off')
    plt.savefig(VIS_OVERLAY_PATH, bbox_inches='tight', dpi=150)
    plt.close()

    # B. 保存纯二值化图 (Binary)
    # 这一步最关键：如果这个图还是看不见，说明分割本身失败了
    binary_img = (best_mask.astype(np.uint8)) * 255
    Image.fromarray(binary_img).save(VIS_BINARY_PATH)

    print(f"\n[✔] 处理完成！")
    print(f"    - 叠加可视化图: {VIS_OVERLAY_PATH}")
    print(f"    - 纯二值化掩码: {VIS_BINARY_PATH}")
    print(f"    - Zarr 存储位置: {EPISODE_ROOT}/camera_0/{MASK_NAME}")

if __name__ == "__main__":
    main()