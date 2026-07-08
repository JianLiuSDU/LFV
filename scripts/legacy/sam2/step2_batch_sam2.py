import os
import glob
import torch
import numpy as np
import cv2  # 用于提取轮廓
import matplotlib.pyplot as plt
from PIL import Image

# --- 处理 Zarr 编解码器 ---
import numcodecs
try:
    from imagecodecs.numcodecs import register_codecs
    register_codecs()
except ImportError:
    print("[!] 警告: 未能导入 imagecodecs，请确保已执行 pip install imagecodecs")

import zarr
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ================= 配置区 =================
DST_ROOT = "/media/ljian/lj/data_3d/pouring"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 模型路径 (请确保路径正确)
SAM2_CHECKPOINT = "/home/users1/ljian/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml" 
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
    """画出 DINO 的输入检测框"""
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='red', facecolor=(0,0,0,0), lw=1.5, linestyle='--'))

def main():
    # --- 1. 全局初始化 SAM 2.1 ---
    print(f"[*] 正在加载 SAM 2.1 模型到显存 (仅加载一次)...")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
        predictor = SAM2ImagePredictor(sam2_model)
    print("[✔] SAM 2.1 模型加载完成！\n")

    # --- 2. 获取待处理列表 ---
    episode_dirs = sorted(glob.glob(os.path.join(DST_ROOT, "episode_*")))
    total_eps = len(episode_dirs)
    print(f"[*] 发现 {total_eps} 个 episode，开始批量分割。")

    failed_episodes = []

    # --- 3. 开始批量循环 ---
    for idx, ep_path in enumerate(episode_dirs):
        ep_name = os.path.basename(ep_path)
        print(f"--- 进度 [{idx+1}/{total_eps}]: 正在处理 {ep_name} ---")

        # 检查前置依赖：是否有 bbox
        bbox_path = os.path.join(ep_path, "target_bbox", "target_bbox.npy")
        if not os.path.exists(bbox_path):
            print(f"  [!] 跳过 {ep_name}: 未找到 BBox 文件，请检查第一阶段结果。")
            failed_episodes.append(ep_name)
            continue

        # 加载数据
        try:
            input_box = np.load(bbox_path)
            episode_rgb = zarr.open(os.path.join(ep_path, "rgb"), mode="r")
            initial_frame = episode_rgb[0]
            
            # --- 建立专属输出目录 ---
            mask_dir = os.path.join(ep_path, "target_sam_mask")
            viz_dir = os.path.join(ep_path, "viz")
            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(viz_dir, exist_ok=True)

            # --- 执行推理 ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictor.set_image(initial_frame)
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

            if best_score < 0.5:
                print(f"  [!] 警告: {ep_name} 分割得分较低 ({best_score:.4f})，可能分割质量不佳。")

            # --- 数据存储 (.npy 格式) ---
            # 我们将掩码保存为布尔型的 numpy 数组，读取极其方便且占用极小
            mask_save_path = os.path.join(mask_dir, "target_mask.npy")
            np.save(mask_save_path, best_mask)

            # --- 双重可视化保存 ---
            viz_overlay_path = os.path.join(viz_dir, "target_sam_overlay.png")
            viz_binary_path = os.path.join(viz_dir, "target_sam_binary.png")

            # A. 保存叠加图 (Overlay)
            fig_overlay, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(initial_frame)
            show_mask_and_border(best_mask, ax) 
            show_box(input_box, ax)            
            ax.set_title(f"{ep_name} | SAM 2.1 Score: {best_score:.4f}")
            ax.axis('off')
            fig_overlay.savefig(viz_overlay_path, bbox_inches='tight', dpi=150)
            plt.close(fig_overlay) # 防止内存泄漏

            # B. 保存纯二值化图 (Binary)
            binary_img = (best_mask.astype(np.uint8)) * 255
            Image.fromarray(binary_img).save(viz_binary_path)

            print(f"  [✔] 成功! 得分: {best_score:.4f}")

        except Exception as e:
            print(f"  [✘] {ep_name} 发生异常: {str(e)}")
            failed_episodes.append(ep_name)

    # 总结
    print("\n================ 批量 SAM2 分割完成 ================")
    print(f"成功: {total_eps - len(failed_episodes)} / {total_eps}")
    if failed_episodes:
        print(f"失败的 Episodes: {failed_episodes}")

if __name__ == "__main__":
    main()