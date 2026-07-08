import os
# 在导入 transformers 之前设置 HF 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import glob
import traceback
import numcodecs
try:
    from imagecodecs.numcodecs import register_codecs
    register_codecs()
except ImportError:
    print("[!] 警告: 未能导入 imagecodecs，请确保已执行 pip install imagecodecs")

import zarr
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# ================= 配置区 =================
# 路径配置
SRC_ROOT = "/media/ljian/lj/data/realworld_human_demonstration/pouring"
DST_ROOT = "/media/ljian/lj/data_3d/pouring"

# 模型配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "IDEA-Research/grounding-dino-base"
OBJECT_NAME = "red bowl ."  # 注意：DINO 的提示词末尾建议保留句号

# 阈值配置
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3
# ==========================================

def show_box(box, ax):
    """可视化辅助函数，用于在图像上画框"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def create_symlink(src_path, dst_path):
    """创建软链接，如果已存在则跳过"""
    if os.path.exists(dst_path) or os.path.islink(dst_path):
        return
    if os.path.exists(src_path):
        os.symlink(src_path, dst_path)
    else:
        print(f"[!] 警告: 源路径不存在，无法创建链接 -> {src_path}")

def get_object_bbox(initial_frame, text, device, processor, model):
    """利用 Grounding DINO 获取目标框"""
    image = Image.fromarray(initial_frame)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )
    
    if len(results[0]["boxes"]) == 0:
        raise ValueError(f"未检测到文本描述: '{text}' 的物体。")
        
    # 获取置信度最高的一个框并转换为 numpy array: [x1, y1, x2, y2]
    best_box = results[0]["boxes"].detach().cpu().numpy()[0]
    return best_box

def main():
    print(f"[*] 正在加载 Grounding DINO 模型: {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
    print("[✔] 模型加载完成！")

    # 获取所有源 episode 文件夹
    #episode_dirs = sorted(glob.glob(os.path.join(SRC_ROOT, "episode_*")))
    # 获取所有源 episode 文件夹 (仅保留目录，过滤掉.png等文件)
    all_matches = glob.glob(os.path.join(SRC_ROOT, "episode_*"))
    episode_dirs = sorted([d for d in all_matches if os.path.isdir(d)])
    total_eps = len(episode_dirs)
    print(f"[*] 发现 {total_eps} 个 episode 待处理。\n")

    os.makedirs(DST_ROOT, exist_ok=True)
    failed_episodes = []

    for idx, src_ep_path in enumerate(episode_dirs):
        ep_name = os.path.basename(src_ep_path)
        print(f"--- 进度 [{idx+1}/{total_eps}]: 正在处理 {ep_name} ---")
        
        # 1. 构建目标目录树
        dst_ep_path = os.path.join(DST_ROOT, ep_name)
        dst_bbox_dir = os.path.join(dst_ep_path, "target_bbox")
        dst_viz_dir = os.path.join(dst_ep_path, "viz")
        
        os.makedirs(dst_ep_path, exist_ok=True)
        os.makedirs(dst_bbox_dir, exist_ok=True)
        os.makedirs(dst_viz_dir, exist_ok=True)

        # 源文件路径
        src_camera_dir = os.path.join(src_ep_path, "camera_0")
        src_rgb = os.path.join(src_camera_dir, "rgb")
        src_depth = os.path.join(src_camera_dir, "depth")
        src_mp4 = os.path.join(src_ep_path, "camera_0.mp4")    # <--- 改为直接从 src_ep_path 获取

        # 2. 建立软链接 (瞬间完成数据迁移)
        create_symlink(src_rgb, os.path.join(dst_ep_path, "rgb"))
        create_symlink(src_depth, os.path.join(dst_ep_path, "depth"))
        create_symlink(src_mp4, os.path.join(dst_ep_path, "camera_0.mp4"))

        # 3. 读取第一帧进行检测
        try:
            # 直接读取软链接指向的 rgb zarr 数据
            rgb_zarr_path = os.path.join(dst_ep_path, "rgb")
            episode_rgb = zarr.open(rgb_zarr_path, mode="r")
            initial_frame = episode_rgb[0]
            
            # 执行 DINO 检测
            bbox = get_object_bbox(initial_frame, OBJECT_NAME, DEVICE, processor, model)
            
            # 4. 保存 BBox 数据为 npy 格式 (纯粹、轻量)
            bbox_save_path = os.path.join(dst_bbox_dir, "target_bbox.npy")
            np.save(bbox_save_path, bbox)
            
            # 5. 可视化并保存
            viz_save_path = os.path.join(dst_viz_dir, "target_dino_detection.png")
            plt.figure(figsize=(8, 6))
            plt.imshow(initial_frame)
            show_box(bbox, plt.gca())
            plt.axis('off')
            plt.title(f"{ep_name} | {OBJECT_NAME}")
            plt.savefig(viz_save_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            
            print(f"[✔] {ep_name} 处理成功！Box: {bbox.astype(int)}")
            
        except Exception as e:
            print(f"[✘] {ep_name} 处理失败: {str(e)}")
            failed_episodes.append((ep_name, str(e)))
            # trace 打印可以帮助你调试，不影响下一个循环
            # traceback.print_exc() 

    # 总结报告
    print("\n================ 批量处理完成 ================")
    print(f"总计处理: {total_eps}，成功: {total_eps - len(failed_episodes)}，失败: {len(failed_episodes)}")
    if failed_episodes:
        print("失败的 Episode 列表:")
        for ep, err in failed_episodes:
            print(f" - {ep}: {err}")
        
        # 写入日志文件方便后续单独处理
        with open(os.path.join(DST_ROOT, "dino_failed_logs.txt"), "w") as f:
            for ep, err in failed_episodes:
                f.write(f"{ep}: {err}\n")

if __name__ == "__main__":
    main()