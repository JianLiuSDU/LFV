# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# import numcodecs
# try:
#     from imagecodecs.numcodecs import register_codecs
#     register_codecs()
# except ImportError:
#     pass

# import zarr
# import torch
# import numpy as np
# from PIL import Image
# from matplotlib import pyplot as plt
# from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# # ================= 配置区 =================
# DST_ROOT = "/media/ljian/lj/data_3d/drawer_open"
# TARGET_EPISODES = [39]

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_ID = "IDEA-Research/grounding-dino-base"
# # ==========================================

# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2.5)) 

# def get_two_stage_bbox(initial_frame, device, processor, model):
#     """两阶段检测法：先找蓝盒子，裁剪后找把手"""
    
#     # ---------------- Stage 1: 找蓝盒子 ----------------
#     image = Image.fromarray(initial_frame)
#     inputs_box = processor(images=image, text="blue box.", return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         outputs_box = model(**inputs_box)
        
#     res_box = processor.post_process_grounded_object_detection(
#         outputs_box, inputs_box.input_ids, box_threshold=0.25, text_threshold=0.25, target_sizes=[image.size[::-1]]
#     )
    
#     if len(res_box[0]["boxes"]) == 0:
#         print("  [!] 第一阶段失败：找不到蓝盒子")
#         return None
        
#     # 获取蓝盒子的坐标
#     base_box = res_box[0]["boxes"][0].detach().cpu().numpy()
#     x1, y1, x2, y2 = map(int, base_box)
    
#     # 向外扩展 30 个像素的 Padding（防止把手凸出盒子边缘被裁掉）
#     pad = 30
#     H, W, _ = initial_frame.shape
#     crop_x1 = max(0, x1 - pad)
#     crop_y1 = max(0, y1 - pad)
#     crop_x2 = min(W, x2 + pad)
#     crop_y2 = min(H, y2 + pad)
    
#     # 执行裁剪
#     crop_img = initial_frame[crop_y1:crop_y2, crop_x1:crop_x2]
#     crop_pil = Image.fromarray(crop_img)
    
#     # ---------------- Stage 2: 在蓝盒子里找把手 ----------------
#     # 此时鸭子已经不在画面里了，可以用最简单的 prompt
#     inputs_handle = processor(images=crop_pil, text="handle.", return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         outputs_handle = model(**inputs_handle)
        
#     res_handle = processor.post_process_grounded_object_detection(
#         outputs_handle, inputs_handle.input_ids, box_threshold=0.15, text_threshold=0.15, target_sizes=[crop_pil.size[::-1]]
#     )
    
#     if len(res_handle[0]["boxes"]) == 0:
#         print("  [!] 第二阶段失败：在盒子上找不到把手")
#         return None
        
#     crop_boxes = res_handle[0]["boxes"].detach().cpu().numpy()
#     crop_area = crop_pil.width * crop_pil.height
    
#     # 寻找符合物理常识的框（把手面积不应超过裁剪区域的 40%）
#     final_local_box = crop_boxes[0] # 默认最高置信度
#     for b in crop_boxes:
#         cx1, cy1, cx2, cy2 = b
#         if (cx2 - cx1) * (cy2 - cy1) < crop_area * 0.40:
#             final_local_box = b
#             break
            
#     # 将局部坐标映射回原图的全局坐标
#     lcx1, lcy1, lcx2, lcy2 = final_local_box
#     global_box = np.array([
#         crop_x1 + lcx1,
#         crop_y1 + lcy1,
#         crop_x1 + lcx2,
#         crop_y1 + lcy2
#     ])
    
#     return global_box

# def main():
#     print(f"[*] 正在加载 Grounding DINO (准备执行两阶段精检测) ...")
#     processor = AutoProcessor.from_pretrained(MODEL_ID)
#     model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
#     print("[✔] 模型加载完成！")

#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.flatten()

#     for idx, ep_num in enumerate(TARGET_EPISODES):
#         ep_name = f"episode_{ep_num}"
#         ep_path = os.path.join(DST_ROOT, ep_name)
#         print(f"--- 正在重新处理 {ep_name} ---")

#         rgb_zarr_path = os.path.join(ep_path, "rgb")
#         episode_rgb = zarr.open(rgb_zarr_path, mode="r")
#         initial_frame = episode_rgb[0]

#         # 调用两阶段检测函数
#         bbox = get_two_stage_bbox(initial_frame, DEVICE, processor, model)

#         ax = axes[idx]
#         ax.imshow(initial_frame)
#         ax.axis('off')

#         if bbox is not None:
#             np.save(os.path.join(ep_path, "bbox", "affordance_bbox.npy"), bbox)
            
#             viz_save_path = os.path.join(ep_path, "viz", "dino_detection.png")
#             single_fig, single_ax = plt.subplots(figsize=(8, 6))
#             single_ax.imshow(initial_frame)
#             show_box(bbox, single_ax)
#             single_ax.axis('off')
#             single_ax.set_title(f"{ep_name} | Two-Stage Handle")
#             single_fig.savefig(viz_save_path, bbox_inches='tight', pad_inches=0, dpi=100)
#             plt.close(single_fig)

#             show_box(bbox, ax)
#             ax.set_title(f"{ep_name}: Success", color='green', fontsize=12)
#             print(f"[✔] 修正成功！Box: {bbox.astype(int)}")
#         else:
#             ax.set_title(f"{ep_name}: Failed", color='red', fontsize=12)
#             print(f"[✘] 依然失败。")

#     plt.tight_layout()
#     combined_path = os.path.join(DST_ROOT, "debug_dino_combined.png")
#     fig.savefig(combined_path, dpi=150)
#     plt.close(fig)

#     print(f"\n[✔] 任务完成！鸭子已被物理隔离。")
#     print(f"请运行指令查看效果： eog {combined_path}")

# if __name__ == "__main__":
#     main()
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numcodecs
try:
    from imagecodecs.numcodecs import register_codecs
    register_codecs()
except ImportError:
    pass

import zarr
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# ================= 配置区 =================
DST_ROOT = "/media/ljian/lj/data_3d/drawer_open" 
TARGET_EPISODES = [96, 97, 99, 100, 101]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "IDEA-Research/grounding-dino-base"

# Stage 1 精简咒语，用于快速锚定大概位置
OBJECT_NAME = "yellow box with a handle ."
# ==========================================

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2.5)) 

def get_two_stage_geometric_bbox(initial_frame, device, processor, model):
    """【两阶段+几何过滤法】：先划定粗糙区域，隔离干扰，再摳出聚焦图像内的扁平箱体"""
    
    # ---------------- Stage 1: 锚定粗略区域 (隔离画面右侧干扰) ----------------
    image = Image.fromarray(initial_frame)
    # 使用最高阈值和最精准 Prompt 找大概位置
    inputs_candidate = processor(images=image, text=OBJECT_NAME, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs_candidate = model(**inputs_candidate)
        
    res_candidate = processor.post_process_grounded_object_detection(
        outputs_candidate, inputs_candidate.input_ids, box_threshold=0.35, text_threshold=0.35, target_sizes=[image.size[::-1]]
    )
    
    if len(res_candidate[0]["boxes"]) == 0:
        print(f"  [!] 第一阶段失败：找不到描述为 '{OBJECT_NAME}' 的锚点区域")
        return None
        
    # 获取粗糙锚点区域坐标 (gx1, gy1, gx2, gy2)
    bbox_candidate = res_candidate[0]["boxes"][0].detach().cpu().numpy()
    gx1, gy1, gx2, gy2 = map(int, bbox_candidate)
    
    H, W, _ = initial_frame.shape
    
    # ---------------- 建立聚焦裁剪区域 (Focused Crop) ----------------
    # 加上一定的 Padding，确保物体完整
    pad = 20
    crop_x1 = max(0, gx1 - pad)
    crop_y1 = max(0, gy1 - pad)
    crop_x2 = min(W, gx2 + pad)
    crop_y2 = min(H, gy2 + pad)
    
    # 执行裁剪
    crop_img = initial_frame[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_pil = Image.fromarray(crop_img)
    
    # ---------------- Stage 2: 在聚焦区域内抠黄色主体 ----------------
    # 这里我们使用低阈值，重点是针对黄色塑料主体抠细节
    inputs_box = processor(images=crop_pil, text="yellow plastic drawer body with a handle.", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs_box = model(**inputs_box)
        
    # 稍微降低阈值，包容残缺
    res_box = processor.post_process_grounded_object_detection(
        outputs_box, inputs_box.input_ids, box_threshold=0.15, text_threshold=0.15, target_sizes=[crop_pil.size[::-1]]
    )
    
    if len(res_box[0]["boxes"]) == 0:
        print("  [!] 第二阶段失败：在聚焦区域内找不到主体。")
        return None
        
    local_boxes = res_box[0]["boxes"].detach().cpu().numpy()
    local_scores = res_box[0]["scores"].detach().cpu().numpy()
    
    valid_boxes = []
    valid_scores = []
    
    # ======= 【核心核心】几何形状过滤逻辑 =======
    # 我们知道我们要的是“扁平”的抽屉箱体，它的宽高比（Width/Height）必须远大于 1
    for box, score in zip(local_boxes, local_scores):
        lx1, ly1, lx2, ly2 = box
        w = lx2 - lx1
        h = ly2 - ly1
        
        # 物理限制：过滤宽高比不符合扁平特征（例如 < 1.8）的框，排除方箱子
        if h > 0 and w / h > 1.8:
            valid_boxes.append(box)
            valid_scores.append(score)
            
    if valid_boxes:
        # 过滤后有符合条件的框，选得分最高的
        best_idx = np.argmax(valid_scores)
        local_bbox = valid_boxes[best_idx]
        
        # 局部 -> 全局坐标映射
        lcx1, lcy1, lcx2, lcy2 = local_bbox
        global_box = np.array([crop_x1 + lcx1, crop_y1 + lcy1, crop_x1 + lcx2, crop_y1 + lcy2])
        return global_box
    else:
        # 过滤太严格导致没有符合条件的框，为了稳健，退回 Stage 1 的粗糙大框
        print("  [!] 警告：几何形状过滤未命中符合条件的扁平框，回退到粗糙大框")
        return bbox_candidate

def main():
    print(f"[*] 正在加载 Grounding DINO (准备执行两阶段精检测) ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
    print("[✔] 模型加载完成！")

    # 动态计算子图布局
    num_episodes = len(TARGET_EPISODES)
    fig, axes = plt.subplots(1, num_episodes, figsize=(num_episodes * 4, 3))
    
    if num_episodes == 1:
        axes = np.array([axes])

    for idx, ep_num in enumerate(TARGET_EPISODES):
        ep_name = f"episode_{ep_num}"
        ep_path = os.path.join(DST_ROOT, ep_name)
        print(f"\n--- [{idx+1}/{num_episodes}] 正在重新处理 {ep_name} (Two-Stage & Geometric) ---")

        rgb_zarr_path = os.path.join(ep_path, "rgb")
        
        if not os.path.exists(rgb_zarr_path):
            print(f"  [✘] 路径不存在，跳过: {rgb_zarr_path}")
            axes[idx].axis('off')
            continue

        try:
            episode_rgb = zarr.open(rgb_zarr_path, mode="r")
            initial_frame = episode_rgb[0]
        except Exception as e:
             print(f"  [✘] 读取 Zarr 失败: {e}")
             axes[idx].axis('off')
             continue

        # 调用新的两阶段修复函数
        bbox = get_two_stage_geometric_bbox(initial_frame, DEVICE, processor, model)

        ax = axes[idx]
        ax.imshow(initial_frame)
        ax.axis('off')

        if bbox is not None:
            # ======= 覆盖保存 refined BBox =======
            target_bbox_dir = os.path.join(ep_path, "target_bbox")
            os.makedirs(target_bbox_dir, exist_ok=True)
            np.save(os.path.join(target_bbox_dir, "affordance_bbox.npy"), bbox)
            
            # ======= 覆盖保存可视化 =======
            viz_dir = os.path.join(ep_path, "viz")
            os.makedirs(viz_dir, exist_ok=True)
            viz_save_path = os.path.join(viz_dir, "target_dino_detection.png")
            
            single_fig, single_ax = plt.subplots(figsize=(8, 6))
            single_ax.imshow(initial_frame)
            show_box(bbox, single_ax)
            single_ax.axis('off')
            single_ax.set_title(f"{ep_name} | Refined BBox")
            single_fig.savefig(viz_save_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(single_fig)

            # 更新总览图
            show_box(bbox, ax)
            ax.set_title(f"EP_{ep_num}", color='green', fontsize=10)
            print(f"  [✔] 修正成功！Box: {bbox.astype(int)}")
        else:
            ax.set_title(f"EP_{ep_num} Failed", color='red', fontsize=10)
            print(f"  [✘] 修正失败。")

    # 保存总览图方便快速确认所有 episode 的修正情况
    combined_path = os.path.join(DST_ROOT, "debug_dino_fixes_combined.png")
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)

    print(f"\n[✔] 终极修复清理测试完成！")
    print(f"请运行指令查看修复所有 episode 的效果： eog {combined_path}")

if __name__ == "__main__":
    main()