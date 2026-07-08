import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from diffusion_policy_3d.model.clip.clip import load_clip, build_model, tokenize

def precompute_language_embedding(task_text, save_dir):
    print(f"正在加载 CLIP 模型，准备编码指令: '{task_text}'")
    
    # 1. 初始化 CLIP (完全复用原项目的做法)
    model, _ = load_clip("RN50", jit=False, device='cuda')
    clip_model = build_model(model.state_dict()).to('cuda')

    # 2. 文本分词
    tokens = tokenize(task_text).numpy()
    token_tensor = torch.from_numpy(tokens).to('cuda')

    # 3. 提取 1024 维特征
    with torch.no_grad():
        sentence_emb, _ = clip_model.encode_text_with_embeddings(token_tensor)

    # 4. 格式化为 [1, 1024] 形状 (关键：保留 1 维度，为了后续和 obs_horizon 对齐)
    lang_emb = sentence_emb[0].float().detach().cpu().numpy()[None, :]

    # 5. 保存到对应的数据集目录
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lang_emb.npy")
    np.save(save_path, lang_emb)
    print(f"✅ 成功保存语言特征到: {save_path}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config",
        default="configs/model/task/goal_pose_multitask.yaml",
        help="包含 instruction 和 dataset.data_dirs 的 task yaml。",
    )
    parser.add_argument("--text", default=None, help="覆盖 task config 中的 instruction。")
    parser.add_argument("--save-dir", default=None, help="覆盖 task config 中的第一个 dataset.data_dirs。")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.task_config)
    task_text = args.text if args.text is not None else str(cfg.instruction)
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        data_dirs = OmegaConf.to_container(cfg.dataset.data_dirs, resolve=True)
        save_dir = data_dirs[0] if isinstance(data_dirs, list) else data_dirs

    precompute_language_embedding(task_text, save_dir)


if __name__ == "__main__":
    main()
