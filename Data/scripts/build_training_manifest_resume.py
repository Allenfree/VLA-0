#!/usr/bin/env python3
import os
import glob
import h5py
import numpy as np
import json
import cv2
import random
from tqdm import tqdm

# --------- 配置区域 ---------
DATA_ROOT = "/data/yzf_pro/pro_ject/VLA-0/Data/LIBERO_datasets"
OUT_ROOT  = "/data/yzf_pro/pro_ject/VLA-0/Data/processed_vla0"

SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
]

H = 8               # 预测未来多少步
ACTION_RES = 1000   # 动作离散化上界
MASK_RATIO = 0.2    # 字符级mask概率
SPLIT_NAME = "train"

# -------------------------------------------------
# 加载全局动作范围 (mins / maxs)，必须先跑 compute_global_action_stats.py
stats_path = os.path.join(OUT_ROOT, "action_stats.npz")
stats = np.load(stats_path)
ACTION_MINS = stats["mins"]  # shape (D,)
ACTION_MAXS = stats["maxs"]  # shape (D,)
D_ACT = ACTION_MINS.shape[0] # 动作维度，比如7

# -------------------------------------------------
def instruction_from_filename(h5_path):
    """
    把文件名恢复成自然语言指令:
    pick_up_the_black_bowl_..._demo.hdf5
    -> "pick up the black bowl ... "
    """
    base = os.path.basename(h5_path)
    if base.endswith(".hdf5"):
        base = base[:-5]
    base = base.replace("_demo", "")
    sent = base.replace("_", " ").strip()
    return sent

def discretize_action_vec(vec, mins, maxs):
    """
    连续动作向量 (D,) -> [0, ACTION_RES] 的整数 (D,)
    """
    norm = (vec - mins) / (maxs - mins + 1e-6)
    scaled = np.round(norm * ACTION_RES)
    scaled = np.clip(scaled, 0, ACTION_RES)
    return scaled.astype(np.int32)

def mask_action_text(txt, ratio=MASK_RATIO):
    """
    随机把一部分数字字符替换成 'X'，用于动作掩码增强
    """
    out_chars = []
    for ch in txt:
        if ch.isdigit() and random.random() < ratio:
            out_chars.append('X')
        else:
            out_chars.append(ch)
    return "".join(out_chars)

def ensure_out_dirs(split_name=SPLIT_NAME):
    split_dir = os.path.join(OUT_ROOT, split_name)
    img_dir = os.path.join(split_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    return split_dir, img_dir

def list_all_hdf5_sorted():
    """
    返回所有 .hdf5 文件，排序后稳定
    """
    all_files = []
    for suite in SUITES:
        suite_dir = os.path.join(DATA_ROOT, suite)
        if not os.path.isdir(suite_dir):
            continue
        all_files.extend(glob.glob(os.path.join(suite_dir, "*.hdf5")))
    # 统一排序，确保重启后顺序一致
    all_files = sorted(all_files)
    return all_files

def build_all_jobs():
    """
    遍历所有 hdf5 -> data/demo_* -> 每个可训练时间步 t
    为每个训练样本生成一个 job:
      job = (h5_path, demo_key, t)
    然后把所有 job 排成一个全局列表
    """
    jobs = []
    for h5_path in list_all_hdf5_sorted():
        with h5py.File(h5_path, "r") as f:
            data_group = f["data"]
            # 为稳定，demo_key 也排序
            for demo_key in sorted(data_group.keys()):
                demo_grp = data_group[demo_key]
                if ("actions" not in demo_grp or
                    "obs/agentview_rgb" not in demo_grp or
                    "obs/eye_in_hand_rgb" not in demo_grp):
                    continue

                T = demo_grp["actions"].shape[0]
                for t in range(T - H):
                    jobs.append((h5_path, demo_key, t))
    return jobs

def count_existing_samples(manifest_path):
    """
    已经生成到第几个样本了？
    我们读取 manifest.jsonl（如果存在），数一数能成功 json 解析的行数。
    这就是我们下次开始的 global_idx 起点。
    这样可以断点续跑。
    """
    if not os.path.exists(manifest_path):
        return 0

    n = 0
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                n += 1
            except json.JSONDecodeError:
                # 最后一行如果被中断写了一半，就跳过
                break
    return n

def load_sample_data(h5_path, demo_key, t):
    """
    读取单个样本需要的原始数据:
    - 第 t 帧的双相机图像 (agentview_rgb[t], eye_in_hand_rgb[t])
    - 未来 H 步动作 actions[t : t+H]
    - 指令文本 (从文件名解析)
    """
    with h5py.File(h5_path, "r") as f:
        demo_grp = f["data"][demo_key]

        # 图像: RGB uint8
        rgb_agent = demo_grp["obs/agentview_rgb"][t]      # (128,128,3)
        rgb_hand  = demo_grp["obs/eye_in_hand_rgb"][t]    # (128,128,3)

        # 动作: float64 -> float32
        acts = demo_grp["actions"][t:t+H].astype(np.float32)  # (H, D)

    instr_text = instruction_from_filename(h5_path)
    return rgb_agent, rgb_hand, acts, instr_text

def build_one_sample(global_idx, rgb_agent, rgb_hand, acts, instr_text,
                     img_dir):
    """
    - 把两路图像横向拼接, RGB -> BGR, 写到磁盘
    - 把未来 H 步动作离散化并flatten成字符串，并做mask
    - 返回 manifest 里要写的一条记录(dict)
    """
    # 拼图像
    tile_rgb = np.concatenate([rgb_agent, rgb_hand], axis=1)  # (128,256,3) RGB
    tile_bgr = tile_rgb[..., ::-1]  # RGB->BGR for cv2

    img_name = f"{global_idx:08d}.png"
    img_path = os.path.join(img_dir, img_name)
    cv2.imwrite(img_path, tile_bgr)

    # 动作离散化
    future_ints = []
    for k in range(acts.shape[0]):      # k in [0..H-1]
        a_cont = acts[k]                # (D,)
        a_int  = discretize_action_vec(a_cont, ACTION_MINS, ACTION_MAXS)  # (D,)
        future_ints.append(a_int)
    future_ints = np.stack(future_ints, axis=0)   # (H,D)
    flat_list   = future_ints.reshape(-1).tolist()# 长度 = H*D

    action_txt = " ".join(str(x) for x in flat_list)
    masked_action_txt = mask_action_text(action_txt, ratio=MASK_RATIO)

    record = {
        "instruction": instr_text,
        "image_path": img_path,
        "target_action_text": masked_action_txt,
        "H": acts.shape[0],
        "D": acts.shape[1],
    }
    return record

def main():
    random.seed(1337)
    np.random.seed(1337)

    split_dir, img_dir = ensure_out_dirs(SPLIT_NAME)
    manifest_path = os.path.join(split_dir, "manifest.jsonl")

    # 1. 构建全量 job 列表
    jobs = build_all_jobs()
    total_jobs = len(jobs)
    print(f"[INFO] total samples to generate: {total_jobs}")

    # 2. 看我们已经做到哪
    start_idx = count_existing_samples(manifest_path)
    print(f"[INFO] resume from sample index: {start_idx}")

    # 3. 以 append 模式打开 manifest.jsonl
    mf = open(manifest_path, "a", buffering=1)

    # 4. 进度条，从 start_idx 跑到 total_jobs
    for global_idx in tqdm(range(start_idx, total_jobs), desc="Building samples", unit="sample"):
        h5_path, demo_key, t = jobs[global_idx]

        rgb_agent, rgb_hand, acts, instr_text = load_sample_data(h5_path, demo_key, t)

        rec = build_one_sample(
            global_idx,
            rgb_agent,
            rgb_hand,
            acts,
            instr_text,
            img_dir
        )

        # 写入一行JSON
        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    mf.close()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
