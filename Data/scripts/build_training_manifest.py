#!/usr/bin/env python3
import os
import glob
import h5py
import numpy as np
import json
import cv2
import random

DATA_ROOT = "/data/yzf_pro/pro_ject/VLA-0/Data/LIBERO_datasets"
OUT_ROOT  = "/data/yzf_pro/pro_ject/VLA-0/Data/processed_vla0"
os.makedirs(OUT_ROOT, exist_ok=True)

SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
]

H = 8          # 预测未来多少步
ACTION_RES = 1000  # 动作离散化上界
MASK_RATIO = 0.2   # 动作字符mask概率

# 载入我们之前算好的全局动作范围
stats = np.load(os.path.join(OUT_ROOT, "action_stats.npz"))
ACTION_MINS = stats["mins"]  # shape (7,)
ACTION_MAXS = stats["maxs"]  # shape (7,)

def discretize_action_vec(vec, mins, maxs):
    """
    vec: (7,) float
    -> (7,) int32 in [0, ACTION_RES]
    """
    norm = (vec - mins) / (maxs - mins + 1e-6)
    scaled = np.round(norm * ACTION_RES)
    scaled = np.clip(scaled, 0, ACTION_RES)
    return scaled.astype(np.int32)

def mask_action_text(txt, ratio=MASK_RATIO):
    """
    对动作序列字符串逐字符mask，只有数字字符会被随机替成 'X'
    例如 "512 498" -> "5X2 4X8"
    """
    out_chars = []
    for ch in txt:
        if ch.isdigit() and random.random() < ratio:
            out_chars.append('X')
        else:
            out_chars.append(ch)
    return "".join(out_chars)

def instruction_from_filename(h5_path):
    """
    把文件名恢复成自然语言指令
    e.g.
    pick_up_the_black_bowl_..._place_it_on_the_plate_demo.hdf5
    -> "pick up the black bowl ... place it on the plate"
    """
    base = os.path.basename(h5_path)
    # 去掉尾部的 _demo.hdf5
    if base.endswith(".hdf5"):
        base = base[:-5]
    # 有些文件用 *_demo 结尾
    base = base.replace("_demo", "")
    # 下划线 -> 空格
    sent = base.replace("_", " ").strip()
    return sent

def ensure_out_dirs(split_name="train"):
    split_dir = os.path.join(OUT_ROOT, split_name)
    img_dir = os.path.join(split_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    return split_dir, img_dir

def iter_all_hdf5():
    for suite in SUITES:
        suite_dir = os.path.join(DATA_ROOT, suite)
        if not os.path.isdir(suite_dir):
            continue
        for h5_path in glob.glob(os.path.join(suite_dir, "*.hdf5")):
            yield h5_path

def process_one_demo(demo_grp, instr_text, split_dir, img_dir, manifest_f, sample_counter):
    """
    处理单个 demo_X group:
      - actions: (T,7) float64
      - obs/agentview_rgb: (T,128,128,3) uint8
      - obs/eye_in_hand_rgb: (T,128,128,3) uint8
    为每个 t 生成一条训练样本。
    """
    actions = np.array(demo_grp["actions"], dtype=np.float32)           # (T,7)
    rgb_agent = np.array(demo_grp["obs/agentview_rgb"], dtype=np.uint8) # (T,128,128,3)
    rgb_hand  = np.array(demo_grp["obs/eye_in_hand_rgb"], dtype=np.uint8)#(T,128,128,3)

    T, D = actions.shape
    assert D == ACTION_MINS.shape[0], f"Action dim mismatch: {D} vs {ACTION_MINS.shape[0]}"

    for t in range(T - H):
        # 1. 拼接图像 (两路视角横向拼)
        img_agent = rgb_agent[t]  # (128,128,3) RGB
        img_hand  = rgb_hand[t]   # (128,128,3) RGB
        tile_rgb  = np.concatenate([img_agent, img_hand], axis=1)  # (128,256,3) RGB

        # OpenCV写盘用BGR
        tile_bgr = tile_rgb[..., ::-1]

        # 2. 构造未来H步动作的整数序列
        future_ints = []
        for k in range(H):
            a_cont = actions[t + k]  # (7,)
            a_int  = discretize_action_vec(a_cont, ACTION_MINS, ACTION_MAXS)  # (7,)
            future_ints.append(a_int)
        future_ints = np.stack(future_ints, axis=0)  # (H,7)
        flat_list = future_ints.reshape(-1).tolist() # 长度 = H*7
        action_txt = " ".join(str(x) for x in flat_list)

        # 3. 做字符级mask增强
        masked_action_txt = mask_action_text(action_txt, ratio=MASK_RATIO)

        # 4. 保存图像到磁盘
        img_name = f"{sample_counter:08d}.png"
        img_path = os.path.join(img_dir, img_name)
        cv2.imwrite(img_path, tile_bgr)

        # 5. 写一行样本到 manifest.jsonl
        record = {
            "instruction": instr_text,             # 任务语言
            "image_path": img_path,                # 这帧对应的拼接相机视图
            "target_action_text": masked_action_txt, # 模型要学着生成的整数序列(带mask)
            "H": H,
            "D": D,
        }
        manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        sample_counter += 1

    return sample_counter

def main(split_name="train"):
    split_dir, img_dir = ensure_out_dirs(split_name)
    manifest_path = os.path.join(split_dir, "manifest.jsonl")
    sample_counter = 0

    with open(manifest_path, "w") as mf:
        for h5_path in iter_all_hdf5():
            instr_text = instruction_from_filename(h5_path)

            with h5py.File(h5_path, "r") as f:
                data_group = f["data"]
                for demo_key in data_group.keys():
                    demo_grp = data_group[demo_key]
                    # 防御式检查
                    if ("actions" not in demo_grp or
                        "obs/agentview_rgb" not in demo_grp or
                        "obs/eye_in_hand_rgb" not in demo_grp):
                        continue

                    sample_counter = process_one_demo(
                        demo_grp,
                        instr_text,
                        split_dir,
                        img_dir,
                        mf,
                        sample_counter
                    )

    print(f"Done. Wrote {sample_counter} samples to {manifest_path}")

if __name__ == "__main__":
    # 为了mask的随机性固定一下随机种子（可选）
    random.seed(1337)
    np.random.seed(1337)

    os.makedirs(OUT_ROOT, exist_ok=True)
    main(split_name="train")
