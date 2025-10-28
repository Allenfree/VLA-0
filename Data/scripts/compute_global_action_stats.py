#!/usr/bin/env python3
import os
import glob
import h5py
import numpy as np

DATA_ROOT = "/data/yzf_pro/pro_ject/VLA-0/Data/LIBERO_datasets"
OUT_ROOT  = "/data/yzf_pro/pro_ject/VLA-0/Data/processed_vla0"
os.makedirs(OUT_ROOT, exist_ok=True)

# 我们会遍历所有这些子目录
SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
]

def iter_all_hdf5():
    for suite in SUITES:
        suite_dir = os.path.join(DATA_ROOT, suite)
        if not os.path.isdir(suite_dir):
            continue
        for h5_path in glob.glob(os.path.join(suite_dir, "*.hdf5")):
            yield h5_path

def collect_all_actions():
    """把所有 demo_* 的 actions 堆起来"""
    all_actions_list = []
    for h5_path in iter_all_hdf5():
        with h5py.File(h5_path, "r") as f:
            # 文件结构: f["data"]["demo_0"]["actions"] ...
            data_group = f["data"]
            for demo_key in data_group.keys():
                demo_grp = data_group[demo_key]
                if "actions" not in demo_grp:
                    continue
                acts = np.array(demo_grp["actions"], dtype=np.float32)  # (T, 7)
                all_actions_list.append(acts)
    if len(all_actions_list) == 0:
        raise RuntimeError("No actions found in dataset. Check paths/keys.")
    all_actions = np.concatenate(all_actions_list, axis=0)  # (N_total, 7)
    return all_actions

def main():
    all_actions = collect_all_actions()
    mins = all_actions.min(axis=0)  # shape (7,)
    maxs = all_actions.max(axis=0)  # shape (7,)
    print("Global action mins:", mins)
    print("Global action maxs:", maxs)
    out_path = os.path.join(OUT_ROOT, "action_stats.npz")
    np.savez(out_path, mins=mins, maxs=maxs)
    print("Saved action stats to:", out_path)

if __name__ == "__main__":
    main()
