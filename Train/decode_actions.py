# decode_actions.py
import re
import numpy as np
import json
import os

def parse_action_text(text, H, D):
    """
    从模型输出里提取整数token（忽略非数字/非X），
    返回 shape [H, D] 的 np.int32，长度不够就截断/补零。
    """
    # 允许X或XXX作为mask字符，先把它们当成随机/中值策略，简单起见这里先把X看成500
    cleaned_tokens = []
    for tok in re.findall(r"[0-9X]+", text):
        if "X" in tok:
            cleaned_tokens.append("500")  # 简单平滑填充
        else:
            cleaned_tokens.append(tok)

    nums = [int(x) for x in cleaned_tokens]
    needed = H * D

    if len(nums) < needed:
        nums = nums + [500] * (needed - len(nums))  # 不足补中值
    else:
        nums = nums[:needed]

    arr = np.array(nums, dtype=np.int32).reshape(H, D)
    return arr  # [H, D] in 0..1000

def load_action_stats(stats_path):
    d = np.load(stats_path)
    return d["action_mins"], d["action_maxs"]  # shape [D], [D]

def dequantize(int_actions, action_mins, action_maxs):
    """
    int_actions: [H, D] in 0..1000
    returns: [H, D] float32 in original control space
    """
    H, D = int_actions.shape
    out = np.zeros((H, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = (
            (int_actions[:, d] / 1000.0)
            * (action_maxs[d] - action_mins[d])
            + action_mins[d]
        )
    return out

if __name__ == "__main__":
    dummy_text = "509 500 651 472 501 503 0 556 431 ..."  # 举例
    H = 8
    D = 7
    stats_path = "/data/yzf_pro/pro_ject/VLA-0/Data/processed_vla0/action_stats.npz"

    mins, maxs = load_action_stats(stats_path)
    arr_int = parse_action_text(dummy_text, H, D)
    arr_cont = dequantize(arr_int, mins, maxs)

    print("int actions:\n", arr_int)
    print("continuous actions:\n", arr_cont)
