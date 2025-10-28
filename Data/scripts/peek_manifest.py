import os, json

MANIFEST = "/data/yzf_pro/pro_ject/VLA-0/Data/processed_vla0/train/manifest.jsonl"

with open(MANIFEST, "r") as f:
    for i, line in zip(range(2), f):  # 只看前2条
        sample = json.loads(line)
        print("="*80)
        print("sample id:", i)

        # 打印所有字段的类型和值（截断到前200字符防止太长）
        for k, v in sample.items():
            print(f"{k}: {type(v)} -> {str(v)[:200]}")

        img_path = sample["image_path"]
        print("image exists?", os.path.exists(img_path))
        print("="*80)
