import torch
from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from vla0_dataset import VLA0Dataset, VLACollator

MANIFEST = "/data/yzf_pro/pro_ject/VLA-0/Data/processed_vla0/train/manifest.jsonl"
MODEL_DIR = "/data/yzf_pro/pro_ject/VLA-0/Model/Qwen2.5-VL-3B-Instruct"

def main():
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    ds = VLA0Dataset(MANIFEST)
    collator = VLACollator(processor)

    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collator)

    batch = next(iter(loader))

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 直接把 batch 解包给模型；多余键会被 **kwargs 接住
    with torch.cuda.amp.autocast(dtype=torch.float16):
        out = model(**{k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()})

    print("sanity loss:", float(out.loss))

if __name__ == "__main__":
    main()
