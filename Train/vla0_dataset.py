import os, json, random
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# =====================================================
# System-level prompt（必须保留）
# =====================================================
SYSTEM_PROMPT = (
    "Analyze the input image and predict robot actions for the next H timesteps. "
    "Each action has D dimensions. Output a single sequence of H×D integers "
    "(0-1000 each), representing the H timesteps sequentially. "
    "Provide only space-separated numbers. Nothing else."
)

# =====================================================
# Dataset
# =====================================================
class VLA0Dataset(Dataset):
    def __init__(self, manifest_path):
        self.samples = []
        with open(manifest_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))
        print(f"[VLA0Dataset] loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path    = item["image_path"]
        instruction = item["instruction"]
        action_text = item["target_action_text"]

        # Qwen-VL 支持 file:// 协议
        img_uri = "file://" + img_path

        # 三段对话：system / user / assistant
        messages_full = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_uri},
                    {"type": "text", "text": f"Instruction: {instruction}"},
                ],
            },
            {
                "role": "assistant",
                "content": action_text,
            },
        ]

        # prefix 部分 = 不包含 assistant
        messages_prefix = messages_full[:-1]

        return {
            "messages_full": messages_full,
            "messages_prefix": messages_prefix,
        }


# =====================================================
# Collator
# =====================================================
class VLACollator:
    """
    collator 做几件事：
      1. 把每个样本的 messages 转成模板文本（apply_chat_template）
      2. 解析图像 (process_vision_info)
      3. 调用 processor 编码 input_ids / pixel_values
      4. 构造 labels，prefix 部分用 -100 屏蔽
    """
    def __init__(self, processor):
        self.processor = processor

    def _encode_messages(self, messages_batch):
        chat_texts = []
        images_list = []
        videos_list = []
        for messages in messages_batch:
            chat_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            chat_texts.append(chat_text)

            imgs, vids = process_vision_info(messages)
            images_list.append(imgs)

            # ✅ 修复：防止空视频导致 IndexError
            if vids and len(vids[0]) > 0:
                videos_list.append(vids)
            else:
                videos_list.append(None)

        return chat_texts, images_list, videos_list

    def __call__(self, batch):
        full_msgs   = [b["messages_full"]   for b in batch]
        prefix_msgs = [b["messages_prefix"] for b in batch]

        full_texts, full_imgs, full_vids = self._encode_messages(full_msgs)
        prefix_texts, prefix_imgs, prefix_vids = self._encode_messages(prefix_msgs)

        # ✅ 仅在存在视频时传入
        videos_full = full_vids if any(v is not None for v in full_vids) else None
        videos_pref = prefix_vids if any(v is not None for v in prefix_vids) else None

        full_inputs = self.processor(
            text=full_texts,
            images=full_imgs,
            videos=videos_full,
            padding=True,
            return_tensors="pt",
        )

        prefix_inputs = self.processor(
            text=prefix_texts,
            images=prefix_imgs,
            videos=videos_pref,
            padding=True,
            return_tensors="pt",
        )

        input_ids_full = full_inputs["input_ids"]
        attn_mask_full = full_inputs["attention_mask"]
        input_ids_pref = prefix_inputs["input_ids"]
        attn_mask_pref = prefix_inputs["attention_mask"]

        labels = input_ids_full.clone()
        for i in range(labels.size(0)):
            L_prefix = int(attn_mask_pref[i].sum().item())
            labels[i, :L_prefix] = -100  # prefix 不计 loss

        batch_out = {
            "input_ids": input_ids_full,
            "attention_mask": attn_mask_full,
            "labels": labels,
        }

        # 保留所有视觉特征字段
        for k, v in full_inputs.items():
            if k not in batch_out:
                batch_out[k] = v

        return batch_out


# =====================================================
# Debug 入口
# =====================================================
if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained(
        "/data/yzf_pro/pro_ject/VLA-0/Model/Qwen2.5-VL-3B-Instruct"
    )
    ds = VLA0Dataset("/data/yzf_pro/pro_ject/VLA-0/Data/processed_vla0/train/manifest.jsonl")
    collator = VLACollator(processor)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, collate_fn=collator)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, type(v), v.shape if torch.is_tensor(v) else None)
