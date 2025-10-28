# inference_policy.py
import torch
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image

# 你需要手动设定
H = 8        # 未来步数
D = 7        # 动作维度（你的数据里actions是shape (?,7)）
B = 1000     # 离散上界

SYSTEM_PROMPT = (
    f"You are a robot control policy. "
    f"Analyze the input observation and predict robot actions for the next {H} timesteps. "
    f"Each action has {D} dimensions. "
    f"Output a single sequence of {H*D} integers, where each integer is between 0 and {B}. "
    f"The integers should describe the {H} actions in temporal order, flattened. "
    "Provide only space-separated numbers. Nothing else."
)

def build_messages(img_paths, instruction):
    # img_paths: list[str]，和训练时一致（比如 [agentview.png, wrist.png]）
    # instruction: str 任务语言指令

    # Qwen-VL接受 "file://path" 形式
    content_blocks = []
    for p in img_paths:
        content_blocks.append({"type": "image", "image": "file://" + p})
    content_blocks.append({"type": "text", "text": f"Instruction: {instruction}"})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content_blocks},
    ]
    return messages

@torch.inference_mode()
def predict_actions(
    ckpt_dir,
    img_paths,
    instruction,
    max_new_tokens=512,
    temperature=0.1,
):
    device = "cuda:0"

    processor = AutoProcessor.from_pretrained(
        ckpt_dir,
        use_fast=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()

    # 构造对话
    messages = build_messages(img_paths, instruction)

    # 用和训练时同一套路的模板 & 视觉编码
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 重要：生成回答
    )
    imgs, vids = process_vision_info(messages)

    inputs = processor(
        text=[chat_text],
        images=[imgs],
        videos=[vids],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 生成
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False if temperature <= 0.1 else True,
        temperature=temperature,
    )

    # decode
    full_txt = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    return full_txt

if __name__ == "__main__":
    # 举例：手动塞一帧agent视角和一帧wrist视角（你可以从 processed_vla0/train/images/ 里随便拿）
    ckpt = "/data/yzf_pro/pro_ject/VLA-0/ckpts/vla0_stage1/epoch_1"  # 举例，等你有保存的checkpoint
    imgs = [
        "/data/yzf_pro/pro_ject/VLA-0/Data/processed_vla0/train/images/00000000.png",
        "/data/yzf_pro/pro_ject/VLA-0/Data/processed_vla0/train/images/00000001.png",
    ]
    instr = "pick up the black bowl on the wooden cabinet and place it on the plate"

    out = predict_actions(ckpt, imgs, instr)
    print("RAW MODEL OUTPUT:\n", out)
