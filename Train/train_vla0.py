import os
import glob
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    get_scheduler,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from vla0_dataset import VLA0Dataset, VLACollator


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=int, default=64)

    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)

    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=50)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--report_to", type=str, default="none")

    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    return parser.parse_args()


def prune_checkpoints(output_dir, limit, accelerator):
    """
    保留最近的 limit 份 checkpoint，其余删掉，避免把磁盘打满
    """
    if limit is None or limit <= 0:
        return
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "step_*")) +
        glob.glob(os.path.join(output_dir, "epoch_*")),
        key=os.path.getmtime,
    )
    if len(ckpts) > limit:
        to_delete = ckpts[0 : len(ckpts) - limit]
        for path in to_delete:
            accelerator.print(f"Pruning old checkpoint: {path}")
            try:
                import shutil
                shutil.rmtree(path)
            except Exception as e:
                accelerator.print(f"Warning: failed to remove {path}: {e}")


def main():
    args = parse_args()

    # 让 Accelerate 走 bf16 路线（不会用 fp16 GradScaler，不会和我们打架）
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=None if args.report_to == "none" else args.report_to,
    )

    # rank0 创建输出目录
    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # 处理器：负责把多模态对话消息 + 图像拼起来
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
    )

    # 模型加载成 Qwen2.5-VL CausalLM，直接 bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map=None,  # 交给 accelerator.prepare 来分到多卡
    )

    # gradient checkpointing 省显存
    if args.gradient_checkpointing:
        accelerator.print(">>> Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False  # 必须关掉，否则会爆显存/报错

    # Dataset / Collator / DataLoader
    dataset = VLA0Dataset(args.train_manifest)
    collator = VLACollator(processor)

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )

    # 学习率调度相关的步数估计（在 prepare 前算，这里 len(dataloader) 还是全局的）
    steps_per_epoch = len(dataloader)
    total_update_steps = (
        steps_per_epoch * args.num_train_epochs
    ) // args.gradient_accumulation_steps
    warmup_steps = int(total_update_steps * args.warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    # 分布式准备：模型、优化器、dataloader、scheduler 全部交给 accelerator
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # 断点续训（可选）
    global_step = 0
    starting_epoch = 0
    if args.resume_from_checkpoint is not None:
        accelerator.print(f">>> Resuming from checkpoint {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        # 我们没有额外保存 global_step / epoch idx，这里就简单续跑

    model.train()

    for epoch in range(starting_epoch, args.num_train_epochs):
        accelerator.print(f"===== Epoch {epoch+1}/{args.num_train_epochs} =====")

        progress_bar = tqdm(
            dataloader,
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(progress_bar):
            # 梯度累积包装
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                # NaN 防护：一旦发现数值爆炸，立即停训，避免污染后续checkpoint
                if torch.isnan(loss):
                    accelerator.print("⚠️ loss is NaN, aborting this run to protect weights.")
                    return

                # 反向 + 优化
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 只有在真正同步梯度（也就是 accumulator 完成一个有效 global_step）后才更新日志/ckpt
            if accelerator.sync_gradients:
                global_step += 1

                # 打log
                if accelerator.is_local_main_process and (global_step % args.logging_steps == 0):
                    mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    accelerator.print(
                        f"[global_step {global_step}] "
                        f"loss={loss.item():.4f} "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e} "
                        f"mem={mem_gb:.2f}GB"
                    )

                # 存 checkpoint
                if accelerator.is_local_main_process and (global_step % args.save_steps == 0):
                    ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                    accelerator.save_state(ckpt_dir)
                    accelerator.print(f"✅ Saved checkpoint: {ckpt_dir}")
                    prune_checkpoints(args.output_dir, args.save_total_limit, accelerator)

            # tqdm 实时可视化
            if accelerator.is_local_main_process:
                progress_bar.set_description(f"loss {loss.item():.4f} | step {global_step}")

        # epoch 末尾也存一份
        if accelerator.is_local_main_process:
            ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            accelerator.save_state(ckpt_dir)
            accelerator.print(f"✅ Saved checkpoint: {ckpt_dir}")
            prune_checkpoints(args.output_dir, args.save_total_limit, accelerator)

    accelerator.print("🎯 Training completed.")


if __name__ == "__main__":
    main()
