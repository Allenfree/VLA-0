# VLA-0

VLA-0: Building State-of-the-Art VLAs with Zero Modification (Test)

> 复现目标：用开源多模态大模型（Qwen2.5-VL-3B-Instruct）直接作为机器人策略模型，
> 不加新头，不改 vocab，不写自定义 policy 网络，只通过监督微调让它输出机器人动作序列，
> 在 LIBERO 基准套件上达到 ~94% 成功率级别的通用操作能力。

本仓库当前是一个最小可运行骨架，包含我们为了复现 VLA-0 论文而实现的训练 / 推理流水线代码：
- 数据预处理（把机器人 demonstration 转成文本监督）
- 多模态 SFT 数据集封装
- 多卡训练脚本（`accelerate` + full finetune）
- 推理时的 ensemble 平滑与动作反归一化

⚠️ 状态说明：
- 这是“进行中的复现实现”（WIP），不是官方代码。
- 代码已经在两张 GPU 上完成了 sanity forward（loss 正常下降）和能跑训练 loop 的验证。
- 全量 64 epoch、8 卡 H100 版本会在下一台服务器上训练，这里提供的是骨架代码和环境说明。
- 由于存储限制，这个仓库目前**不包含：原始数据、预处理产物 (`manifest.jsonl` 等)、训练好的 checkpoint**。这些会在新机器上重新生成。

---

## 目录结构（当前计划）

```text
VLA-0/
├─ README.md
├─ LICENSE
├─ environment.yml        # conda 环境定义（建议放进仓库根目录）
│
├─ Train/
│   ├─ sanity_forward.py          # 单卡/小batch前向+loss检查，确认数据管道没炸
│   ├─ train_vla0.py              # 主训练脚本（accelerate 多卡 + 全量微调 + 保存ckpt）
│   ├─ vla0_dataset.py            # Dataset + Collator，把多模态对话样本整理成模型输入/labels
│   ├─ decode_actions.py          # 把模型输出的"0..1000"整数序列还原成连续动作 (反归一化辅助)
│   ├─ ensemble_buffer.py         # 推理时的滑动窗口集成(ensemble) 做动作平滑
│   ├─ inference_policy.py        # 推理/rollout时调用模型：取当前观测->让模型出H步动作->ensemble->反归一化
│
└─ Data/
    └─ scripts/
        ├─ compute_global_action_stats.py   # 扫所有 demonstration，统计每个动作维度的全局 min/max
        ├─ build_training_manifest.py       # 构建训练清单(manifest.jsonl)，把每个时间步变成一条 SFT 样本
        ├─ build_training_manifest_resume.py# 同上，但支持断点续扫大数据
        ├─ peek_manifest.py                 # 打开 manifest.jsonl 抽查样本正确性
