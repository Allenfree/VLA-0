# VLA-0

**VLA-0: Building State-of-the-Art VLAs with Zero Modification (Test)**

This repo is an attempt to reproduce the core ideas of **VLA-0** on the LIBERO manipulation benchmark using a standard multimodal VLM (Qwen2.5-VL-3B-Instruct) with *no architectural changes*.

The goal is to turn a general vision-language model into a robot policy that:
- takes an RGB observation and a language instruction,
- and directly outputs the next `H` robot actions
- as a single string of integers.

No new vocab. No extra policy head. No special tokens.  
Just full-parameter finetuning of an existing VLM.

---

## 1. What this repo does

The training objective is:
- Input to the model:
  - A system prompt telling it to act like a low-level controller.
  - The current visual observation (RGB image(s) from the robot / sim).
  - The high-level task instruction text.
- Target output:
  - A flat sequence of `H × D` integers, space-separated, where:
    - `H` = number of future timesteps we want to predict (e.g. 8),
    - `D` = action dimension (e.g. 7-8 DoF arm + gripper),
    - each integer is in `[0,1000]` and represents one dimension of the action after discretization.

Example supervision target (what the model is asked to generate):
```text
509 5X0 651 472 50X 50X 0 556 431 711 475 500 4X1 0 6X9 ...
```

Note: we apply *masked action augmentation* by randomly replacing digits with `X` characters (like `5X0`, `4X1`, etc.).
This forces the model to actually look at the image and instruction instead of purely copying previous tokens.

At inference time, the finetuned model is prompted with the *current observation only*.
It autoregressively generates the next `H` future actions (as text).
We parse those integers back to continuous control and execute them.

We also implement the sliding-window **ensemble smoothing** trick described by the VLA-0 paper:
to execute the current action at step `t`, we average multiple overlapping future predictions from the most recent `H` calls.
This stabilizes control and improves success.

---

## 2. Current status (WIP)

This repository is **work in progress**, not an official release.

What is already included:

* Data preprocessing scripts (`Data/scripts/`) to build the training manifest.
* Dataset + collator + trainer code (`Train/`) that runs under `accelerate` with bf16.
* Inference utilities to decode model output back into executable actions.
* The ensemble buffer for rollout smoothing.
* A `conda` environment spec (`environment.yml`).

What is **not** included:

* Raw LIBERO demos (frames, actions, instructions).
* Precomputed `manifest.jsonl` (the processed training set).
* Model checkpoints.
* Final evaluation rollouts.

You are expected to regenerate those on your own machine (e.g. on an 8×H100 server).

---

## 3. Repo layout

```text
VLA-0/
├─ README.md               # this file
├─ LICENSE                 # MIT license
├─ .gitignore
├─ environment.yml         # conda env spec (PyTorch + transformers + accelerate)

├─ Train/
│  ├─ vla0_dataset.py
│  │    - VLA0Dataset: loads the manifest.jsonl with (image_path, instruction, target_action_text, H, D).
│  │    - VLACollator: builds Qwen2.5-VL style chat messages:
│  │        [system prompt,
│  │         user(content=[image(s) + instruction]),
│  │         assistant(action-sequence-as-text)]
│  │      Uses `AutoProcessor` and `process_vision_info(...)`
│  │      to create tensors (input_ids, pixel_values, etc.)
│  │      and builds labels with causal masking.
│  │
│  ├─ train_vla0.py
│  │    - Main training script (multi-GPU via `accelerate`).
│  │    - Loads `Qwen2_5_VLForConditionalGeneration` from
│  │      `Qwen2.5-VL-3B-Instruct`.
│  │    - Uses bf16 mixed precision, gradient accumulation,
│  │      gradient checkpointing, cosine LR schedule, etc.
│  │    - Periodically logs loss / lr / GPU memory and saves checkpoints.
│  │
│  ├─ sanity_forward.py
│  │    - Quick smoke test for a single batch:
│  │      loads dataset, runs one forward pass,
│  │      prints loss to confirm wiring.
│  │
│  ├─ decode_actions.py
│  │    - Converts a generated text string
│  │      ("509 5X0 651 ...") back into arrays of integers.
│  │    - Maps `[0..1000]` integers back to continuous control
│  │      using global min/max stats.
│  │
│  ├─ ensemble_buffer.py
│  │    - Implements temporal ensemble smoothing for rollout:
│  │      to execute the current step, average aligned
│  │      predictions from the last H model calls.
│  │
│  └─ inference_policy.py
│       - High-level inference loop for rollout:
│         capture current RGB(s) + instruction,
│         generate next H×D integers from the model,
│         decode, smooth via ensemble, and output an action.

└─ Data/
   └─ scripts/
      ├─ compute_global_action_stats.py
      │    - Scan ALL demonstrations (all LIBERO tasks).
      │    - Compute per-dimension action min/max.
      │    - Saves e.g. `action_stats.npz`.
      │
      ├─ build_training_manifest.py
      ├─ build_training_manifest_resume.py
      │    - Construct `manifest.jsonl`.
      │    - Each line = one supervised training sample containing:
      │         {
      │           "image_path": <RGB frame for timestep t>,
      │           "instruction": <task language>,
      │           "target_action_text": <future H×D integers>,
      │           "H": <prediction horizon>,
      │           "D": <action dimension>
      │         }
      │    - The `_resume` version can safely restart
      │      if preprocessing was interrupted.
      │
      └─ peek_manifest.py
           - Debug utility: open `manifest.jsonl`,
             print a few samples, verify paths and label format.
```

**Important:**
We intentionally do **not** commit the huge processed dataset directory (e.g. `Data/processed_vla0/train/manifest.jsonl`, training images, etc.) or any checkpoints.
Those are machine-specific and very large.

---

## 4. Environment setup

We provide a `conda` environment that targets:

* Python 3.10
* PyTorch 2.1+ with CUDA 12.x support (for A100 / H100)
* `transformers` (with the Qwen2.5-VL model family)
* `accelerate`
* basic vision / video deps (`opencv-python`, `imageio`, etc.)

On a fresh machine (Linux + NVIDIA GPUs):

```bash
# clone this repo
git clone git@github.com:Allenfree/VLA-0.git
cd VLA-0

# create the environment
conda env create -f environment.yml
conda activate VLA-0
```

After activation, you should be able to import:

* `torch`
* `accelerate`
* `transformers`
* and run `python Train/sanity_forward.py` once you have a manifest.

---

## 5. Data preprocessing pipeline

The preprocessing scripts in `Data/scripts/` do three jobs:

### 5.1 Compute global action stats

We first need global `action_mins` and `action_maxs` across **all** demos.

```bash
python Data/scripts/compute_global_action_stats.py
```

This script:

* iterates through all demonstrations from LIBERO (all suites),
* reads the raw continuous control vectors `a_t` (shape `[D]`),
* records global min/max for each dimension `d`.

It saves something like:

```text
Data/processed_vla0/action_stats.npz
    action_mins[d], action_maxs[d]
```

We use these ranges for discretization.

### 5.2 Discretize actions and build targets

For each timestep `t`:

1. Take the next `H` future actions (`a_t`, `a_{t+1}`, ..., `a_{t+H-1}`).
2. For each action dimension:

   * Normalize to `[0,1]` using the global min/max.
   * Scale to `[0,1000]`, round to int, and clip.
3. Flatten all `H × D` integers into one long space-separated string.
4. Randomly mask digits in that string with `'X'` to form **masked action augmentation**.

This becomes `target_action_text`.

### 5.3 Write `manifest.jsonl`

`build_training_manifest.py` (or `_resume.py`) writes one JSON line per supervised sample:

```json
{
  "image_path": "/abs/path/to/frame_000123.png",
  "instruction": "put the black bowl on the wooden cabinet onto the plate",
  "target_action_text": "509 5X0 651 472 50X 50X 0 556 ...",
  "H": 8,
  "D": 7
}
```

Notes:

* `image_path` should be an RGB frame representing the *current* timestep `t`.
  In the paper, they typically use multi-view (e.g. third-person camera + wrist camera).
  You can either tile them into one image or pass multiple images to Qwen2.5-VL.
* We only use the *current* visual observation, not a stack of history frames.

You can sanity check with:

```bash
python Data/scripts/peek_manifest.py
```

If that script prints out samples without crashing, and all `image_path` files exist, then your dataset is ready.

---

## 6. Training

Training is done with Hugging Face `accelerate` in bf16 mixed precision.

### 6.1 Sanity check

Before long training, make sure everything forwards correctly:

```bash
cd Train
python sanity_forward.py
```

Expected:

* It loads the manifest.
* Builds a batch with `VLACollator` (including the proper chat template for Qwen2.5-VL).
* Runs a single forward pass through `Qwen2_5_VLForConditionalGeneration`.
* Prints a finite (non-NaN) loss.

### 6.2 Distributed finetuning

Example for 2 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --multi_gpu \
  Train/train_vla0.py \
  --model_name_or_path /path/to/Qwen2.5-VL-3B-Instruct \
  --train_manifest /path/to/Data/processed_vla0/train/manifest.jsonl \
  --output_dir ckpts/vla0_stage1 \
  --learning_rate 5e-6 \
  --num_train_epochs 64 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 24 \
  --save_steps 500 \
  --save_total_limit 5 \
  --logging_steps 50 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --report_to none \
  --gradient_checkpointing \
  --dataloader_num_workers 4
```

Key details:

* We train in **bf16** (bfloat16), not fp16.
  fp16 caused gradient-scaler issues like `ValueError: Attempting to unscale FP16 gradients.`
  bf16 on A100/H100 works well.
* `gradient_accumulation_steps` simulates a large global batch size to match the paper (they report ~192 global batch).
* `gradient_checkpointing` reduces memory usage at the cost of a small speed hit.
* The script prints:

  * current global step,
  * loss,
  * learning rate,
  * peak GPU memory per device.
* The script periodically calls `accelerate.save_state(...)` into `--output_dir`.
  Old checkpoints are pruned so the directory doesn’t explode.

Scaling to more GPUs:

* You can increase `CUDA_VISIBLE_DEVICES` and `--num_processes` to match how many GPUs you want to use.
* You should proportionally adjust `gradient_accumulation_steps` if you want to keep the same effective global batch size.

---

## 7. Inference and rollout

After (or during) training, you can treat the finetuned model as a policy.

High-level inference loop:

1. Capture the current observation (RGB from the robot / sim at time `t`) and the language instruction.
2. Run `inference_policy.py`:

   * This builds the same chat-style prompt as training (system prompt + vision + instruction),
   * Uses the finetuned model to **generate** a sequence of integers (length `H × D`).
   * We typically use greedy decoding / low temperature for stability.
3. Parse the generated text using `decode_actions.py`:

   * Extract only integers / `X`-masked integers.
   * Convert them back into arrays of shape `[H, D]`.
   * Map those `[0..1000]` integers back to continuous control using the saved global min/max from preprocessing.
4. Stabilize with `ensemble_buffer.py`:

   * Maintain a ring buffer of the last `H` predictions.
   * The action you actually execute now is the mean of the aligned future steps across that buffer
     (step1 from `t`, step2 from `t-1`, step3 from `t-2`, ...).
   * This is the same "sliding window ensemble" trick highlighted by the paper and is important for success rate.

Finally, send the smoothed continuous action vector to your simulator / robot controller.

---

## 8. Roadmap / to-do

* [ ] Re-run full preprocessing on a large machine (8×H100) to rebuild `manifest.jsonl` for all LIBERO suites.
* [ ] Train for the full 64 epochs with a global batch size close to 192.
* [ ] Reproduce the reported LIBERO success rates:

  * ~96–98% success on Spatial / Object / Goal suites,
  * ~87% on Long-Horizon tasks,
  * ~94% overall average.
* [ ] Export a clean evaluation script to automatically run LIBERO rollouts and compute success.

---

## 9. License / citation

This project is released under the MIT license (see `LICENSE`).

If you build on this work, please also credit:

* The VLA-0 paper and authors.
* Qwen2.5-VL-3B-Instruct and its authors.
* The LIBERO benchmark authors.

This repo is not an official release of any of the above projects. It is an independent reproduction / engineering effort.
