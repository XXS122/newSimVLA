# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言要求

**始终使用中文回复用户。**

## Project Overview

SimVLA is a Vision-Language-Action (VLA) model for robot manipulation. It uses **SmolVLM-500M-Instruct** as the vision-language backbone and a custom **Flow Matching** action transformer head trained on LIBERO robot datasets.

Paper: https://arxiv.org/abs/2602.18224 | Models/Data: HuggingFace `YuankaiLuo/SimVLA-LIBERO`

## Environment Setup

```bash
conda create -n simvla python=3.10 -y
conda activate simvla
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.57.0
pip install peft accelerate fastapi tensorboard uvicorn json_numpy safetensors scipy einops timm mmengine pyarrow h5py mediapy num2words av wandb websockets msgpack_numpy
pip install flash-attn==2.5.6 --no-build-isolation
pip install tensorflow tensorflow-datasets
```

**Critical**: `transformers>=4.57.0` is required. SmolVLM uses the `Idefics3` architecture internally.

## Training Commands

**Prepare dataset metadata (one-time):**
```bash
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json
```

**Compute normalization statistics (one-time):**
```bash
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json
```

**Train (small model, 768-hidden/12-layer, GPUs 0-3):**
```bash
bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
# Defaults: batch=64, coef=0.1, output=./runs/simvla_libero_small
```

**Train (large model, 1024-hidden/24-layer, GPUs 4-7):**
```bash
bash train_smolvlm_large.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
# Defaults: batch=64, coef=0.2, output=./runs/simvla_libero_large
```

Both scripts use `accelerate launch --num_processes=4 --mixed_precision bf16`.

**Resume training from checkpoint:**
```bash
bash train_smolvlm_small.sh 64 0.1 ./runs/my_run ./runs/my_run/ckpt-50000
```

**Direct python training (custom config):**
```bash
accelerate launch --num_processes=4 --mixed_precision bf16 train_smolvlm.py \
    --output_dir ./runs/test \
    --train_metas_path ./datasets/metas/libero_train.json \
    --norm_stats_path ./norm_stats/libero_norm.json \
    --action_mode libero_joint \
    --batch_size 32 --learning_rate 1e-4 --num_actions 10 \
    --hidden_size 768 --depth 12 --num_heads 12 --image_size 384
```

## Evaluation Commands

Evaluation uses a client-server architecture with two separate conda environments.

**Start policy server** (in `simvla` env):
```bash
cd evaluation/libero
CUDA_VISIBLE_DEVICES=1 python serve_smolvlm_libero.py \
    --checkpoint ../../runs/simvla_libero_large/ckpt-150000 \
    --norm_stats ../../norm_stats/libero_norm.json \
    --port 8102
# Or load from HuggingFace: --checkpoint YuankaiLuo/SimVLA-LIBERO
```

**Run evaluation** (in `libero` env — separate conda env with LIBERO simulator):
```bash
cd evaluation/libero
bash run_eval_all.sh 8102 10 "eval_run_name" "0 1 2 3"  # num_trials=10
bash run_eval_all.sh 8102 50 "eval_run_name" "0 1 2 3"  # num_trials=50
```

The `libero` conda env requires a separate setup: `conda create -n libero python=3.8.13` with the LIBERO simulator package installed.

## Architecture

### Forward Pass Flow

```
Images [B, V, C, H, W]   Language text
        ↓                      ↓
SmolVLM vision encoder    SmolVLM tokenizer
        ↓                      ↓
  image features ──── concat ──── text embeds
                           ↓
               SmolVLM text_model (Idefics3)
                           ↓
                  vlm_features [B, T, 576]
                           ↓
                   SmolVLMActionTransformer
                  (Flow Matching, 768/1024-dim)
                           ↓
               predicted velocity v_t → actions
```

### Key Modules

- **`models/modeling_smolvlm_vla.py`**: Main `SmolVLMVLA` model (HuggingFace `PreTrainedModel`). Contains:
  - `forward_vlm_efficient()`: Full VLM forward for training (vision + language fused)
  - `forward()`: Flow Matching training loss (Beta(1.5,1) time sampling, MSE velocity loss)
  - `generate_actions()`: Euler integration inference (t=1→0)
  - `run()`: FastAPI/WebSocket server for deployment

- **`models/transformer_smolvlm.py`**: `SmolVLMActionTransformer` — two modes:
  - **Concat mode** (`use_adaln=False`): VLM features concat'd to action token sequence
  - **AdaLN/DiT mode** (`use_adaln=True`): VLM/time/proprio injected via Adaptive Layer Norm

- **`models/action_hub.py`**: Action space registry. Currently only `libero_joint` is registered (`dim_action=7`, `dim_proprio=8`). Add new action spaces with `@register_action("name")` decorator.

- **`models/configuration_smolvlm_vla.py`**: `SmolVLMVLAConfig` (HuggingFace `PretrainedConfig`). Key fields: `smolvlm_model_path`, `hidden_size`, `depth`, `num_heads`, `action_mode`, `num_actions`, `use_adaln`, `image_size`.

- **`models/processing_smolvlm_vla.py`**: `SmolVLMVLAProcessor` — handles image preprocessing (ImageNet normalization, bicubic resize) and language tokenization. `encode_image()` is the fast GPU-based path; `encode_image_legacy()` uses HuggingFace processor.

- **`datasets/dataset_smolvlm.py`**: `SmolVLMDataReader` — infinite `IterableDataset` with weighted multi-dataset sampling. Output sample: `{language_instruction, image_input [V,C,H,W], image_mask [V], proprio, action}`.

- **`datasets/domain_handler/libero_hdf5.py`**: `LiberoHDF5Handler` — reads LIBERO HDF5 files. Images are **rotated 180°** before processing. Euler orientation is converted to axis-angle for proprio.

- **`datasets/domain_config.py`**: `DATA_WEIGHTS` dict controls sampling weights when mixing multiple datasets.

### Training Details

- **Optimizer**: AdamW with 3 param groups — `vlm` (frozen for `freeze_steps=1000`, then `lr * learning_coef`), `transformer_core`, `action_heads`
- **LR schedule**: VLM frozen for first 1000 steps, then linear warmup + optional cosine decay
- **Image size**: 384×384 (default); 512×512 also supported
- **Views**: 2 active views per LIBERO sample (agentview + wrist), padded to 3
- **Checkpoints**: Saved as HuggingFace `safetensors` format in `{output_dir}/ckpt-{step}/`; resume with `--models ./ckpt-N --resume`

### LIBERO Data Format

HDF5 structure: `data/demo_X/{actions, obs/agentview_rgb, obs/eye_in_hand_rgb, obs/ee_pos, obs/ee_ori, obs/gripper_states}`
- Actions: 7-dim delta `[xyz(3), euler(3), gripper(1)]`, range `[-1, 1]`
- Proprio: 8-dim `[ee_pos(3), axis_angle(3), gripper_states(2)]`
- Images: 128×128 native, upscaled to 384×384 during training

### Norm Stats Format

`norm_stats/libero_norm.json` has keys `state` and `actions`, each with `mean`, `std`, `q01`, `q99` arrays. Loaded by `LiberoJointActionSpace` for Z-score or quantile normalization.

### Inference Server Protocol

`serve_smolvlm_libero.py` exposes a **WebSocket** server using `msgpack_numpy` serialization. Receives: `{observation/image, observation/wrist_image, observation/state, prompt}`. Returns: `{actions: [[7-dim] × horizon]}`.
