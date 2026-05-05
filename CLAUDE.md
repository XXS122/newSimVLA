# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言要求

**始终使用中文回复用户。**

## 项目概述

SimVLA 是用于机器人操作的视觉-语言-动作（VLA）模型，以 **SmolVLM-500M-Instruct** 作为视觉语言骨干网络，配合自定义的 **Flow Matching** 动作 Transformer 头。支持两个数据集：**LIBERO**（HDF5 格式）和 **VLABench**（RLDS/TFRecord 格式）。

论文：https://arxiv.org/abs/2602.18224 | 模型/数据：HuggingFace `YuankaiLuo/SimVLA-LIBERO`

## 环境配置

```bash
conda create -n simvla python=3.10 -y
conda activate simvla
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.57.0
pip install peft accelerate fastapi tensorboard uvicorn json_numpy safetensors scipy einops timm mmengine pyarrow h5py mediapy num2words av wandb websockets msgpack_numpy
pip install flash-attn==2.5.6 --no-build-isolation
pip install tensorflow tensorflow-datasets
```

**重要**：必须使用 `transformers>=4.57.0`，SmolVLM 内部使用 `Idefics3` 架构。

## 路径配置

训练前修改 `paths.env` 设置本地路径，脚本在 `SIMVLA_SMOLVLM_MODEL` 未设置时会自动加载：

| 变量 | 用途 |
|---|---|
| `SIMVLA_SMOLVLM_MODEL` | SmolVLM 模型路径（默认：`HuggingFaceTB/SmolVLM-500M-Instruct`） |
| `SIMVLA_VLABENCH_DATA` | VLABench RLDS 数据目录 |
| `SIMVLA_VLABENCH_CODE` | VLABench 代码仓库路径（评估时使用） |
| `SIMVLA_EVAL_RESULTS` | 评估结果保存目录 |
| `SIMVLA_CUDA_DEVICES` | `CUDA_VISIBLE_DEVICES` 的值 |
| `SIMVLA_NUM_GPUS` | `accelerate` 使用的 GPU 数量 |

## 训练命令

**准备 LIBERO 数据集元数据（一次性）：**
```bash
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json
```

**计算 LIBERO 归一化统计（一次性）：**
```bash
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json
```

**训练 LIBERO 小模型（768-hidden/12-layer，GPU 0-3）：**
```bash
bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
# 默认：batch=64, coef=0.1, output=./runs/simvla_libero_small
```

**训练 LIBERO 大模型（1024-hidden/24-layer，GPU 4-7）：**
```bash
bash train_smolvlm_large.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
# 默认：batch=64, coef=0.2, output=./runs/simvla_libero_large
```

两个脚本均使用 `accelerate launch --num_processes=4 --mixed_precision bf16`。

**训练 VLABench 小模型（fp16，4 视角）：**
```bash
bash train_vlabench_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
# 默认：batch=32, coef=0.1, output=./runs/simvla_vlabench_small
# 使用 fp16（非 bf16），num_workers=0，action_mode=vlabench_joint
```

**从 checkpoint 恢复训练：**
```bash
bash train_smolvlm_small.sh 64 0.1 ./runs/my_run ./runs/my_run/ckpt-50000
```

**直接调用 Python 训练（自定义配置）：**
```bash
accelerate launch --num_processes=4 --mixed_precision bf16 train_smolvlm.py \
    --output_dir ./runs/test \
    --train_metas_path ./datasets/metas/libero_train.json \
    --norm_stats_path ./norm_stats/libero_norm.json \
    --action_mode libero_joint \
    --batch_size 32 --learning_rate 1e-4 --num_actions 10 \
    --hidden_size 768 --depth 12 --num_heads 12 --image_size 384
```

**准备 VLABench 元数据和归一化统计（一次性）：**
```bash
python create_vlabench_meta.py --data_dir ./datasets/vlabench/data/1.0.0 --output ./datasets/metas/vlabench_train.json
python compute_vlabench_norm_stats.py --data_dir ./datasets/vlabench/data/1.0.0 --output ./norm_stats/vlabench_norm.json
```

## 评估命令

评估采用客户端-服务端架构，需要两个独立的 conda 环境。

**启动策略服务器**（在 `simvla` 环境中）：
```bash
cd evaluation/libero
CUDA_VISIBLE_DEVICES=1 python serve_smolvlm_libero.py \
    --checkpoint ../../runs/simvla_libero_large/ckpt-150000 \
    --norm_stats ../../norm_stats/libero_norm.json \
    --port 8102
# 也可从 HuggingFace 加载：--checkpoint YuankaiLuo/SimVLA-LIBERO
```

**运行评估**（在 `libero` 环境中——需单独安装 LIBERO 模拟器的 conda 环境）：
```bash
cd evaluation/libero
bash run_eval_all.sh 8102 10 "eval_run_name" "0 1 2 3"  # num_trials=10
bash run_eval_all.sh 8102 50 "eval_run_name" "0 1 2 3"  # num_trials=50
```

`libero` 环境需单独创建：`conda create -n libero python=3.8.13`，并安装 LIBERO 模拟器包。

## 架构

### 前向传播流程

```
图像 [B, V, C, H, W]     语言文本
        ↓                      ↓
SmolVLM 视觉编码器        SmolVLM 分词器
        ↓                      ↓
  图像特征 ──── concat ──── 文本嵌入
                           ↓
               SmolVLM text_model (Idefics3)
                           ↓
                  vlm_features [B, T, 576]
                           ↓
                   SmolVLMActionTransformer
                  (Flow Matching, 768/1024-dim)
                           ↓
               预测速度 v_t → 动作
```

### 核心模块

- **`models/modeling_smolvlm_vla.py`**：主模型 `SmolVLMVLA`（HuggingFace `PreTrainedModel`）：
  - `forward_vlm_efficient()`：训练用完整 VLM 前向（视觉+语言融合）
  - `forward()`：Flow Matching 训练损失（Beta(1.5,1) 时间采样，MSE 速度损失）
  - `generate_actions()`：Euler 积分推理（t=1→0）
  - `run()`：FastAPI/WebSocket 部署服务

- **`models/transformer_smolvlm.py`**：`SmolVLMActionTransformer`，两种模式：
  - **Concat 模式**（`use_adaln=False`）：VLM 特征拼接到动作 token 序列
  - **AdaLN/DiT 模式**（`use_adaln=True`）：VLM/时间/本体感知通过自适应层归一化注入

- **`models/action_hub.py`**：动作空间注册表，已注册两个动作空间：
  - `libero_joint`：`dim_action=7`，`dim_proprio=8`（ee_pos + axis_angle + gripper×2）
  - `vlabench_joint`：`dim_action=7`，`dim_proprio=7`
  - 用 `@register_action("name")` 装饰器添加新动作空间

- **`models/configuration_smolvlm_vla.py`**：`SmolVLMVLAConfig`（HuggingFace `PretrainedConfig`）。关键字段：`smolvlm_model_path`、`hidden_size`、`depth`、`num_heads`、`action_mode`、`num_actions`、`use_adaln`、`image_size`

- **`models/processing_smolvlm_vla.py`**：`SmolVLMVLAProcessor`，处理图像预处理（ImageNet 归一化、双三次插值缩放）和语言分词。`encode_image()` 为快速 GPU 路径；`encode_image_legacy()` 使用 HuggingFace processor

- **`datasets/dataset_smolvlm.py`**：`SmolVLMDataReader`——带加权多数据集采样的无限 `IterableDataset`。输出样本：`{language_instruction, image_input [V,C,H,W], image_mask [V], proprio, action}`

- **`datasets/domain_handler/libero_hdf5.py`**：`LiberoHDF5Handler`——读取 LIBERO HDF5 文件。图像在处理前**旋转 180°**，欧拉角转换为轴角表示作为本体感知

- **`datasets/domain_handler/vlabench_rlds.py`**：`VLABenchRLDSHandler`——读取 VLABench RLDS/TFRecord 格式。4 个视角：front、wrist、image_0、image_1。无图像旋转，无欧拉角→轴角转换

- **`datasets/domain_config.py`**：`DATA_WEIGHTS` 字典控制多数据集混合时的采样权重

### 训练细节

- **优化器**：AdamW，3 个参数组——`vlm`（前 `freeze_steps=1000` 步冻结，之后 `lr * learning_coef`）、`transformer_core`、`action_heads`
- **学习率调度**：前 1000 步冻结 VLM，之后线性预热 + 可选余弦衰减
- **图像尺寸**：默认 384×384，也支持 512×512
- **视角数**：LIBERO 每样本 2 个有效视角（agentview + wrist），填充至 3；VLABench 4 个视角（front、wrist、image_0、image_1）
- **Checkpoint**：以 HuggingFace `safetensors` 格式保存在 `{output_dir}/ckpt-{step}/`；用 `--models ./ckpt-N --resume` 恢复训练

### LIBERO 数据格式

HDF5 结构：`data/demo_X/{actions, obs/agentview_rgb, obs/eye_in_hand_rgb, obs/ee_pos, obs/ee_ori, obs/gripper_states}`
- 动作：7 维增量 `[xyz(3), euler(3), gripper(1)]`，范围 `[-1, 1]`
- 本体感知：8 维 `[ee_pos(3), axis_angle(3), gripper_states(2)]`
- 图像：原始 128×128，训练时上采样至 384×384

### 归一化统计格式

`norm_stats/libero_norm.json` 包含 `state` 和 `actions` 两个键，每个键下有 `mean`、`std`、`q01`、`q99` 数组。由 `LiberoJointActionSpace` 加载，用于 Z-score 或分位数归一化。

### 推理服务协议

`serve_smolvlm_libero.py` 通过 `msgpack_numpy` 序列化暴露 **WebSocket** 服务器。接收：`{observation/image, observation/wrist_image, observation/state, prompt}`。返回：`{actions: [[7维] × horizon]}`。
