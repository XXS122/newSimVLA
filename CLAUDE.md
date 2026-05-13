# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言要求

**始终使用中文回复用户。**

**不要修改或新增任何与 LIBERO 相关的代码、文档或配置。** 本项目当前只关注 VLABench，LIBERO 相关文件（`evaluation/libero/`、`datasets/domain_handler/libero_hdf5.py`、`train_smolvlm_small.sh`、`train_smolvlm_large.sh` 等）保持原样即可，无需维护。

## 项目概述

SimVLA 是用于机器人操作的视觉-语言-动作（VLA）模型。

- 骨干网络：**SmolVLM-500M-Instruct**（视觉语言模型，Idefics3 架构）
- 动作头：**Flow Matching Action Transformer**（预测连续动作轨迹）
- 数据集：**VLABench**（RLDS/TFRecord 格式，4 视角）
- 创新点：**双流多视角融合**（Dual-Stream Multi-View Fusion）

论文：https://arxiv.org/abs/2602.18224

## 算法原理

### Flow Matching 动作生成

模型不直接回归动作，而是学习从噪声到动作的速度场：
1. 训练时：对真实动作加噪 `x_t = (1-t)*noise + t*action`，学习预测速度 `v = action - noise`
2. 推理时：从纯噪声出发，用 Euler 积分沿速度场走到 t=0，得到动作
3. 时间采样：Beta(1.5, 1) 分布，偏向 t 接近 1（接近真实动作的区域）
4. 损失函数：MSE(predicted_velocity, true_velocity)

### 双流多视角融合（Dual-Stream Fusion）

将 4 个视角的 VLM 特征分为两路：
- **静态流**（front / image_0 / image_1）：提供场景全局语义
- **动态流**（wrist 手腕相机）：提供末端执行器运动细节

融合方式（`--dual_stream_fusion`）：
- `cross_attn`（默认）：静态流为 Query，动态流为 Key/Value，跨注意力融合
- `concat_linear`：拼接后线性投影
- `add`：直接相加

### 前向传播流程

```
图像 [B, 4, C, 384, 384]     语言指令
        ↓                         ↓
SmolVLM 视觉编码器           SmolVLM 分词器
        ↓                         ↓
  图像 patch 特征 ─── concat ─── 文本嵌入
                              ↓
               SmolVLM text_model (Idefics3)
                              ↓
                  vlm_features [B, T, 576]
                              ↓
              [可选] DualStreamFusion（双流融合）
                              ↓
               SmolVLMActionTransformer (768-dim, 12层)
              (Flow Matching: 输入噪声动作+时间步+本体感知)
                              ↓
                   预测速度 v_t → 积分得到动作 [B, 10, 7]
```

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

必须使用 `transformers>=4.57.0`。

## 路径配置

所有脚本通过 `paths.env` 统一管理路径（已 gitignore，首次使用从 `paths.env.template` 复制）：

| 变量 | 用途 |
|---|---|
| `SIMVLA_SMOLVLM_MODEL` | SmolVLM 模型本地路径 |
| `SIMVLA_VLABENCH_DATA` | VLABench RLDS 数据目录 |
| `SIMVLA_VLABENCH_CODE` | VLABench 代码仓库路径（评估时使用） |
| `SIMVLA_OUTPUT_DIR` | checkpoint 保存根目录（子目录自动以时间戳命名） |
| `SIMVLA_EVAL_RESULTS` | 评估结果保存目录（子目录自动以时间戳命名） |
| `SIMVLA_CUDA_DEVICES` | CUDA_VISIBLE_DEVICES |
| `SIMVLA_NUM_GPUS` | accelerate 使用的 GPU 数量 |

## 训练

### 数据准备（一次性）

```bash
# 生成训练元数据（列出所有 shard 文件路径）
python create_vlabench_meta.py \
    --data_dir /path/to/vlabench/data/1.0.0 \
    --output ./datasets/metas/vlabench_train.json

# 计算归一化统计（动作和本体感知的 mean/std/q01/q99）
python compute_vlabench_norm_stats.py \
    --data_dir /path/to/vlabench/data/1.0.0 \
    --output ./norm_stats/vlabench_norm.json
```

### 训练脚本一览

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `train_vlabench_small.sh` | 基线全量训练（无双流） | 200K 步，batch=32，全部 shard |
| `train_vlabench_dualstream.sh` | 双流融合全量训练 | 200K 步，batch=32，`--use_dual_stream` |
| `train_vlabench_debug.sh` | 快速调试训练 | 10K 步，batch=16，50 shard |

### 基线训练（无双流）

```bash
bash train_vlabench_small.sh [batch_size] [learning_coef] [resume_ckpt]
# 默认：batch=32, coef=0.1
# 输出：${SIMVLA_OUTPUT_DIR}/vlabench_small_<timestamp>/
```

### 双流融合训练

```bash
bash train_vlabench_dualstream.sh [batch_size] [learning_coef] [resume_ckpt] [fusion_type]
# 默认：batch=32, coef=0.1, fusion=cross_attn
# fusion_type 可选：add | concat_linear | cross_attn
# 输出：${SIMVLA_OUTPUT_DIR}/vlabench_dualstream_<fusion>_<timestamp>/
```

### 快速调试训练（1-2 小时出 checkpoint）

```bash
bash train_vlabench_debug.sh [batch_size] [learning_coef] [resume_ckpt]
# 默认：batch=16, coef=0.1, 50 shard, 10K 步
# 输出：${SIMVLA_OUTPUT_DIR}/vlabench_debug_<timestamp>/
# 脚本内 USE_DUAL_STREAM=true/false 控制是否开启双流
```

### 从 checkpoint 恢复训练

```bash
bash train_vlabench_small.sh 32 0.1 /path/to/ckpt-50000
```

### 训练超参数

| 参数 | 全量训练 | 调试训练 |
|---|---|---|
| 总步数 | 200,000 | 10,000 |
| Batch size | 32 | 16 |
| 学习率 | 1e-4 | 1e-4 |
| VLM 学习率系数 | 0.1 | 0.1 |
| VLM 冻结步数 | 1000 | 500 |
| 保存间隔 | 10,000 | 2,000 |
| 图像尺寸 | 384×384 | 384×384 |
| 视角数 | 4 | 4 |
| 动作维度 | 7 | 7 |
| 动作 horizon | 10 | 10 |
| 混合精度 | fp16 | fp16 |
| Action Transformer | 768-dim, 12层, 12头 | 同左 |

## 评估

评估采用**客户端-服务端架构**，需要两个终端、两个 conda 环境：
- 终端1（`simvla` 环境）：加载模型，启动 WebSocket 策略服务器
- 终端2（`vlabench` 环境）：运行 MuJoCo 模拟器，通过 WebSocket 请求动作

### 步骤1：启动策略服务器（终端1，simvla 环境）

```bash
conda activate simvla
cd /datasets/code/newSimVLA
CUDA_VISIBLE_DEVICES=0 python evaluation/vlabench/serve_smolvlm_vlabench.py \
    --checkpoint     /datasets/simvla_output/checkpoint/dualstream/vlabench_dualstream_cross_attn_20260509_07/ckpt-50000
 \
    --norm_stats ./norm_stats/vlabench_norm.json \
    --port 8200
```

等待打印 `Uvicorn running on ...` 后再启动客户端。

### 步骤2：运行评估客户端（终端2，vlabench 环境）

```bash
conda activate vlabench
cd /datasets/code/newSimVLA/evaluation/vlabench
bash run_eval_vlabench.sh <port> <n_episode> <eval_track> <save_name> [checkpoint_path]
```

参数说明：
- `port`：服务器端口（与步骤1一致）
- `n_episode`：每个任务跑多少个 episode（调试用 10，正式用 50）
- `eval_track`：评估 track 名称
  - `track_debug_simple`：调试用，含 select_fruit / select_drink 等少量任务
  - `track_1_in_distribution`：完整分布内评估
- `save_name`：结果保存子目录名
- `checkpoint_path`（可选）：仅写入 eval_info.txt 记录，客户端不加载模型

示例：
```bash
# 调试评估（快速验证）
bash run_eval_vlabench.sh 8200 10 track_debug_simple debug_eval \
    /datasets/simvla_output/checkpoint/dualstream/vlabench_dualstream_cross_attn_20260509_07/ckpt-50000

# 完整评估
bash run_eval_vlabench.sh 8200 50 track_1_in_distribution full_eval \
    /datasets/simvla_output/checkpoint/dualstream/vlabench_dualstream_cross_attn_20260509_07/ckpt-50000

```

结果保存到 `$SIMVLA_EVAL_RESULTS/<save_name>_<timestamp>/`，包含 `eval_info.txt` 和评估指标。

### 离线 Action MSE 评估（无需模拟器）

```bash
python eval_action_mse.py \
    --checkpoint /path/to/ckpt-XXXXX \
    --norm_stats ./norm_stats/vlabench_norm.json \
    --data_dir /path/to/vlabench/data/1.0.0 \
    --num_shards 10 --num_samples 200 --num_views 4
```

## 核心代码结构

| 文件 | 功能 |
|---|---|
| `models/modeling_smolvlm_vla.py` | 主模型 SmolVLMVLA（VLM + Action Transformer + Flow Matching） |
| `models/transformer_smolvlm.py` | Action Transformer（Concat 模式 / AdaLN 模式） |
| `models/dual_stream.py` | 双流多视角融合模块（CrossAttention / Add / ConcatLinear） |
| `models/action_hub.py` | 动作空间注册表（vlabench_joint: 7维动作 + 7维本体感知） |
| `models/configuration_smolvlm_vla.py` | 模型配置（HuggingFace PretrainedConfig） |
| `models/processing_smolvlm_vla.py` | 图像预处理 + 语言分词 |
| `datasets/dataset_smolvlm.py` | 数据加载器（无限 IterableDataset） |
| `datasets/domain_handler/vlabench_rlds.py` | VLABench RLDS 数据读取 |
| `train_smolvlm.py` | 统一训练入口（所有 shell 脚本调用此文件） |
| `evaluation/vlabench/serve_smolvlm_vlabench.py` | 评估策略服务器 |
| `evaluation/vlabench/simvla_policy.py` | VLABench 客户端策略（WebSocket 通信） |
| `evaluation/vlabench/run_eval_vlabench.sh` | 评估启动脚本 |
| `eval_action_mse.py` | 离线 Action MSE 评估 |

## VLABench 数据格式

RLDS/TFRecord 格式，每个 shard 文件对应一条轨迹：
- 4 个视角：front、wrist、image_0、image_1
- 动作：7 维 `[xyz(3), euler(3), gripper(1)]`，绝对位置
- 本体感知：7 维 `[xyz(3), euler(3), gripper(1)]`
- 图像无需旋转（与 LIBERO 不同）

## Checkpoint 格式

HuggingFace safetensors 格式，保存在 `{output_dir}/ckpt-{step}/`：
- `config.json`：模型配置（含 use_dual_stream、num_views 等）
- `model.safetensors`：模型权重
- `state.json`：训练状态（global_step）

每次训练自动在输出目录生成 `run_info.txt`，记录算法、超参数、数据路径等信息。

