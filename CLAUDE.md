# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言要求

**始终使用中文回复用户。**

## 项目概述

本项目是基于 SimVLA（arXiv:2602.18224）骨架的视觉-语言-动作（VLA）模型，针对 **VLABench**（RLDS/TFRecord 格式，4 视角）数据集进行系统性改进，引入以下创新：

1. **双流多视角融合**（DualStreamFusion）：静态视角（front/image_0/image_1）作为 Query，腕部视角（wrist）作为 Key/Value 做跨注意力融合
2. **运动引导跨视角注意力**（MotionCNN）：帧差分图驱动的轻量 CNN 生成运动激活图，注入跨注意力作为偏置，使静态视角自动聚焦于运动活跃区域
3. **ActionVAE 隐式扩散策略**：将动作块编码至低维隐空间（d_z=32）做 Flow Matching，再解码恢复完整动作序列
4. **AdaLN + Concat 混合条件注入**：低维全局信号（t + proprio）走 AdaLN，VLM token 保留在 Concat 序列

骨干网络：**SmolVLM-500M-Instruct**（Idefics3 架构）；动作头：Flow Matching Transformer 或 LatentFlowNet（隐空间）。

详细算法说明见 `ALGORITHM.md`。

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

**准备 VLABench 元数据和归一化统计（一次性）：**
```bash
python create_vlabench_meta.py --data_dir ./datasets/vlabench/data/1.0.0 --output ./datasets/metas/vlabench_train.json
python compute_vlabench_norm_stats.py --data_dir ./datasets/vlabench/data/1.0.0 --output ./norm_stats/vlabench_norm.json
```

**训练 VLABench 小模型（fp16，4 视角）：**
```bash
bash train_vlabench_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
# 默认：batch=32, coef=0.1, output=./runs/simvla_vlabench_small
# 使用 fp16（非 bf16），num_workers=0，action_mode=vlabench_joint
```

**从 checkpoint 恢复训练：**
```bash
bash train_vlabench_small.sh 32 0.1 ./runs/my_run ./runs/my_run/ckpt-50000
```

**直接调用 Python 训练（启用全部创新）：**
```bash
accelerate launch --num_processes=4 --mixed_precision fp16 train_smolvlm.py \
    --output_dir ./runs/vlabench_full \
    --train_metas_path ./datasets/metas/vlabench_train.json \
    --norm_stats_path ./norm_stats/vlabench_norm.json \
    --action_mode vlabench_joint \
    --batch_size 32 --learning_rate 1e-4 --num_actions 10 \
    --hidden_size 768 --depth 12 --num_heads 12 --image_size 384 \
    --num_views 4 \
    --use_dual_stream --dual_stream_fusion cross_attn \
    --use_motion_guided_attn \
    --use_action_vae --latent_dim 32 \
    --use_adaln_hybrid
```

**可选 CLI 开关**：
- `--use_dual_stream --dual_stream_fusion {add|concat_linear|cross_attn}`：双流融合
- `--use_motion_guided_attn`：运动引导跨视角注意力（需同时启用 `--use_dual_stream`）
- `--use_action_vae --latent_dim 32`：ActionVAE 隐空间流匹配
- `--use_adaln` / `--use_adaln_hybrid`：AdaLN 或混合条件注入
- `--use_view_dropout --view_dropout_prob 0.1 --use_missing_token`：视角 dropout + 可学习缺失 token
- `--proprio_history_len 4`：本体感知历史窗口（GRU 编码）
- `--time_sampling {logit_normal|beta}`：流匹配时间采样策略

## 评估命令

评估采用客户端-服务端架构，需要两个独立的 conda 环境。

**启动策略服务器**（在 `simvla` 环境中）：
```bash
cd evaluation/vlabench
CUDA_VISIBLE_DEVICES=0 python serve_smolvlm_vlabench.py \
    --checkpoint ../../runs/simvla_vlabench_small/ckpt-XXXXX \
    --norm_stats ../../norm_stats/vlabench_norm.json --port 8103
```

**运行评估**（在 VLABench 专属环境中）：
```bash
cd evaluation/vlabench
bash run_eval_vlabench.sh 8103 "eval_run_name"
```

`vlabench` 环境需单独创建，安装 VLABench 模拟器，路径由 `SIMVLA_VLABENCH_CODE` 指向。

## 架构

### 前向传播流程

```
图像 [B, V=4, C, H, W]      语言文本     前一帧 wrist [B, C, H, W]   proprio [B, 7]
        ↓                       ↓                ↓                       ↓
SmolVLM 视觉编码器          分词器        帧差分 Δ_t                   Linear/GRU
        ↓                       ↓                ↓
  视觉 token ──── concat ─── 文本 token   MotionCNN
                       ↓                         ↓
            Idefics3 text_model              运动激活图 M [B, P]
                       ↓                         ↓
         VLM 特征 F_vlm [B, L, 576]              │
                       ↓                         │
        DualStreamFusion（按视角切分）           │
        Q=F_static, K/V=F_wrist, bias=α·M ←─────┘
                       ↓
            F_fused [B, n_s·P, 576]
                       ↓
        ┌──────────────┴──────────────┐
        │                              │
   标准 Flow Matching            ActionVAE 模式
   v_t = Transformer(...)        z [B,32] → LatentFlowNet → v_z
        │                              │
        └──────────────┬──────────────┘
                       ↓
             Euler 积分（10 步）
                       ↓
            动作序列 [B, T=10, 7]
```

### 核心模块

- **`models/modeling_smolvlm_vla.py`**：主模型 `SmolVLMVLA`（HuggingFace `PreTrainedModel`）
  - `forward_vlm_efficient()`：训练用 VLM 前向（视觉+语言融合）
  - `forward()`：流匹配训练损失（含 VAE 重建/KL 损失，gripper 加权）
  - `generate_actions()`：Euler 积分推理（t=1→0），支持 ActionVAE 隐空间路径
  - `run()`：FastAPI/WebSocket 部署服务

- **`models/transformer_smolvlm.py`**：`SmolVLMActionTransformer`，两种模式：
  - **Concat 模式**（`use_adaln=False`）：VLM 特征拼接到动作 token 序列
  - **AdaLN/DiT 模式**（`use_adaln=True` 或 `use_adaln_hybrid=True`）：VLM/时间/本体感知通过自适应层归一化注入

- **`models/action_hub.py`**：动作空间注册表
  - `vlabench_joint`：`dim_action=7`，`dim_proprio=7`（xyz + euler + gripper）
  - 用 `@register_action("name")` 装饰器添加新动作空间

- **`models/dual_stream.py`**：
  - `DualStreamFusion`：双流多视角融合，三种策略 `add` / `concat_linear` / `cross_attn`（默认）
  - `CrossAttentionFusion`：跨注意力实现，支持运动激活图偏置注入（可学习 `motion_bias_scale`，零初始化）
  - `MotionCNN`：帧差分 → 3 层卷积（通道 16/32/32，结构化剪枝）→ AdaptiveAvgPool → Sigmoid → 运动激活图 `[B, P]`

- **`models/action_vae.py`**：
  - `ActionVAE`：动作块 `[B,T,7]` ↔ 隐变量 `z [B,32]`，编码器为 Transformer + CLS，解码器跨注意力到 VLM 特征
  - `LatentFlowNet`：隐空间速度场 MLP（4 层残差），条件为 VLM 全局池化 + proprio + 时间编码

- **`models/configuration_smolvlm_vla.py`**：`SmolVLMVLAConfig`。关键字段：
  - `smolvlm_model_path`、`hidden_size`、`depth`、`num_heads`、`action_mode`、`num_actions`、`image_size`、`num_views`
  - `use_adaln` / `use_adaln_hybrid`
  - `use_dual_stream` / `dual_stream_fusion`
  - `use_motion_guided_attn`
  - `use_action_vae` / `latent_dim` / `vae_beta` / `vae_recon_weight`
  - `use_view_dropout` / `view_dropout_prob` / `use_missing_token`
  - `proprio_history_len`
  - `time_sampling` / `logit_normal_mean` / `logit_normal_std`

- **`models/processing_smolvlm_vla.py`**：`SmolVLMVLAProcessor`，图像预处理（ImageNet 归一化、双三次插值至 384×384）和语言分词。`encode_image()` 为快速 GPU 路径

- **`datasets/dataset_smolvlm.py`**：`SmolVLMDataReader`——带加权多数据集采样的无限 `IterableDataset`。输出样本：`{language_instruction, image_input [V,C,H,W], image_mask [V], proprio, abs_trajectory, wrist_prev_pixels}`

- **`datasets/domain_handler/vlabench_rlds.py`**：`VLABenchRLDSHandler`——读取 VLABench RLDS/TFRecord 格式
  - 4 个视角：front、wrist、image_0、image_1
  - 无图像旋转，无欧拉角→轴角转换
  - 同时读取 `wrist_bytes[idx]` 和 `wrist_bytes[idx-1]`，提供给 MotionCNN 计算帧差分（idx=0 时用当前帧重复）

- **`datasets/domain_handler/registry.py`**：数据集 handler 注册表。添加新数据集需：①继承 `DomainHandler` 并实现 `iter_episode()`；②在 `registry.py` 的 `_REGISTRY` 字典中注册；③在 `domain_config.py` 的 `DATA_WEIGHTS` 中添加权重

- **`datasets/domain_config.py`**：`DATA_WEIGHTS` 字典控制多数据集混合时的采样权重

### 训练细节

- **优化器**：AdamW，3 个参数组——`vlm`（前 `freeze_steps=1000` 步冻结，之后 `lr * learning_coef`）、`transformer_core`、`action_heads`（含 ActionVAE/LatentFlowNet 的输出层）
- **学习率调度**：前 1000 步冻结 VLM，之后线性预热 + 可选余弦衰减
- **图像尺寸**：默认 384×384，也支持 512×512
- **视角数**：VLABench 4 个视角（front、wrist、image_0、image_1）
- **混合精度**：VLABench 使用 fp16（兼容性更好），`num_workers=0`
- **Checkpoint**：以 HuggingFace `safetensors` 格式保存在 `{output_dir}/ckpt-{step}/`；用 `--models ./ckpt-N --resume` 恢复训练

### VLABench 数据格式

TFRecord 结构（每个 shard 一条轨迹）：
- `steps/action` [T·7]：7 维绝对位置 `[xyz(3), euler(3), gripper(1)]`
- `steps/observation/ee_state` [T·7]：本体感知（与动作维度同义）
- `steps/observation/{front, wrist, image_0, image_1}`：4 视角图像字节序列
- `steps/language_instruction`：语言指令（取首帧）

训练时上采样图像至 384×384 并执行 ImageNet 归一化。

### 归一化统计格式

`norm_stats/vlabench_norm.json` 包含 `state` 和 `actions` 两个键，每个键下有 `mean`、`std`、`q01`、`q99` 数组。由 `VLABenchJointActionSpace` 加载，用于 Z-score 或分位数归一化。

### 推理服务协议

`evaluation/vlabench/serve_smolvlm_vlabench.py` 通过 `msgpack_numpy` 序列化暴露 **WebSocket** 服务器。

**请求**：`{observation/images: [4×np数组], observation/state: [7], prompt: str}`，或 `{reset: true}` 重置 episode 级 `prev_wrist_tensor` 缓冲。

**响应**：`{actions: [[7维] × horizon]}`。

服务器为每个连接维护 episode 级 `prev_wrist_tensor`，episode 首帧 `prev = current`（差分全零）。
