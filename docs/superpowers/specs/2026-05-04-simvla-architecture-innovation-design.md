# SimVLA 架构创新设计文档

**日期：** 2026-05-04
**状态：** 草稿

---

## 背景与动机

SimVLA 当前架构（SmolVLM-500M + Flow Matching Action Transformer）在 VLABench 和 LIBERO 上已有基础性能。本文档设计三个协同创新方向，目标是在不改变 SmolVLM 骨干结构的前提下，通过改进 `SmolVLMActionTransformer` 提升任务成功率、训练效率和推理速度，面向 AI 顶会（NeurIPS/ICLR/ICML）投稿。

**核心约束：**
- 不改变 SmolVLM-500M-Instruct 的模型结构
- 所有改动集中在 `models/transformer_smolvlm.py` 和 `models/action_hub.py`
- 保持与现有训练脚本（`train_smolvlm.py`）的兼容性
- 每个创新点可独立开关，支持消融实验

---

## 创新方向一：双流多视角融合（Dual-Stream Multi-View Fusion）

### 问题

SimVLA 当前将所有视角图像（VLABench：front/wrist/image_0/image_1，LIBERO：agentview/wrist）直接 concat 后统一送入 SmolVLM，没有区分视角的语义角色：

- **静态流视角**（agentview/front/image_0/image_1）：提供场景全局语义，物体位置、任务目标
- **动态流视角**（wrist）：提供末端执行器的动态运动信息，手与物体的接触状态

两类视角的信息性质不同，混合处理导致 Action Transformer 无法有效利用各视角的互补信息。

### 方案设计

在 SmolVLM 输出的 VLM 特征层面，将静态流和动态流特征分别提取，通过 **Cross-Attention 融合**后送入 Action Transformer。

```
图像输入 [B, V, C, H, W]
         ↓
SmolVLM（不变）
         ↓
vlm_features [B, T_enc, 576]
         ↓
┌─────────────────────────────────┐
│  双流分离（按视角 token 位置）    │
│  静态流特征 F_s [B, T_s, 576]   │
│  动态流特征 F_d [B, T_d, 576]   │
└─────────────────────────────────┘
         ↓
Cross-Attention 融合：
  Query = F_s（静态流主导）
  Key/Value = F_d（动态流提供运动信息）
         ↓
融合特征 F_fused [B, T_s, 576]
         ↓
Action Transformer（现有结构）
```

**直觉：** 静态流知道"要抓哪个物体"，通过 Cross-Attention 动态地从 wrist 视角中选择"手现在离目标有多近"的信息，比简单 concat 更有针对性。

### 消融实验设计

| 方案 | 描述 | 参数增量 |
|------|------|----------|
| Baseline | 现有 concat 处理 | 0 |
| Add | 静态流 + 动态流特征直接相加 | 0 |
| Concat+Linear | Concat 后线性压缩 | 576×1152 |
| **Cross-Attn（主方案）** | Cross-Attention 融合 | ~1.2M |

### 实现要点

- SmolVLM 对所有视角图像统一编码，输出 `vlm_features [B, T_enc, 576]`
- 按视角 token 的位置索引分离静态/动态特征（需要在 `forward_vlm_efficient()` 中记录各视角 token 的起止位置）
- Cross-Attention 模块：1 层，8 头，hidden_size=576，无位置编码
- 新增配置项：`use_dual_stream: bool`，默认 False（向后兼容）

---

## 创新方向二：动作头稀疏 MoE（Sparse MoE Action Head）

### 问题

SimVLA 的 `SmolVLMActionTransformer` 是密集 Transformer，所有参数对所有任务都激活：

- LIBERO 和 VLABench 的动作空间、场景分布差异很大
- 同一数据集内不同任务阶段（接近/抓取/放置）需要不同的运动模式
- 密集激活导致任务间干扰，且显存占用与参数量成正比

### 方案设计

将 `SmolVLMActionTransformer` 每个 Transformer Block 中的 FFN 层替换为**稀疏 MoE FFN**，每步只激活 Top-2 专家。

```
原始 FFN：
  x → Linear(H, 4H) → GELU → Linear(4H, H) → x

MoE FFN（替换后）：
  x → Router（Softmax）→ Top-2 专家选择
      ├── Expert_0: Linear(H, 4H) → GELU → Linear(4H, H)
      ├── Expert_1: Linear(H, 4H) → GELU → Linear(4H, H)
      ├── Expert_2: Linear(H, 4H) → GELU → Linear(4H, H)
      └── Expert_3: Linear(H, 4H) → GELU → Linear(4H, H)
  加权求和（Top-2 权重归一化）→ x
```

**路由设计：** 使用语言指令的 CLS token 特征作为路由输入（任务感知路由），而非仅用动作 token，让不同任务类型激活不同专家组合。

**辅助损失：** 添加负载均衡损失（Load Balancing Loss），防止所有 token 都路由到同一专家：

```
L_aux = α × Σ_i (f_i × P_i)
```
其中 f_i 是专家 i 处理的 token 比例，P_i 是路由到专家 i 的平均概率，α=0.01。

### 参数与显存分析

| 配置 | 参数量 | 每步激活参数 | 显存（估算） |
|------|--------|-------------|-------------|
| 密集 FFN（baseline） | 768×4×768×2×12 = 56.6M | 56.6M | ~220MB |
| MoE FFN（4专家 Top-2） | 56.6M × 4 = 226M | 56.6M × 2/4 = 28.3M | ~110MB（激活部分） |

**关键优势：** 参数量增加 4 倍，但每步激活参数减少 50%，实际计算量下降，显存峰值降低。

### 消融实验设计

| 方案 | 专家数 | Top-K | 路由输入 |
|------|--------|-------|----------|
| Baseline | 1（密集） | - | - |
| MoE-4-Token | 4 | 2 | 动作 token |
| **MoE-4-Task（主方案）** | 4 | 2 | 任务 CLS token |
| MoE-8-Task | 8 | 2 | 任务 CLS token |

### 实现要点

- 新建 `models/moe_transformer.py`，实现 `MoEFFN` 和 `MoETransformerBlock`
- `SmolVLMActionTransformer` 新增配置项 `use_moe: bool`，`num_experts: int = 4`，`top_k: int = 2`
- 训练时总损失 = Flow Matching 损失 + α × 负载均衡损失
- 推理时与密集模型接口完全相同，无需修改评估代码

---

## 创新方向三：OFP 自蒸馏推理加速（Self-Distillation for Fast Inference）

### 问题

SimVLA 推理时使用 Euler 积分（默认 10 步），每步都需要完整的 Transformer 前向传播：

- 10 步推理 = 10 × 完整前向，延迟高
- 实时控制（10 Hz）要求单次推理 < 100ms，多步积分难以满足
- 现有方法（OneDP 等）需要额外教师模型，训练成本高

### 方案设计

基于 OFP（arXiv 2603.12480）的**无教师自蒸馏**思路，在现有 Flow Matching 训练损失上增加**自洽性损失（Self-Consistency Loss）**，训练模型在 1-2 步内生成高质量动作。

**自洽性损失：**

对同一样本，分别用 t=1 和 t=0.5 作为起点进行 Euler 积分，要求两条路径在 t=0 处收敛到相同的动作：

```
x_0^(1) = Euler_integrate(x_1, t=1→0, steps=10)   # 多步参考
x_0^(2) = Euler_integrate(x_0.5, t=0.5→0, steps=1) # 单步快速

L_consistency = MSE(x_0^(1).detach(), x_0^(2))
```

**总训练损失：**
```
L_total = L_flow_matching + β × L_consistency
```
其中 β=0.1（超参数，需消融）。

**推理时：** 将 `generate_actions()` 的 `steps` 参数从 10 改为 1-2，无需重新训练。

### 速度提升预期

| 推理步数 | 延迟（估算） | 成功率损失 |
|----------|-------------|-----------|
| 10 步（baseline） | ~200ms | 0% |
| 2 步 | ~40ms | < 2% |
| 1 步 | ~20ms | < 5% |

### 消融实验设计

| 方案 | β 值 | 推理步数 |
|------|------|----------|
| Baseline（无自蒸馏） | 0 | 10 |
| β=0.01 | 0.01 | 1/2/5/10 |
| **β=0.1（主方案）** | 0.1 | 1/2/5/10 |
| β=1.0 | 1.0 | 1/2/5/10 |

### 实现要点

- 仅修改 `models/modeling_smolvlm_vla.py` 的 `forward()` 方法，增加自洽性损失计算
- 新增配置项 `consistency_loss_weight: float = 0.0`（默认 0，向后兼容）
- 推理时通过 `--inference_steps N` 参数控制步数，不需要重新训练

---

## 整体架构图

```
图像输入 [B, V, C, H, W]
         ↓
SmolVLM-500M（结构不变，权重微调）
         ↓
vlm_features [B, T_enc, 576]
         ↓
┌─────────────────────────────────────┐
│  创新点1：双流分离 + Cross-Attention  │
│  F_s（静态流）× F_d（动态流）→ F_fused│
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  SmolVLMActionTransformer           │
│  创新点2：FFN → MoE FFN（4专家）     │
│  [action_tokens, F_fused] → blocks  │
└─────────────────────────────────────┘
         ↓
预测速度 v_t [B, T_action, 7]
         ↓
┌─────────────────────────────────────┐
│  创新点3：自洽性损失（训练时）        │
│  推理时 steps=1-2（无需重训）        │
└─────────────────────────────────────┘
         ↓
动作输出 [B, 10, 7]
```

---

## 文件改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `models/transformer_smolvlm.py` | 修改 | 添加 MoE FFN、双流融合 Cross-Attention |
| `models/moe_transformer.py` | 新建 | MoEFFN、Router、负载均衡损失 |
| `models/modeling_smolvlm_vla.py` | 修改 | 双流特征分离、自洽性损失 |
| `models/configuration_smolvlm_vla.py` | 修改 | 新增配置项 |
| `train_smolvlm.py` | 修改 | 新增训练参数 |

---

## 实验计划

### 基线对比

| 模型 | 描述 |
|------|------|
| SimVLA-Base | 现有代码，无任何改动 |
| SimVLA-DS | + 双流融合（Cross-Attention） |
| SimVLA-MoE | + MoE 动作头 |
| SimVLA-Fast | + 自蒸馏（β=0.1） |
| **SimVLA-Full** | 三个创新全部启用 |

### 评估指标

- **VLABench track_debug_simple**：select_fruit + select_drink，10 episodes/task
- **VLABench track_1_in_distribution**：全量评估，50 episodes/task
- **离线 Action MSE**：`eval_action_mse.py`，快速迭代指标
- **推理延迟**：单次 `generate_actions()` 耗时（ms）

### 消融实验

1. 双流融合方式：Add vs Concat+Linear vs Cross-Attention
2. MoE 专家数：4 vs 8，路由输入：token vs task-CLS
3. 自蒸馏权重 β：0.01 / 0.1 / 1.0，推理步数：1 / 2 / 5 / 10

---

## 参考论文

| 论文 | 启发 |
|------|------|
| Cortical Policy (ICLR 2026, arXiv 2603.21051) | 双流视角设计 |
| MoE-DP (arXiv 2511.05007) | 动作头 MoE |
| OFP (arXiv 2603.12480) | 自蒸馏推理加速 |
| HiMoE-VLA (arXiv 2512.05693) | 分层 MoE 设计 |
| AdaMoE (arXiv 2510.14300) | MoE 训练稳定性 |
