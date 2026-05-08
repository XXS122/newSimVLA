# 算法说明文档

> **基础工作（Baseline）**：SimVLA（arXiv:2602.18224）  
> **数据集**：VLABench（RLDS/TFRecord 格式，4 视角）  
> **本工作**：在 SimVLA 基础上提出若干系统性改进

---

## 目录

1. [符号表](#1-符号表)
2. [SimVLA Baseline 回顾](#2-simvla-baseline-回顾)
3. [本工作与 Baseline 的差异总览](#3-本工作与-baseline-的差异总览)
4. [主要创新点（Major Contributions）](#4-主要创新点)
5. [次要改进点（Minor Contributions）](#5-次要改进点)
6. [完整数据流](#6-完整数据流)
7. [训练目标](#7-训练目标)
8. [推理流程](#8-推理流程)
9. [架构图与动机图绘制说明](#9-架构图与动机图绘制说明)
10. [创新点来源对照表](#10-创新点来源对照表)

---

## 1. 符号表

| 符号 | 含义 | 值 |
|---|---|---|
| B | batch size | 32 |
| V | 相机视角数 | 4（front / wrist / image_0 / image_1） |
| T | 动作预测步数（action horizon） | 30 |
| D_a | 动作维度 | 7（xyz + euler + gripper） |
| D_p | 本体感知维度 | 7（xyz + euler + gripper） |
| D_v | SmolVLM 隐层维度 | 576 |
| H | 动作 Transformer 隐层维度 | 768 |
| L | VLM 输出序列长度 | 可变（视 patch 数 + 文本长度） |
| P | 每视角 patch 数 | ≈ 576（384×384 / 16² SigLIP） |
| K | Proprio 历史窗口长度 | 1（无历史）/ 4（历史模式） |
| d_z | ActionVAE 隐变量维度 | 32 |
| t | Flow Matching 时间步 | t ∈ (0.001, 0.999) |
| ε | 标准高斯噪声 | ε ~ N(0, I) |

---

## 2. SimVLA Baseline 回顾

SimVLA（arXiv:2602.18224）的核心设计：

1. **视觉语言骨干**：SmolVLM-500M-Instruct（Idefics3 架构），隐层 D_v=576
2. **多视角处理**：所有视角图像**拼成单流**，统一送入 SmolVLM，无视角间显式交互
3. **动作预测**：SmolVLMActionTransformer，**Concat 模式**（VLM 特征拼入动作 token 序列）
4. **本体感知**：**单帧** proprio，直接线性映射后拼入 action token
5. **时间采样**：Beta(1.5, 1) 采样，偏向小 t，无理论依据
6. **动作归一化**：7 维统一 Z-score，gripper 与连续维度等权处理
7. **视角缺失**：缺失视角用 **torch.zeros 零向量**填充
8. **辅助监督**：无任何辅助损失，仅 velocity MSE

本工作在此 baseline 上进行系统性改进，改进点分为**主要创新（对性能有显著影响）**和**次要改进（鲁棒性和工程优化）**两类。

---

## 3. 本工作与 Baseline 的差异总览

| 维度 | SimVLA Baseline | 本工作 | 重要性 |
|---|---|---|---|
| 多视角融合 | 单流 Concat 进 SmolVLM | **双流跨注意力融合** | ★★★ 主要 |
| 动作空间 | 直接在 [B,T,D_a] 做 FM | **隐空间 ActionVAE + FM** | ★★★ 主要 |
| 辅助监督 | 无 | **光流向量辅助预测头** | ★★★ 主要 |
| 条件注入方式 | 纯 Concat | **AdaLN + Concat 混合** | ★★★ 主要 |
| 时间采样 | Beta(1.5, 1) | **Logit-Normal（SD3）** | ★★ 次要 |
| 本体感知 | 单帧 proprio | **K 帧历史 + GRU 编码** | ★★ 次要 |
| 视角缺失处理 | 零向量填充 | **可学习 Missing Token + View Dropout** | ★★ 次要 |
| 动作归一化 | 统一 Z-score | **gripper 加权损失 + 推理二值化** | ★ 次要 |

---

## 4. 主要创新点

### 4.1 双流多视角融合（Dual-Stream Multi-View Fusion）

**问题**：SimVLA 将所有视角图像拼成单流送入 SmolVLM，VLM 内部通过 Self-Attention 隐式融合视角信息。这种设计忽略了不同视角的语义差异：

- **静态视角**（front / image_0 / image_1）：观测环境全局状态，感知目标物体位置与场景布局
- **动态视角**（wrist camera）：观测末端执行器局部状态，感知抓取姿态与接触关系

两类视角携带的信息性质根本不同，混入单流后静态视角的强信号可能压制腕部相机的局部动态信息。

**方法**：在 VLM 输出后引入 `DualStreamFusion` 模块，对 VLM 特征按视角显式分流并做跨注意力融合。

**架构细节**

```
VLM 输出 F_vlm [B, V·P + L_text, D_v]
    ↓  按视角切分
F_static  = F_vlm[:, :n_s·P, :]      [B, n_s·P, D_v]   n_s = V-1（静态视角）
F_dynamic = F_vlm[:, n_s·P:(n_s+1)·P, :]  [B, P, D_v]   （wrist 视角）

跨注意力融合（cross_attn 模式）：
    Q = F_static · W_Q              [B, n_s·P, D_v]
    K = F_dynamic · W_K             [B, P, D_v]
    V = F_dynamic · W_V             [B, P, D_v]
    F_fused = Softmax(QK^T/√d) · V + F_static   （残差连接）

输出 F_fused [B, n_s·P, D_v]  （替代原始单流特征）
```

支持三种融合策略（消融实验）：`add`（逐元素相加）、`concat_linear`（拼接+线性投影）、`cross_attn`（跨注意力，默认）

**缺失视角处理**：wrist 视角缺失时，用**可学习 Missing Token** m ∈ ℝ^{D_v} 作为 Key/Value，代替零填充。这使得跨注意力可以感知"此视角未观测"而非"此视角全黑"。

**为什么适合 VLABench**：VLABench 包含 4 个视角（front + wrist + image_0 + image_1），视角数量多、语义差异大，比 LIBERO 更能体现显式视角融合的收益。

---

### 4.2 ActionVAE 隐式扩散策略（Latent Diffusion Policy）

**问题**：SimVLA 在动作序列空间 [B, T, D_a]（T=30, D_a=7 → 210 维）直接做 Flow Matching，扩散网络需要在高维空间学习复杂的速度场，收敛慢、对分布尾部拟合差。

**方法（启发自 RoLD arXiv:2403.07312）**：引入 `ActionVAE` 将动作块压缩至 d_z=32 维连续隐空间，在低维隐空间做 Flow Matching，再由具备 VLM 语言/视觉上下文的解码器还原动作序列。

**与 RoLD 原论文的区别**：
- RoLD 用 **VQVAE（离散 codebook）**，我们用**连续 VAE**（梯度直通，与 FM 更兼容）
- RoLD 的 decoder 无语言/视觉条件，我们的 decoder **跨注意力到 VLM 特征**
- RoLD 基于 CNN Diffusion Policy（DDPM），我们集成进 SmolVLM-VLA（Flow Matching）

**编码器 Encoder**

$$\text{输入：} a \in \mathbb{R}^{B \times T \times D_a} \quad(\text{归一化动作块})$$

```
a [B, T, D_a]
 → Linear → [B, T, H_vae=256]
 → 拼接 CLS token → [B, T+1, H_vae]
 → + 位置编码
 → Transformer Self-Attention × 3 层
 → 取 CLS 输出 [B, H_vae]
 → Linear → [μ, log σ²] ∈ R^{B × 2·d_z}
```

重参数化采样：

$$z = \mu + \varepsilon \cdot \exp\!\left(\tfrac{1}{2}\log\sigma^2\right), \quad \varepsilon \sim \mathcal{N}(0, I_{d_z})$$

**解码器 Decoder（带 VLM 跨注意力）**

```
z [B, d_z]
 → Linear → reshape → [B, T, H_vae]   （T 个动作槽）
 → + 位置编码
 → DecBlock × 3 层：
      Self-Attention(slots)
      Cross-Attention(slots 作为 Q，F_vlm 作为 KV)   ← 注入语言/视觉上下文
      FFN
 → LayerNorm → Linear → â [B, T, D_a]
```

**隐空间 LatentFlowNet（MLP）**

```
输入：z_t [B,d_z]，t [B]，F_vlm [B,L,D_v]，p_norm [B,D_p]
  vlm_pool = mean_l(F_vlm) → Linear → [B, H_flow=512]
  prop_emb = Linear(p_norm) → [B, H_flow]
  time_emb = SinCosEmb(t, 64) → MLP → [B, H_flow]
  x = Linear([z_t ‖ vlm_pool ‖ prop_emb ‖ time_emb]) → [B, H_flow]
  x = x + ResidualMLP(x)  × 4 层
  → Linear(H_flow, d_z)
  → v_z [B, d_z]
```

**训练损失**

$$\mathcal{L} = \underbrace{\mathbb{E}\!\left[\|v_z - (\varepsilon_z - z)\|^2\right]}_{\mathcal{L}_\text{FM}} + \lambda_r \underbrace{\|\hat{a} - a\|^2}_{\mathcal{L}_\text{recon}} + \beta \underbrace{\left(-\tfrac{1}{2}\mathbb{E}\!\left[1 + \log\sigma^2 - \mu^2 - \sigma^2\right]\right)}_{\mathcal{L}_\text{KL}}$$

其中 λ_r = 1.0，β = 0.001，z_t = t·ε_z + (1−t)·z，ε_z ~ N(0, I)

---

### 4.3 辅助运动预测头（Auxiliary Motion Prediction Head）

**问题**：SimVLA 仅用 velocity MSE 监督 VLM 特征，VLM 不被要求理解"当前帧到下一帧的运动方向"，导致 VLM 特征缺乏运动语义。

**方法（启发自 arXiv:2512.18007）**：在 VLM 特征上新增轻量 2 层 MLP，预测**每视角全局光流向量**作为辅助监督，推理时删去此头，**零推理开销**。

**与原论文的区别**：原论文（π0，PaliGemma 3B）预测完整**光流图**（H×W×2 空间分辨率），标签计算和存储成本极高。我们简化为预测**全图均值光流**（per-view mean flow），每视角 2 维，V 视角共 2V = 8 维。

**数学表达**

$$\bar{F} = \frac{1}{L}\sum_{l=1}^{L} F_\text{vlm}[:,l,:] \in \mathbb{R}^{B \times D_v}$$

$$\hat{m} = W_2 \cdot \text{SiLU}(W_1 \cdot \bar{F}) \in \mathbb{R}^{B \times 2V}$$

$$\mathcal{L}_\text{motion} = \mathbb{E}\!\left[\|\hat{m} - m^*\|^2\right]$$

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{velocity} + \lambda_m \cdot \mathcal{L}_\text{motion}, \quad \lambda_m = 0.1$$

光流标签 m* 离线用 `cv2.calcOpticalFlowFarneback(frame_t, frame_{t+1})` 计算后取全图均值。

---

### 4.4 AdaLN + Concat 混合条件注入（Hybrid Conditioning）

**问题**：SimVLA 只有单一 Concat 模式，将 VLM 特征、时间 t、proprio 全部拼入 action token 序列，存在两个缺陷：
- 序列长度膨胀（L ≫ T），Self-Attention 计算量增大
- 时间 t（1 维标量）和 proprio（7 维向量）被当作 patch token 对待，建模方式不匹配其全局性质

**方法（DiT arXiv:2212.09748 + π0 arXiv:2410.24164）**：引入混合模式 `use_adaln_hybrid=True`：

- **AdaLN 通道**：低维全局信号（时间 t + proprio p）→ 经 SiLU + Linear 生成 scale/shift，注入每层 LayerNorm
- **Concat 通道**：高维 VLM 图像 token F_vlm 保留在序列中参与 Self-Attention，保留位置/细节信息

**全局条件**（不含 VLM GAP，与纯 AdaLN 的关键区别）：

$$c^* = \text{MLP}(e_\text{sin}(t,\ H)) + \text{Linear}(p_\text{norm}) \in \mathbb{R}^{B \times H}$$

**DiTBlock 前向**（序列 = action tokens + VLM tokens）：

$$[\text{shift}_1, \text{scale}_1, \text{gate}_1, \text{shift}_2, \text{scale}_2, \text{gate}_2] = \text{SiLU}(c^*) \cdot W_\text{ada}$$

$$x' = x + \text{gate}_1 \cdot \text{Attn}\!\left((1+\text{scale}_1) \cdot \text{LN}(x) + \text{shift}_1\right)$$

$$x'' = x' + \text{gate}_2 \cdot \text{FFN}\!\left((1+\text{scale}_2) \cdot \text{LN}(x') + \text{shift}_2\right)$$

输入序列 x = [x_action || x_vlm] ∈ ℝ^{B × (T+L) × H}，最终仅解码前 T 个 token

**三种模式对比**

| 模式 | t/p 注入 | VLM 注入 | 序列长度 | 空间细节 |
|---|---|---|---|---|
| Concat（baseline） | 拼入序列 | 拼入序列 | T + L | 保留 |
| 纯 AdaLN | AdaLN（含 VLM GAP） | GAP 池化 | T | **丢失** |
| **Hybrid（本工作）** | **AdaLN（仅 t+p）** | **拼入序列** | **T + L** | **保留** |

---

## 5. 次要改进点

### 5.1 Logit-Normal 时间采样

**问题**：Beta(1.5, 1) 是手调经验值，偏向小 t（高噪声区间），训练分布不均衡。

**改动**（来自 SD3 arXiv:2403.03206）：

$$z \sim \mathcal{N}(0,\ 1), \quad t = \sigma(z) = \frac{1}{1+e^{-z}}, \quad t \in (0.001,\ 0.999)$$

集中在 t ≈ 0.5 的"中等难度"时间步，SD3 消融实验验证此策略比均匀采样 FID 显著更低。动作空间（7 维）远比图像低维，时间采样的影响更直接可见。

---

### 5.2 Proprio 历史窗口（GRU 编码）

**问题**：SimVLA 只用单帧 proprio，无法感知末端执行器的运动趋势（速度方向）。

**改动**（来自 Diffusion Policy arXiv:2303.04137）：保存最近 K=4 帧历史，用单层 GRU 编码：

$$h_K = \text{GRU}(p_1, p_2, \ldots, p_K), \quad h_K \in \mathbb{R}^{B \times H}$$

h_K 替代原单帧线性映射，注入 AdaLN 条件 c*（在 Hybrid 和纯 AdaLN 模式下生效）

---

### 5.3 View Dropout + 可学习缺失 Token

**问题**：wrist 视角在机器人实际部署中容易遮挡或缺失，模型对缺失视角完全无鲁棒性。SimVLA 用零向量填充，attention 无法区分"缺失"和"全黑画面"。

**两点独立改动**：

1. **可学习 Missing Token**（启发自 MAE arXiv:2111.06377）：  
   将缺失视角的零填充替换为可学习参数 $m \in \mathbb{R}^{D_v}$（零初始化），  
   让模型感知"此位置视角缺失"

2. **View Dropout**（启发自 WristWorld arXiv:2510.07313）：  
   训练时以概率 p=0.1 随机将 wrist 视角 image_mask 置 False，  
   强迫模型在 wrist 缺失时仍能给出合理动作

---

### 5.4 动作分组归一化（Gripper 加权损失）

**问题**：gripper 命令是近二值信号（0=开 / 1=关），与连续 xyz/euler 信号性质完全不同，统一权重的 MSE 损失不能有效监督 gripper 预测。

**改动**（参考 Octo，DeepMind 2024）：

- **训练**：对 gripper 维度损失乘以权重 w_g = 2.0  
  $$\mathcal{L}_\text{FM} = \mathbb{E}\!\left[\sum_{d=1}^{D_a} w_d (v_d - u_d)^2\right], \quad w_d = \begin{cases} 2.0 & d = \text{gripper} \\ 1.0 & \text{otherwise} \end{cases}$$

- **推理**：对 gripper 维度做硬二值化，消除连续值抖动  
  $$\hat{a}_\text{gripper} = \text{sign}(a_\text{gripper})$$

---

## 6. 完整数据流

```
输入
  images   [B, V=4, C, H=384, W=384]   front / wrist / image_0 / image_1
  text     {str} × B                   语言指令
  proprio  [B, D_p=7]  或  [B, K, D_p]  末端状态（可选历史）

───────────────────────────────────────────────
Step 1  SmolVLM 视觉编码
───────────────────────────────────────────────
  flatten images → [B·V, C, H, W]
  SigLIP-400M vision encoder → [B·V, P, D_v=576]
  connector 投影 → [B·V, P, D_v]
  per-sample concat valid views + text_embed → [B, V·P+L_text, D_v]
  Idefics3 text_model (Transformer) → F_vlm [B, L, D_v]

───────────────────────────────────────────────
Step 2  双流多视角融合（可选，use_dual_stream=True）
───────────────────────────────────────────────
  F_static  = F_vlm[:, :n_s·P, :]      静态视角特征
  F_dynamic = F_vlm[:, n_s·P:..., :]   wrist 视角特征（或 missing token）
  Cross-Attention(Q=F_static, K=V=F_dynamic) + 残差
  → F_fused [B, n_s·P, D_v]

───────────────────────────────────────────────
Step 3  本体感知编码
───────────────────────────────────────────────
  if K > 1:  GRU(p_hist [B,K,D_p]) → h_K [B, H]
  else:       Linear(p [B,D_p])   → p_cond [B, H]

───────────────────────────────────────────────
Step 4A  标准 Flow Matching（transformer is not None）
───────────────────────────────────────────────
  t ~ Logit-Normal(0, 1)          时间采样
  x_t = t·ε + (1-t)·a_norm       插值
  u_t = ε - a_norm                目标速度
  
  Hybrid 模式：
    c* = MLP(sin_emb(t)) + Linear(p_cond)
    x_seq = [action_encoder(x_t) ‖ vlm_proj(F_fused)]
    for DiTBlock in blocks:  x_seq = DiTBlock(x_seq, c*)
    v_t = FinalLayer(x_seq[:, :T], c*)

  L_FM = MSE(v_t, u_t) with gripper weight

───────────────────────────────────────────────
Step 4B  ActionVAE 模式（use_action_vae=True）
───────────────────────────────────────────────
  [编码]
    CLS || action_proj(a_norm) → Enc-Transformer → μ, log σ²
    z = μ + ε_vae · exp(0.5·log σ²)

  [重建]
    z_slots = z_to_slots(z) → Dec-Transformer(Cross-Attn to F_fused) → â
    L_recon = MSE(â, a_norm)
    L_KL = -½·E[1 + log σ² - μ² - σ²]

  [隐空间 Flow Matching]
    t ~ Logit-Normal(0, 1)
    z_t = t·ε_z + (1-t)·z
    v_z = LatentFlowNet(z_t, t, F_fused, p_norm)
    L_FM = MSE(v_z, ε_z - z)

  [辅助损失（可选）]
    motion_pred = MLP(mean_l(F_fused))  → [B, 2V]
    L_motion = MSE(motion_pred, m*)
    L_total = L_FM + λ_r·L_recon + β·L_KL + λ_m·L_motion
```

---

## 7. 训练目标

**标准模式**：

$$\mathcal{L} = \mathbb{E}_{t,\varepsilon}\!\left[\sum_d w_d(v_d - u_d)^2\right] + \lambda_m \cdot \mathcal{L}_\text{motion}$$

**ActionVAE 模式**：

$$\mathcal{L} = \mathcal{L}_\text{FM}(z) + \lambda_r \cdot \mathcal{L}_\text{recon} + \beta \cdot \mathcal{L}_\text{KL} + \lambda_m \cdot \mathcal{L}_\text{motion}$$

| 超参数 | 含义 | 值 |
|---|---|---|
| λ_r | VAE 重建损失权重 | 1.0 |
| β | KL 散度权重（β-VAE） | 0.001 |
| λ_m | 运动辅助损失权重 | 0.1 |
| w_g | gripper 损失加权 | 2.0 |

---

## 8. 推理流程

### 8.1 标准模式

```
x_1 ~ N(0, I)  [B, T=30, D_a=7]
for step in 1..S (S=10 Euler 步):
    t_tensor = ones(B) * (1 - step/S)
    v_t = ActionTransformer(x_t, t_tensor, F_fused, p_norm)
    x_t = x_t - (1/S) · v_t
â = postprocess(x_0)   # 反归一化 + gripper 二值化
```

### 8.2 ActionVAE 模式

```
z_1 ~ N(0, I)  [B, d_z=32]
for step in 1..S (S=10):
    t_tensor = ones(B) * (1 - step/S)
    v_z = LatentFlowNet(z_t, t_tensor, F_fused, p_norm)
    z_t = z_t - (1/S) · v_z
â = VAEDecoder(z_0, F_fused)   # cross-attn to VLM → [B, T, D_a]
â = postprocess(â)
```

---

## 9. 架构图与动机图绘制说明

### 9.1 架构图（Architecture Diagram）

**整体布局**：从左到右分四列，从上到下表示数据流向。

```
第一列（输入层）
┌─────────────────────────────────┐
│ 4 个相机视角图像                 │  front / wrist / image_0 / image_1
│ [4个小矩形，wrist 用虚线框突出]  │  分别标注 384×384
│                                  │
│ 语言指令（文本框）               │  "pick up the cup"
│                                  │
│ 本体感知 p [7维]                 │  xyz + euler + gripper
└─────────────────────────────────┘
           ↓
第二列（SmolVLM 视觉语言编码）
┌─────────────────────────────────┐
│  SigLIP-400M 视觉编码器          │  ← 所有4视角扁平化输入
│  connector 投影层                │
│  Idefics3 LLM 主干（可训练）    │  ← 文本 token 拼入后 forward
└─────────────────────────────────┘
    输出：F_vlm [B, L, 576]
           ↓
第三列（本工作新增模块，用彩色边框区分）
┌─────────────────────────────────┐
│  ① 双流多视角融合（橙色框）      │
│     静态流 F_static              │
│     动态流 F_dynamic（wrist）   │  ← Missing Token 替代零填充
│     跨注意力 ↕                   │
│     F_fused [B, n_s·P, 576]     │
└─────────────────────────────────┘
    同时：
┌─────────────────────────────────┐
│  ② Proprio GRU 历史编码（绿色框）│
│     p_hist [B, K=4, 7]          │
│     GRU → h_K [B, H]            │
└─────────────────────────────────┘
           ↓
    两条并行路径（用分叉箭头表示）：

路径 A（标准模式）               路径 B（ActionVAE 模式，用紫色框）
┌────────────────┐              ┌────────────────────────────────┐
│ AdaLN+Concat   │              │ ActionVAE Encoder               │
│ Hybrid Transformer│           │ a → μ,σ → z [B,32]             │
│                │              │                                  │
│ c* = t_emb+p   │              │ LatentFlowNet（z_t,t,F,p → v_z）│
│ DiTBlock×12   │              │                                  │
│               │              │ ActionVAE Decoder                │
│ 输出 v_t       │              │ z + F_fused → â                 │
└───────┬────────┘              └──────────────┬─────────────────┘
        │                                       │
        ↓                                       ↓
  Euler 积分 → x_0                    Euler 积分 → z_0 → decode
        │                                       │
        └──────────────┬────────────────────────┘
                       ↓
第四列（输出与辅助头）
┌─────────────────────────────────┐
│  动作后处理                      │
│  反归一化 + gripper 二值化       │
│  输出 â [B, T=30, D_a=7]        │
└─────────────────────────────────┘

旁边单独一个小框（虚线，表示仅训练时使用）：
┌───────────────────────────┐
│ ③ 运动预测头（灰色虚线框）  │
│  F_fused → GAP → MLP      │
│  → 光流向量 [B, 2×4=8]    │
│  （推理时删除，零推理开销） │
└───────────────────────────┘
```

**图注说明**：
- 实线框 = 从 SimVLA 继承的模块
- **彩色实线框** = 本工作新增/改动的模块（用不同颜色区分 ①②③）
- 虚线框 = 仅训练时使用的辅助模块
- F_vlm / F_fused 等变量标注在箭头上

---

### 9.2 动机图（Motivation Diagram）

动机图用于解释"为什么需要这些改进"，建议做成 **2×2 的格子图** 或 **问题-方案对比图**，每格包含：左侧问题描述（可能配性能曲线/示意图），右侧解决方案。

**推荐布局：2 行 × 2 列**

```
┌───────────────────────────┬───────────────────────────┐
│ 问题①：单流视角融合信息丢失  │ 问题②：动作空间维度过高     │
│                            │                            │
│ [示意图：4条视角特征流合并  │ [示意图：动作序列 [T×D_a]  │
│  后进入LM，wrist信息被稀释] │  与隐向量 d_z 维度对比     │
│         ↓                  │         ↓                  │
│ 解决：双流融合              │ 解决：ActionVAE 隐空间 FM  │
│ Static ⇄ CrossAttn ⇄ Wrist │ a[210维] → z[32维] → FM   │
├───────────────────────────┼───────────────────────────┤
│ 问题③：VLM 不理解运动方向  │ 问题④：条件注入方式不匹配   │
│                            │                            │
│ [示意图：VLM 特征空间无运  │ [对比图：三种注入方式的     │
│  动语义；光流可视化示例]    │  计算量 vs 空间细节保留]    │
│         ↓                  │         ↓                  │
│ 解决：Motion Head 辅助监督  │ 解决：AdaLN+Concat Hybrid │
│ VLM Pool → MLP → 光流向量  │ t/p → AdaLN; VLM → Concat │
└───────────────────────────┴───────────────────────────┘
```

**各格配图建议**：

- **格①（双流融合）**：画 Attention 热力图对比——单流时 wrist 视角特征的 attention 权重极低，双流融合后 wrist attention 显著激活
- **格②（ActionVAE）**：画 loss curve 对比——隐空间 FM（d_z=32）比直接动作空间 FM（210维）收敛更快；或用 t-SNE 可视化隐空间的语义聚类
- **格③（Motion Head）**：展示两帧图像之间的光流向量（箭头可视化），以及有/无 Motion Head 时 VLM 特征的 PCA 分布差异
- **格④（Hybrid Conditioning）**：画 3 列柱状图——Concat / AdaLN / Hybrid 在序列长度（计算量代理指标）和下游任务成功率的对比

---

## 10. 创新点来源对照表

| 创新点 | 重要性 | 来源论文 | ArXiv | 原论文场景 | 本工作改动 |
|---|---|---|---|---|---|
| 双流多视角融合 | ★★★ 主要 | — | 本工作原创 | 无对应 | 对 VLM 输出按视角显式分流，wrist 跨注意力融合 |
| ActionVAE 隐式扩散 | ★★★ 主要 | RoLD | 2403.07312 | CNN DP + VQVAE（离散） | 连续 VAE + FM；decoder 跨注意力 VLM |
| 辅助运动预测头 | ★★★ 主要 | Joint Motion Diffusion | 2512.18007 | π0（3B）+ 完整光流图 | SmolVLM-500M + 全局光流向量（2V维） |
| AdaLN+Concat 混合 | ★★★ 主要 | DiT + π0 | 2212.09748 + 2410.24164 | 图像生成 / 纯 AdaLN | 首次混合：AdaLN（t+p）+ Concat（VLM token） |
| Logit-Normal 采样 | ★★ 次要 | SD3 | 2403.03206 | 图像生成 | 直接迁移至动作 FM |
| Proprio 历史 GRU | ★★ 次要 | Diffusion Policy | 2303.04137 | 纯视觉扩散 | K=4 帧 GRU 注入 AdaLN 条件 |
| View Dropout + Token | ★★ 次要 | WristWorld + MAE | 2510.07313 + 2111.06377 | 4D 世界模型 | 轻量 dropout + 可学习 token |
| 动作分组归一化 | ★ 次要 | Octo / DP | — | 独立维度 scaling | gripper 加权损失 + 推理二值化 |
