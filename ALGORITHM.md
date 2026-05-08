# SimVLA 算法说明文档

> **版本**：v2.0 · 2026-05-08  
> **基础模型**：SmolVLM-500M-Instruct（Idefics3 架构）+ Flow Matching 动作头  
> **论文预印本**：https://arxiv.org/abs/2602.18224

---

## 目录

1. [符号表](#1-符号表)
2. [系统整体架构](#2-系统整体架构)
3. [数据集说明](#3-数据集说明)
4. [主要创新点（Major Contributions）](#4-主要创新点)
5. [次要改进点（Minor Contributions）](#5-次要改进点)
6. [训练流程](#6-训练流程)
7. [推理流程](#7-推理流程)
8. [创新点论文来源汇总](#8-创新点论文来源汇总)

---

## 1. 符号表

| 符号 | 含义 | 典型值 |
|---|---|---|
| B | batch size | 32 / 64 |
| V | 相机视角数 | 3（LIBERO）/ 4（VLABench） |
| T | 动作预测步数（action horizon） | 10（LIBERO）/ 30（VLABench） |
| D_a | 动作维度 | 7（xyz + euler/axis-angle + gripper） |
| D_p | 本体感知（proprio）维度 | 8（LIBERO）/ 7（VLABench） |
| D_v | SmolVLM 隐层维度 | 576 |
| H | 动作 Transformer 隐层维度 | 768（小模型）/ 1024（大模型） |
| L | VLM 输出序列长度 | 可变（视图像patch数+文本长度） |
| K | Proprio 历史窗口长度 | 1（无历史）/ 4（历史模式） |
| d_z | ActionVAE 隐变量维度 | 32 |
| t | Flow Matching 时间步 | t ∈ (0, 1) |

---

## 2. 系统整体架构

### 2.1 整体数据流

```
输入图像 [B, V, C, H, W]          语言指令 {str}
         |                              |
  SmolVLM 视觉编码器               SmolVLM 分词器
  (SigLIP-400M vision encoder)           |
         |                         文本嵌入 [B, L_text, D_v]
  图像特征 [B·V, P, D_v]               |
  ── connector 投影 ──                  |
         |                              |
  合并序列 [B, V·P + L_text, D_v]  ←──┘
         |
  SmolVLM text_model（Idefics3 LLM 主干，frozen or fine-tuned）
         |
  VLM 融合特征 F_vlm [B, L, D_v]
         |
  ┌──────┴──────────────────────┐
  │ [标准模式]                   │ [ActionVAE 模式]
  │ SmolVLMActionTransformer    │ ActionVAE + LatentFlowNet
  │ Flow Matching on [B,T,D_a]  │ Flow Matching on z [B,d_z]
  └──────┬──────────────────────┘
         |
  后处理（反归一化 + gripper 二值化）
         |
  输出动作块 â [B, T, D_a]
```

### 2.2 双流融合（可选，use_dual_stream=True）

VLM 特征输出后可经 `DualStreamFusion` 按视角分组做跨注意力融合：

- **静态流**（agentview / front / image_0 / image_1）：感知环境全局状态
- **动态流**（wrist camera）：感知末端执行器局部动态

融合公式（cross_attn 模式）：

```
F_static  = F_vlm[:, :n_static·P, :]
F_dynamic = F_vlm[:, n_static·P:(n_static+1)·P, :]

Q = F_static · W_Q,  K = F_dynamic · W_K,  V = F_dynamic · W_V
F_fused = Softmax(QK^T / √d_head) · V + F_static  (残差)
```

---

## 3. 数据集说明

### 3.1 LIBERO（HDF5 格式）

- **任务套件**：libero_10、libero_goal、libero_object、libero_spatial
- **视角**：agentview_rgb（128×128） + eye_in_hand_rgb（128×128），训练时上采样至 384×384
- **图像预处理**：旋转 180°（相机安装方向）
- **本体感知**：8 维 `[ee_pos(3), axis_angle(3), gripper_states(2)]`
- **动作**：7 维增量 `[Δxyz(3), Δeuler(3), gripper_cmd(1)]`，范围 [−1, 1]

### 3.2 VLABench（RLDS/TFRecord 格式）

- **视角**：front、wrist、image_0、image_1（4 视角）
- **图像预处理**：无旋转，直接 resize 至 384×384
- **本体感知**：7 维 `[xyz(3), euler(3), gripper(1)]`
- **动作**：7 维绝对末端执行器状态（非增量）
- **数据特点**：相邻帧动作幅度极小（< 0.001），训练时随机打乱时间索引以降低冗余

---

## 4. 主要创新点

> 以下三个创新点是本工作的**核心技术贡献**，直接来自高引用论文，有明确的方法论改进和量化收益预期。

---

### 4.1 ActionVAE 隐式扩散策略（Latent Diffusion Policy）

**论文来源**
- RoLD: Compressing Robot Learning Data via Latent Distillation，ArXiv [2403.07312](https://arxiv.org/abs/2403.07312)，ICRA 2025
- 原论文在 Diffusion Policy 上引入动作 Tokenizer，将动作块压缩至隐空间做扩散

**原论文做法**
- 用 VQVAE 对动作序列做离散化编码（codebook），在 codebook 索引上做扩散
- 训练两阶段：①预训练 VQVAE；②在离散隐空间训练扩散模型
- 结论：隐空间扩散比直接动作空间扩散收敛更快、样本质量更高

**我们的改动**
- 原论文用 VQVAE（离散），我们用**连续 VAE**（更平滑的隐空间，与 Flow Matching 更兼容）
- 原论文 backbone 是 CNN Diffusion Policy，我们集成进 VLM-VLA（SmolVLM-500M）框架
- 解码器做**跨注意力到 VLM 特征**，使动作解码具备语言和视觉上下文感知能力
- Flow Matching（非 DDPM）在隐空间做，推理步数 ≤ 10 步（比 DDPM 快 10–50×）

**架构细节**

```
【编码器 Encoder】
action [B, T, D_a]  (归一化)
    ↓  Linear → [B, T, H_vae=256]
    ↓  拼接 CLS token → [B, T+1, H_vae]
    ↓  Transformer Self-Attention × enc_depth=3 层
    ↓  取 CLS 输出 → Linear(H_vae, 2·d_z)
    → μ [B, d_z],  log σ² [B, d_z]
    → z = μ + ε · exp(0.5 · log σ²)   ← 重参数化，ε ~ N(0, I)

【解码器 Decoder】
z [B, d_z]
    ↓  Linear → reshape → [B, T, H_vae]    (T 个动作槽)
    ↓  + 位置编码 pos_dec [1, T, H_vae]
    ↓  DecBlock × dec_depth=3 层：
         Self-Attention(slots)
         Cross-Attention(slots → F_vlm)    ← 注入 VLM 语言/视觉上下文
         FFN
    ↓  LayerNorm → Linear(H_vae, D_a)
    → â [B, T, D_a]   (重建动作)

【隐空间 Flow Matching（LatentFlowNet）】
输入：z_t [B, d_z],  t [B],  F_vlm [B, L, D_v],  p [B, D_p]
    ↓  VLM GAP: F_vlm.mean(dim=1) → Linear → [B, H_flow=512]
    ↓  proprio: Linear → [B, H_flow]
    ↓  时间: sin/cos_emb(t, 64) → MLP → [B, H_flow]
    ↓  concat([z_t, vlm, prop, time]) → Linear → [B, H_flow]
    ↓  残差 MLP × depth=4 层
    ↓  LayerNorm → Linear(H_flow, d_z)
    → v_z [B, d_z]   (速度预测)
```

**训练损失**

$$\mathcal{L} = \underbrace{\mathbb{E}\left[\|v_z - (z_\text{noise} - z)\|^2\right]}_{\mathcal{L}_\text{FM}} + \lambda_r \underbrace{\mathbb{E}\left[\|\hat{a} - a\|^2\right]}_{\mathcal{L}_\text{recon}} + \beta \underbrace{\left(-\tfrac{1}{2}\mathbb{E}\left[1 + \log\sigma^2 - \mu^2 - \sigma^2\right]\right)}_{\mathcal{L}_\text{KL}}$$

其中 $\lambda_r=1.0$，$\beta=0.001$，$z_\text{noise}\sim\mathcal{N}(0,I)$，$z_t = t \cdot z_\text{noise} + (1-t) \cdot z$

**推理流程**

```
z_1 ~ N(0, I) [B, d_z]          ← 从纯噪声出发
for t = 1.0, ..., 0.0 (10步 Euler):
    v_z = LatentFlowNet(z_t, t, F_vlm, p_norm)
    z_t = z_t - (1/steps) · v_z
â = Decoder(z_0, F_vlm)         ← VAE 解码到动作序列
return postprocess(â)
```

**为什么适合 SimVLA**
- 动作空间 [B, T·D_a] = [B, 70~210] 维度较高，隐空间 d_z=32 大幅降低 flow matching 难度
- 解码器跨注意力到 VLM 特征，使动作解码具备完整语言+视觉感知（原论文 decoder 无此设计）
- VLM backbone 无需修改，ActionVAE 作为即插即用模块接在 VLM 后面

---

### 4.2 辅助运动预测头（Auxiliary Motion Prediction Head）

**论文来源**
- Robotic VLA Benefits from Joint Learning with Motion Image Diffusion，ArXiv [2512.18007](https://arxiv.org/abs/2512.18007)，2025 年 12 月

**原论文做法**
- 在 π0（PaliGemma 3B + 300M action expert）上增加 Motion Head（DiT 结构）
- Motion Head 预测完整**光流图**（optical flow image，H×W×2），捕捉帧间像素级运动场
- 双头联合训练：action loss + motion MSE loss，共享 VLM backbone
- 推理时丢弃 Motion Head，**零推理开销**
- 结果：LIBERO 成功率 97.5%（baseline ~94%），RoboTwin 真实世界 +23%

**我们的改动**
- 原论文预测**完整光流图**（高分辨率，存储计算成本高）
- 我们简化为**全局运动向量**：per-view mean optical flow，每视角 2 维 × V 视角 = 2V 维
- 光流标签计算：`cv2.calcOpticalFlowFarneback(frame_t, frame_{t+1})` 后取全图均值
- 改动量极小：VLM GAP → 2 层 MLP → 运动向量

**数学表达**

设 $F_\text{vlm} \in \mathbb{R}^{B \times L \times D_v}$，全局池化 $\bar{F} = \text{mean}_{l}(F_\text{vlm}) \in \mathbb{R}^{B \times D_v}$

$$\hat{m} = W_2 \cdot \text{SiLU}(W_1 \cdot \bar{F}) \in \mathbb{R}^{B \times 2V}$$

$$\mathcal{L}_\text{motion} = \mathbb{E}\left[\|\hat{m} - m^*\|^2\right]$$

总损失：$\mathcal{L}_\text{total} = \mathcal{L}_\text{velocity} + \lambda_m \cdot \mathcal{L}_\text{motion}$，$\lambda_m = 0.1$

**为什么适合 SimVLA**
- SmolVLM-500M 参数量是 PaliGemma 3B 的 1/6，轻量模型更需要额外监督信号引导 VLM 特征学习运动语义
- LIBERO HDF5 天然存储连续帧，光流标签可离线预处理，额外存储 < 1%
- 推理时删去 Motion Head，无延迟代价

---

### 4.3 AdaLN + Concat 混合注入模式

**论文来源**
- DiT: Scalable Diffusion Models with Transformers，ArXiv [2212.09748](https://arxiv.org/abs/2212.09748)，ICCV 2023
  - 提出 adaLN-zero：用类别/时间 embedding 调制每层的 scale/shift，实验证明优于 cross-attn 和 in-context conditioning
- π0: A Vision-Language-Action Flow Model，ArXiv [2410.24164](https://arxiv.org/abs/2410.24164)，Physical Intelligence，2024
  - 将 adaLN 引入机器人 VLA 的 action expert（300M），与 VLM（PaliGemma 3B）分离

**原论文做法**
- DiT：纯 AdaLN，低维条件（类别 or 时间 t）调制所有层的 LayerNorm 参数
- π0：纯 AdaLN，action expert 对时间 t 和 proprio 做 adaLN；VLM 特征通过 concat 注入

**我们的改动**
- 当前 SimVLA 二选一：Concat 模式（`use_adaln=False`）或纯 AdaLN 模式（`use_adaln=True`）
- 新增 **Hybrid 混合模式**（`use_adaln_hybrid=True`）：
  - **AdaLN 通道**：低维全局信号（时间 t + proprio p）→ AdaLN 注入每层 scale/shift
  - **Concat 通道**：高维 VLM 图像 token F_vlm 保留在序列里参与 Self-Attention
- 两路并行，各得其所

**数学表达**

$$\text{DiTBlock}(x, c^*)：$$

$$x' = x + \text{gate}_{msa} \cdot \text{Attn}\left(\text{AdaLN}(x, c^*)\right)$$
$$x'' = x' + \text{gate}_{mlp} \cdot \text{FFN}\left(\text{AdaLN}(x', c^*)\right)$$

其中条件 $c^* = \text{MLP}(t_\text{emb}) + \text{Linear}(p_\text{norm})$（无 VLM GAP）

序列构成：$x = [x_\text{action} \| x_\text{vlm}] \in \mathbb{R}^{B \times (T+L) \times H}$

输出仅取前 T 个 token：$\hat{v} = \text{FinalLayer}(x[:, :T, :],\ c^*)$

**对比三种模式**

| 模式 | VLM 注入方式 | 条件注入方式 | 序列长度 | 细粒度 |
|---|---|---|---|---|
| Concat | 拼入序列 | 拼入 action token | T + L | 高（位置保留） |
| AdaLN | GAP 池化入 c | AdaLN | T | 低（空间信息丢失） |
| **Hybrid（本工作）** | **拼入序列** | **AdaLN（仅 t + p）** | **T + L** | **高（两路各自最优）** |

**为什么适合 SimVLA**
- SmolVLM 输出 token 已按视角分布排列，保留空间位置信息可显著提升细粒度操作任务精度
- 时间 t 和 proprio 都是低维标量/向量，全局性强，最适合 AdaLN 调制

---

## 5. 次要改进点

> 以下四个改进点为**工程优化和鲁棒性提升**，改动量小，论文中作为 ablation study 组件。

---

### 5.1 Logit-Normal 时间采样

**来源**：Stable Diffusion 3，ArXiv [2403.03206](https://arxiv.org/abs/2403.03206)，Esser et al.，2024

**原来做法**：SimVLA 使用 Beta(1.5, 1) 采样，偏向小 t（经验调参，无理论依据）

**改动**：
$$t = \sigma(z),\quad z \sim \mathcal{N}(\mu=0,\ \sigma^2=1)$$

即 $t = \text{sigmoid}(\mathcal{N}(0, 1))$，集中在 t ≈ 0.5 附近的"中等难度"时间步

SD3 消融实验证明此策略比均匀采样 FID 显著更低，理论上对 Flow Matching 同样适用

**代码位置**：`models/modeling_smolvlm_vla.py:forward()` ～ 5 行

---

### 5.2 Proprio 历史窗口（GRU 编码）

**来源**：Diffusion Policy，ArXiv [2303.04137](https://arxiv.org/abs/2303.04137)，Chi et al.，RSS 2023

**原来做法**：SimVLA 只用单帧 proprio（无历史），无法感知末端执行器的运动趋势

**改动**：保存最近 K=4 帧 proprio 历史 `p_hist [B, K, D_p]`，用单层 GRU 编码：

$$h_K = \text{GRU}(p_1, p_2, \ldots, p_K)$$

$h_K \in \mathbb{R}^{B \times H}$ 替代原单帧线性映射，注入 AdaLN 条件或拼入 action token

**适用**：`use_adaln=True` 和 `use_adaln_hybrid=True` 模式；Concat 模式取最后一帧（向后兼容）

**代码位置**：`models/transformer_smolvlm.py`（GRU 初始化 + forward 分支）

---

### 5.3 View Dropout + 可学习缺失 Token

**来源**
- WristWorld（ArXiv [2510.07313](https://arxiv.org/abs/2510.07313)，2025）：wrist 视角缺失导致 Calvin 任务 −42.4% 成功率
- MAE（ArXiv [2111.06377](https://arxiv.org/abs/2111.06377)，He et al.，CVPR 2022）：可学习 [MASK] token 替代缺失位置

**改动**

两点独立改动，均可单独启用：

1. **可学习缺失 Token**（`use_missing_token=True`）：  
   DualStreamFusion 中缺失视角补零改为补可学习向量 $m \in \mathbb{R}^{D_v}$（零初始化），  
   使 attention 感知"此视角缺失"而非"此视角全黑"

2. **View Dropout**（`use_view_dropout=True`，概率 p=0.1）：  
   训练时随机将 wrist 视角 image_mask 置 False，强迫模型在单视角下也能给出合理动作

**代码位置**：`models/dual_stream.py`（missing token）、`datasets/dataset_smolvlm.py`（dropout）

---

### 5.4 动作分组归一化（Action Grouped Normalization）

**来源**：Octo（DeepMind，2024）、Diffusion Policy（2303.04137）——对不同语义维度独立归一化

**原来做法**：7 维动作统一 Z-score，混合处理性质差异极大的三组信号

**改动**：保持 Z-score 不变，对**gripper 维度加大损失权重**（gripper_loss_weight=2.0），推理时对 gripper 做**硬二值化**：

$$\hat{a}_\text{gripper} = \begin{cases} +1 & \text{if } a_\text{gripper} > 0 \\ -1 & \text{otherwise} \end{cases}$$

减少连续值预测在近二值信号上的抖动

**代码位置**：`models/action_hub.py:compute_loss()` 和 `postprocess()`

---

## 6. 训练流程

### 6.1 完整训练损失

**标准模式（无 ActionVAE）**：

$$\mathcal{L} = \underbrace{\mathbb{E}_{t,\epsilon}\left[\|v_\theta(x_t, t, F_\text{vlm}, p) - (\epsilon - a)\|^2_W\right]}_{\mathcal{L}_\text{FM（动作空间）}} + \lambda_m \cdot \mathcal{L}_\text{motion}$$

其中权重矩阵 $W$ 对 gripper 维度乘以 2.0，$x_t = t\epsilon + (1-t)a$，$\epsilon \sim \mathcal{N}(0, I)$

**ActionVAE 模式**：

$$\mathcal{L} = \underbrace{\mathbb{E}_{t,\epsilon_z}\left[\|v_\phi(z_t, t, F_\text{vlm}, p) - (\epsilon_z - z)\|^2\right]}_{\mathcal{L}_\text{FM（隐空间）}} + \lambda_r \cdot \mathcal{L}_\text{recon} + \beta \cdot \mathcal{L}_\text{KL}$$

### 6.2 时间采样（Logit-Normal）

$$z \sim \mathcal{N}(0, 1),\quad t = \sigma(z) = \frac{1}{1+e^{-z}},\quad t \in (0.001,\ 0.999)$$

### 6.3 优化器与学习率策略

三个参数组，分别设置学习率：

| 参数组 | 前 1000 步 | 之后 |
|---|---|---|
| `vlm`（SmolVLM backbone） | 冻结（lr=0） | lr × learning_coef |
| `transformer_core`（Transformer 主干） | 冻结 | lr |
| `action_heads`（输入/输出层） | lr | lr |

学习率调度：线性预热（warmup=2000步）+ 可选余弦衰减

---

## 7. 推理流程

### 7.1 标准模式（动作空间 Euler 积分）

```
x_1 ~ N(0, I)  [B, T, D_a]
for t = 1.0, 1-1/S, ..., 0.0 (S=10步):
    v_t = Transformer(x_t, t, F_vlm, p_norm)
    x_t = x_t - (1/S) · v_t
return ActionSpace.postprocess(x_0)  # 反归一化 + gripper 二值化
```

### 7.2 ActionVAE 模式（隐空间 Euler 积分 + VAE 解码）

```
z_1 ~ N(0, I)  [B, d_z]
for t = 1.0, ..., 0.0 (S=10步):
    v_z = LatentFlowNet(z_t, t, F_vlm, p_norm)
    z_t = z_t - (1/S) · v_z
â = VAEDecoder(z_0, F_vlm)   # cross-attn 解码到 [B, T, D_a]
return ActionSpace.postprocess(â)
```

---

## 8. 创新点论文来源汇总

| 创新点 | 重要性 | 来源论文 | ArXiv | 原论文主要场景 | 我们的改动 |
|---|---|---|---|---|---|
| ActionVAE 隐式扩散策略 | ★★★ 主要 | RoLD | 2403.07312 | CNN Diffusion Policy + VQVAE（离散） | 连续 VAE + Flow Matching + VLM cross-attn decoder |
| 辅助运动预测头 | ★★★ 主要 | Joint Motion Diffusion | 2512.18007 | π0（PaliGemma 3B）+ 完整光流图 | SmolVLM-500M + 全局光流向量（2V维） |
| AdaLN+Concat 混合注入 | ★★★ 主要 | DiT + π0 | 2212.09748 + 2410.24164 | DiT（图像生成）/ π0（纯 AdaLN） | 混合模式：AdaLN（t+p）+ Concat（VLM token） |
| Logit-Normal 时间采样 | ★★ 次要 | Stable Diffusion 3 | 2403.03206 | 图像生成（T2I） | 直接迁移至动作 Flow Matching |
| Proprio 历史窗口 | ★★ 次要 | Diffusion Policy | 2303.04137 | 纯视觉扩散策略 | GRU 编码 K=4 帧历史注入 AdaLN 条件 |
| View Dropout + Missing Token | ★★ 次要 | WristWorld + MAE | 2510.07313 + 2111.06377 | 4D 世界模型（复杂） | 轻量 dropout + 可学习 token（< 1KB） |
| 动作分组归一化 | ★ 次要 | Octo / Diffusion Policy | — | 独立维度 scaling | gripper 加权损失 + 推理二值化 |
