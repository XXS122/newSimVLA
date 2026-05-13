"""
SmolVLM Action Transformer

核心功能：
  - 专为 SmolVLM-VLA 设计的动作 Transformer
  - 与原始 transformer 的关键区别：
    - 无 aux_visual_inputs：所有视角由 SmolVLM 一起处理
    - VLM 输出统一的特征表示
    - 更简单的架构：x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)

支持三种模式：
  - Concat 模式（use_adaln=False）：VLM 特征拼接到动作 token 序列
  - AdaLN 模式（use_adaln=True）：时间/本体感知通过自适应层归一化注入
  - Hybrid 模式（use_adaln_hybrid=True）：AdaLN 注入低维信号，Concat 保留 VLM token
"""

from __future__ import annotations

import math
from functools import partial
from typing import Final, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- 小工具函数 ----------------------------------

def _to_2tuple(x) -> Tuple:
    """将标量转换为 2 元组（timm.layers.to_2tuple 的最小替代）"""
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


def _has_sdp_attention() -> bool:
    """检查是否可以使用 PyTorch 融合的 scaled_dot_product_attention"""
    return hasattr(F, "scaled_dot_product_attention")


# ---------------------------------- MLP --------------------------------------

class Mlp(nn.Module):
    """ViT 风格块中使用的 MLP"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -------------------------------- Attention ----------------------------------

class Attention(nn.Module):
    """多头自注意力，支持可选的融合 SDPA 回退"""

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = _has_sdp_attention()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            # 使用 PyTorch 融合的注意力（更高效）
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            # 手动计算注意力
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ------------------------------- 工具函数 -----------------------------------

def basic_init(module: nn.Module) -> None:
    """对 Linear 层应用基本初始化"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """
    创建正弦时间步嵌入

    参数
    ----
    t : torch.Tensor
        时间步 [B]
    dim : int
        嵌入维度
    max_period : int
        最大周期（默认 100）

    返回
    ----
    torch.Tensor
        时间步嵌入 [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ------------------------------- 核心层 --------------------------------------

class TransformerBlock(nn.Module):
    """标准 Transformer 块（pre-LN）"""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------- DiT 层（AdaLN）----------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    AdaLN 调制：x * (1 + scale) + shift

    这是 DiT 的核心：通过自适应层归一化注入条件信息
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT 块，使用自适应层归一化（AdaLN）

    核心设计：
      - 条件 c（时间 + 本体感知 + VLM）通过 AdaLN 注入
      - 生成 6 个调制参数：shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
      - gate 参数从 0 初始化，训练初期等效标准 Transformer，随训练逐渐学会利用条件

    参考：DiT (arxiv:2212.09748)
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # 无仿射参数的 LayerNorm（由 AdaLN 控制）
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

        # AdaLN 调制网络：条件 c → 6 个调制参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # 零初始化，训练初期等效标准 Transformer
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : torch.Tensor
            输入特征 [B, T, H]
        c : torch.Tensor
            条件向量 [B, H]（时间 + 本体感知 + VLM）

        返回
        ----
        torch.Tensor
            输出特征 [B, T, H]
        """
        # 生成调制参数
        modulation_params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation_params.chunk(6, dim=-1)
        )

        # 自注意力 + AdaLN 调制
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm)

        # MLP + AdaLN 调制
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class FinalLayer(nn.Module):
    """
    DiT 最终层，使用 AdaLN

    核心设计：
      - 条件 c 通过 AdaLN 调制最终的 LayerNorm
      - 线性层从 0 初始化，训练初期输出接近 0
      - 这保证了训练初期的稳定性
    """

    def __init__(self, hidden_size: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)

        # 零初始化，保证训练初期稳定性
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : torch.Tensor
            输入特征 [B, T, H]
        c : torch.Tensor
            条件向量 [B, H]

        返回
        ----
        torch.Tensor
            输出动作 [B, T, D_action]
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# --------------------------- 主模型（SmolVLM 版本）---------------------------------------

class SmolVLMActionTransformer(nn.Module):
    """
    Flow Matching Transformer，用于动作预测 - SmolVLM 版本

    与 ActionTransformer 的关键区别：
      - 无 aux_visual_inputs：SmolVLM 一起处理所有视角
      - 更简单的前向传播：x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)
      - 只有一个视觉输入流

    支持两种模式：
      - Concat 模式（use_adaln=False）：原始架构
      - AdaLN 模式（use_adaln=True）：DiT 风格的条件注入
      - Hybrid 模式（use_adaln_hybrid=True）：AdaLN + Concat 混合

    参数
    ----
    hidden_size : int
        Transformer 隐藏层维度（默认 768）
    vlm_hidden_size : int
        VLM 特征维度（SmolVLM-500M 为 576）
    depth : int
        Transformer 层数（默认 12）
    num_heads : int
        注意力头数（默认 12）
    mlp_ratio : float
        MLP 隐藏层倍率（默认 4.0）
    dim_action : int
        动作维度（VLABench 为 7）
    dim_propio : int
        本体感知维度（VLABench 为 7）
    dim_time : int
        时间嵌入维度（默认 32）
    max_len_seq : int
        最大序列长度（默认 512）
    use_adaln : bool
        是否使用 AdaLN 模式
    proprio_history_len : int
        本体感知历史窗口长度（K=1 无历史，K>1 使用 GRU 编码）
    use_adaln_hybrid : bool
        是否使用混合模式（AdaLN + Concat）
    """

    def __init__(
        self,
        hidden_size: int = 768,
        vlm_hidden_size: int = 576,  # 将被实际模型配置覆盖
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dim_action: int = 26,
        dim_propio: int = 21,
        dim_time: int = 32,
        max_len_seq: int = 1024,
        use_adaln: bool = False,
        proprio_history_len: int = 1,  # K=1 → 无历史（原行为）；K>1 → 使用 GRU 编码历史
        use_adaln_hybrid: bool = False,  # 混合模式：AdaLN(time+proprio) + Concat(VLM token)
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.dim_propio = dim_propio
        self.use_adaln = use_adaln
        self.use_adaln_hybrid = use_adaln_hybrid
        self.proprio_history_len = proprio_history_len

        if use_adaln_hybrid:
            # ====== 混合模式：AdaLN（time+proprio）+ Concat（VLM token 保留在序列里）======
            # 参考 DiT (arxiv:2212.09748) + π0 (arxiv:2410.24164)
            # DiTBlock 处理全局低维条件；VLM 图像 token 保留空间细节
            self.blocks = nn.ModuleList(
                [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
            )

            # 全局条件编码（time + proprio），不再池化 VLM
            self.time_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

            # 本体感知编码：GRU（K>1）或线性映射（K=1）
            if proprio_history_len > 1:
                self.proprio_gru = nn.GRU(dim_propio, hidden_size, batch_first=True)
                self.proprio_proj = None
            else:
                self.proprio_proj = nn.Linear(dim_propio, hidden_size)
                self.proprio_gru = None

            # VLM 投影到 action 空间（拼接进序列）
            self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)

            # Action encoder（无需携带 time/proprio，它们由 AdaLN 注入）
            self.action_encoder = nn.Linear(dim_action, hidden_size)
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
            nn.init.normal_(self.pos_emb, std=0.02)
            self.final_layer = FinalLayer(hidden_size, dim_action)

        elif use_adaln:
            # ========== DiT 模式：AdaLN ==========
            self.blocks = nn.ModuleList(
                [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
            )

            # 条件编码器
            self.time_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

            # VLM 池化投影（无 aux_visual 需要）
            self.vlm_cond_proj = nn.Linear(vlm_hidden_size, hidden_size)

            # 本体感知条件：GRU（K>1）或线性映射（K=1）
            # 参考 Diffusion Policy (arxiv:2303.04137) 的 observation horizon
            if proprio_history_len > 1:
                self.proprio_gru = nn.GRU(
                    input_size=dim_propio,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                self.proprio_proj = None  # GRU 替代线性映射
            else:
                self.proprio_proj = nn.Linear(dim_propio, hidden_size)
                self.proprio_gru = None

            # Action encoder
            self.action_encoder = nn.Linear(dim_action, hidden_size)

            # 位置编码
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
            nn.init.normal_(self.pos_emb, std=0.02)

            # 最终层
            self.final_layer = FinalLayer(hidden_size, dim_action)

        else:
            # ========== Concat 模式：原始架构 ==========
            self.blocks = nn.ModuleList(
                [TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
            )

            # VLM 投影（无 aux_visual_proj 需要）
            self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)

            self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
            nn.init.normal_(self.pos_emb, std=0.02)

            self.norm = nn.LayerNorm(hidden_size)

            # Action encoder/decoder
            action_input_dim = dim_action + dim_time + dim_propio
            self.action_encoder = nn.Linear(action_input_dim, hidden_size)
            self.action_decoder = nn.Linear(hidden_size, dim_action)

        # 应用基本初始化
        self.apply(basic_init)

    def forward(
        self,
        vlm_features: torch.Tensor,  # [B, T_vlm, D] - SmolVLM 的统一特征
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        SmolVLM Action Transformer 前向传播

        参数
        ----
        vlm_features : torch.Tensor
            SmolVLM 的统一特征 [B, T_vlm, D]（所有视角一起处理）
        action_with_noise : torch.Tensor
            带噪声的动作 [B, T_action, dim_action]
        proprio : torch.Tensor
            本体感知 [B, dim_proprio]
        t : torch.Tensor
            时间步 [B]

        返回
        ----
        torch.Tensor
            预测的速度 [B, T_action, dim_action]
        """
        if self.use_adaln_hybrid:
            return self._forward_hybrid(vlm_features, action_with_noise, proprio, t)
        elif self.use_adaln:
            return self._forward_adaln(vlm_features, action_with_noise, proprio, t)
        else:
            return self._forward_concat(vlm_features, action_with_noise, proprio, t)

    def _forward_concat(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concat 模式前向传播

        简化：x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)
        无需 aux_visual_inputs
        """
        B, num_actions = action_with_noise.shape[:2]

        # 编码 (action + proprio + time) → tokens
        # Concat 模式只用最新帧 proprio（K>1 时取最后一帧，向后兼容）
        if proprio.dim() == 3:
            proprio = proprio[:, -1, :]  # [B, K, D] → [B, D]

        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions, proprio.shape[-1])

        # 拼接动作、本体感知、时间
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens)  # [B, T_action, H]

        # 投影 VLM 特征并拼接（无 aux_visual 需要）
        x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)

        # 添加位置编码
        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}."
            )
        x = x + self.pos_emb[:, :seq_len, :]

        # Transformer 骨干网络
        for block in self.blocks:
            x = block(x)

        # 只解码动作段
        return self.action_decoder(self.norm(x[:, :num_actions]))

    def _forward_adaln(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        DiT/AdaLN 模式前向传播

        条件（时间、VLM、本体感知）通过 AdaLN 注入
        无需 aux_visual（SmolVLM 处理所有视角）
        """
        B, num_actions = action_with_noise.shape[:2]

        # ========== 1. 构建全局条件 c ==========
        # 时间嵌入
        t_emb = timestep_embedding(t, self.hidden_size)
        t_emb = self.time_proj(t_emb)  # [B, H]

        # VLM 条件：全局平均池化
        vlm_cond = self.vlm_cond_proj(vlm_features.mean(dim=1))  # [B, H]

        # 本体感知条件：GRU 历史编码（K>1）或单帧线性映射（K=1）
        if self.proprio_gru is not None:
            # proprio: [B, K, dim_proprio] 或 [B, dim_proprio]（自动 unsqueeze）
            if proprio.dim() == 2:
                proprio = proprio.unsqueeze(1)  # [B, 1, D]
            _, h_n = self.proprio_gru(proprio)  # h_n: [1, B, H]
            proprio_cond = h_n.squeeze(0)       # [B, H]
        else:
            if proprio.dim() == 3:
                proprio = proprio[:, -1, :]     # 仅用最新帧，兼容历史输入
            proprio_cond = self.proprio_proj(proprio)  # [B, H]

        # 融合所有条件
        c = t_emb + vlm_cond + proprio_cond  # [B, H]

        # ========== 2. 编码动作序列 ==========
        x = self.action_encoder(action_with_noise)  # [B, T_action, H]

        # 添加位置编码
        x = x + self.pos_emb[:, :num_actions, :]

        # ========== 3. DiT Blocks + AdaLN ==========
        for block in self.blocks:
            x = block(x, c)

        # ========== 4. 最终层 + AdaLN ==========
        return self.final_layer(x, c)

    def _forward_hybrid(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        混合模式 Forward：AdaLN（time+proprio）+ Concat（VLM token 保留在序列里）

        参考论文：
          - DiT (arxiv:2212.09748): AdaLN-zero 为最优条件注入方式
          - π0 (arxiv:2410.24164): action expert 使用 AdaLN

        设计思路：
          - 低维全局信号（时间 t、本体感知 proprio）→ AdaLN 注入（计算高效、全局一致）
          - 高维空间信号（VLM 图像 token）→ Concat 拼接（保留位置/细节信息）
        """
        B, num_actions = action_with_noise.shape[:2]

        # ===== 1. 全局条件 c = time + proprio（不包含 VLM GAP）=====
        t_emb = timestep_embedding(t, self.hidden_size)
        t_emb = self.time_proj(t_emb)                     # [B, H]

        if self.proprio_gru is not None:
            if proprio.dim() == 2:
                proprio = proprio.unsqueeze(1)
            _, h_n = self.proprio_gru(proprio)
            proprio_cond = h_n.squeeze(0)
        else:
            if proprio.dim() == 3:
                proprio = proprio[:, -1, :]
            proprio_cond = self.proprio_proj(proprio)     # [B, H]

        c = t_emb + proprio_cond                          # [B, H]（无 VLM GAP，保留细节）

        # ===== 2. 构建输入序列：action token + VLM image token =====
        x_action = self.action_encoder(action_with_noise)          # [B, T_action, H]
        x_vlm    = self.vlm_proj(vlm_features)                     # [B, T_vlm, H]
        x = torch.cat([x_action, x_vlm], dim=1)                    # [B, T_action+T_vlm, H]

        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(f"序列长度 {seq_len} 超过 max_len_seq={self.pos_emb.shape[1]}")
        x = x + self.pos_emb[:, :seq_len, :]

        # ===== 3. DiTBlock（AdaLN 注入 c，序列包含 VLM token）=====
        for block in self.blocks:
            x = block(x, c)

        # ===== 4. 只解码 action 段 =====
        x_out = x[:, :num_actions, :]                              # [B, T_action, H]
        return self.final_layer(x_out, c)


__all__ = [
    "SmolVLMActionTransformer",
    "TransformerBlock",
    "DiTBlock",
    "FinalLayer",
    "Attention",
    "Mlp",
    "timestep_embedding",
]
