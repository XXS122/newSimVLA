"""
ActionVAE: 动作块的变分自编码器

核心功能：
  - 基于 RoLD (arxiv:2403.07312) 的动作序列编码器
  - 将动作轨迹 [B,T,D_a] 压缩到紧凑的隐变量 z [B,d_z]
  - 在低维隐空间做 Flow Matching，而不是在完整动作序列空间
  - 推理时：Euler 积分得到 z，再解码为动作序列

优势：
  - 隐空间维度 d_z=32 远小于动作序列维度 T·D_a=10·7=70
  - Flow Matching 在低维空间收敛更快
  - 解码器通过跨注意力到 VLM 特征，实现上下文感知的重建

组件：
  - ActionVAE：Transformer 编码器 → (μ,log σ²) → z；跨注意力解码器 → â
  - LatentFlowNet：MLP 速度场，在隐空间 [B,d_z] 做 Flow Matching
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── 辅助函数 ───────────────────────────────────────

def _sinusoidal_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    正弦时间步嵌入

    参数
    ----
    t : torch.Tensor
        时间步 [B]
    dim : int
        嵌入维度

    返回
    ----
    torch.Tensor
        时间步嵌入 [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(100) * torch.arange(half, dtype=t.dtype, device=t.device) / half
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ─────────────────────────── 注意力块 ───────────────────────────────────────

class _SelfAttn(nn.Module):
    """
    自注意力块

    用于 ActionVAE 编码器，处理动作序列的内部关系
    """

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=True)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : torch.Tensor
            输入特征 [B, T, C]

        返回
        ----
        torch.Tensor
            输出特征 [B, T, C]
        """
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class _CrossAttn(nn.Module):
    """
    跨注意力块

    用于 ActionVAE 解码器，让动作序列关注 VLM 特征
    实现上下文感知的动作重建
    """

    def __init__(self, hidden: int, heads: int, kv_dim: int):
        """
        参数
        ----
        hidden : int
            隐藏层维度
        heads : int
            注意力头数
        kv_dim : int
            Key/Value 的维度（VLM 特征维度）
        """
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=True)
        self.kv_proj = nn.Linear(kv_dim, 2 * hidden, bias=True)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : torch.Tensor
            查询特征 [B, T, C]（动作序列）
        ctx : torch.Tensor
            上下文特征 [B, L, kv_dim]（VLM 特征）

        返回
        ----
        torch.Tensor
            输出特征 [B, T, C]
        """
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(ctx).reshape(B, -1, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, C))


class _EncBlock(nn.Module):
    """
    编码器块：自注意力 + FFN

    用于 ActionVAE 编码器，处理动作序列
    """

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.attn = _SelfAttn(hidden, heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _DecBlock(nn.Module):
    """
    解码器块：自注意力 + 跨注意力（到 VLM）+ FFN

    用于 ActionVAE 解码器，实现上下文感知的动作重建
    """

    def __init__(self, hidden: int, heads: int, vlm_dim: int):
        """
        参数
        ----
        hidden : int
            隐藏层维度
        heads : int
            注意力头数
        vlm_dim : int
            VLM 特征维度
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.self_attn = _SelfAttn(hidden, heads)
        self.cross_attn = _CrossAttn(hidden, heads, vlm_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden)
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : torch.Tensor
            输入特征 [B, T, H]（动作序列）
        ctx : torch.Tensor
            上下文特征 [B, L, vlm_dim]（VLM 特征）

        返回
        ----
        torch.Tensor
            输出特征 [B, T, H]
        """
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), ctx)
        x = x + self.mlp(self.norm3(x))
        return x


# ─────────────────────────── ActionVAE ──────────────────────────────────────

class ActionVAE(nn.Module):
    """
    动作块的 VAE（RoLD arxiv:2403.07312）

    编码器流程：
      1. 输入：action [B, T, D_a]（归一化）
      2. 线性投影 → [B, T, H]
      3. 添加可学习 CLS token → [B, T+1, H]
      4. Transformer 自注意力（enc_depth 层）
      5. CLS 输出 → Linear → [μ, log σ²] ∈ R^{2·d_z}
      6. 重参数化：z = μ + ε·exp(0.5·log σ²)

    解码器流程：
      1. 输入：z [B, d_z] + VLM 特征 [B, L, D_vlm]
      2. 线性 z → [B, T, H]（T 个动作槽）
      3. dec_depth 层 (自注意力 → 跨注意力到 VLM → FFN)
      4. 线性 → [B, T, D_a] 重建的动作

    参数
    ----
    dim_action : int
        动作维度（VLABench 为 7）
    seq_len : int
        动作序列长度（默认 10）
    vlm_hidden : int
        VLM 特征维度（SmolVLM-500M 为 576）
    latent_dim : int
        隐变量维度（默认 32）
    hidden : int
        隐藏层维度（默认 256）
    enc_depth : int
        编码器深度（默认 3）
    dec_depth : int
        解码器深度（默认 3）
    heads : int
        注意力头数（默认 4）
    """

    def __init__(
        self,
        dim_action: int,
        seq_len: int,
        vlm_hidden: int,
        latent_dim: int = 32,
        hidden: int = 256,
        enc_depth: int = 3,
        dec_depth: int = 3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.dim_action = dim_action
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden = hidden

        # ── 编码器 ──
        self.enc_action_proj = nn.Linear(dim_action, hidden)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.cls_token, std=0.02)
        self.enc_pos = nn.Parameter(torch.zeros(1, seq_len + 1, hidden))
        nn.init.normal_(self.enc_pos, std=0.02)
        self.enc_blocks = nn.ModuleList([_EncBlock(hidden, heads) for _ in range(enc_depth)])
        self.enc_norm = nn.LayerNorm(hidden)
        self.to_mu_logvar = nn.Linear(hidden, 2 * latent_dim)

        # ── 解码器 ──
        self.z_to_slots = nn.Linear(latent_dim, seq_len * hidden)
        self.dec_pos = nn.Parameter(torch.zeros(1, seq_len, hidden))
        nn.init.normal_(self.dec_pos, std=0.02)
        self.dec_blocks = nn.ModuleList([_DecBlock(hidden, heads, vlm_hidden) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(hidden)
        self.action_out = nn.Linear(hidden, dim_action)

        # 零初始化输出层，训练初期输出接近 0
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def encode(
        self, action: torch.Tensor, vlm_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码动作块 → (μ, log σ², z)

        参数
        ----
        action : torch.Tensor
            归一化的动作序列 [B, T, D_a]
        vlm_features : torch.Tensor
            VLM 特征 [B, L, D_vlm]（编码器未使用，保持 API 对称）

        返回
        ----
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            mu, log_var : [B, d_z]
            z : [B, d_z]（训练时重参数化采样，评估时 = mu）
        """
        B = action.shape[0]

        # 1. 线性投影
        x = self.enc_action_proj(action)                          # [B, T, H]

        # 2. 添加 CLS token
        cls = self.cls_token.expand(B, -1, -1)                    # [B, 1, H]
        x = torch.cat([cls, x], dim=1) + self.enc_pos             # [B, T+1, H]

        # 3. Transformer 编码
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)

        # 4. 提取 CLS token → μ, log σ²
        mu_logvar = self.to_mu_logvar(x[:, 0])                    # [B, 2·d_z]
        mu, log_var = mu_logvar.chunk(2, dim=-1)

        # 5. 重参数化采样
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        else:
            z = mu
        return mu, log_var, z

    def decode(self, z: torch.Tensor, vlm_features: torch.Tensor) -> torch.Tensor:
        """
        解码隐变量 z → 动作块，通过跨注意力到 VLM 特征

        参数
        ----
        z : torch.Tensor
            隐变量 [B, d_z]
        vlm_features : torch.Tensor
            VLM 特征 [B, L, D_vlm]

        返回
        ----
        torch.Tensor
            重建的动作 [B, T, D_a]
        """
        B = z.shape[0]

        # 1. 隐变量 → 动作槽
        slots = self.z_to_slots(z).reshape(B, self.seq_len, self.hidden)  # [B, T, H]
        slots = slots + self.dec_pos

        # 2. 解码器：自注意力 + 跨注意力到 VLM + FFN
        for blk in self.dec_blocks:
            slots = blk(slots, vlm_features)
        slots = self.dec_norm(slots)

        # 3. 输出动作
        return self.action_out(slots)                               # [B, T, D_a]

    def forward(
        self, action: torch.Tensor, vlm_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整前向传播

        参数
        ----
        action : torch.Tensor
            动作序列 [B, T, D_a]
        vlm_features : torch.Tensor
            VLM 特征 [B, L, D_vlm]

        返回
        ----
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            z, mu, log_var, recon_action
        """
        mu, log_var, z = self.encode(action, vlm_features)
        recon = self.decode(z, vlm_features)
        return z, mu, log_var, recon


# ─────────────────────────── LatentFlowNet ──────────────────────────────────

class LatentFlowNet(nn.Module):
    """
    隐空间 Flow Matching 速度场网络

    核心功能：
      - 在 VAE 隐空间 [B,d_z] 做 Flow Matching
      - 输入：z_t [B,d_z], t [B], vlm_features [B,L,D_vlm], proprio [B,D_p]
      - 输出：速度 v_t [B,d_z]

    架构：
      - MLP + 残差块
      - 条件：VLM 全局池化 + 本体感知 + 正弦时间嵌入
      - 紧凑设计：d_z=32 << T·D_a=70

    参数
    ----
    latent_dim : int
        隐变量维度（默认 32）
    vlm_hidden : int
        VLM 特征维度（SmolVLM-500M 为 576）
    dim_proprio : int
        本体感知维度（VLABench 为 7）
    hidden : int
        隐藏层维度（默认 512）
    depth : int
        残差块深度（默认 4）
    dim_time : int
        时间嵌入维度（默认 64）
    """

    def __init__(
        self,
        latent_dim: int,
        vlm_hidden: int,
        dim_proprio: int,
        hidden: int = 512,
        depth: int = 4,
        dim_time: int = 64,
    ) -> None:
        super().__init__()
        self.dim_time = dim_time

        # 条件编码器
        self.vlm_proj    = nn.Linear(vlm_hidden, hidden)  # VLM 全局池化
        self.proprio_proj = nn.Linear(dim_proprio, hidden)  # 本体感知
        self.time_proj   = nn.Sequential(  # 时间嵌入
            nn.Linear(dim_time, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

        # 输入投影：z_t || vlm_pool || proprio || time → hidden
        self.input_proj = nn.Linear(latent_dim + hidden * 3, hidden)

        # 残差块
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 2),
                nn.SiLU(),
                nn.Linear(hidden * 2, hidden),
            )
            for _ in range(depth)
        ])

        # 输出层
        self.out_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, latent_dim)

        # 零初始化输出，训练初期速度 ≈ 0（稳定训练开始）
        # 这是一个重要的设计：让模型在训练初期预测接近 0 的速度
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        z_t: torch.Tensor,           # [B, d_z]
        t: torch.Tensor,             # [B]
        vlm_features: torch.Tensor,  # [B, L, D_vlm]
        proprio: torch.Tensor,       # [B, D_p] 或 [B, K, D_p]
    ) -> torch.Tensor:
        """
        参数
        ----
        z_t : torch.Tensor
            带噪隐变量 [B, d_z]
        t : torch.Tensor
            时间步 [B]
        vlm_features : torch.Tensor
            VLM 特征 [B, L, D_vlm]
        proprio : torch.Tensor
            本体感知 [B, D_p] 或 [B, K, D_p]

        返回
        ----
        torch.Tensor
            预测速度 [B, d_z]
        """
        # 全局 VLM 表示
        vlm_pool = self.vlm_proj(vlm_features.mean(dim=1))    # [B, H]

        # 本体感知：如果有历史窗口，取最后一帧
        if proprio.dim() == 3:
            proprio = proprio[:, -1, :]
        prop_emb = self.proprio_proj(proprio)                  # [B, H]

        # 正弦时间嵌入
        t_emb = self.time_proj(_sinusoidal_emb(t, self.dim_time))  # [B, H]

        # 拼接所有输入
        x = self.input_proj(torch.cat([z_t, vlm_pool, prop_emb, t_emb], dim=-1))

        # 残差块
        for blk in self.blocks:
            x = x + blk(x)

        # 输出速度
        return self.output(self.out_norm(x))                   # [B, d_z]


__all__ = ["ActionVAE", "LatentFlowNet"]
