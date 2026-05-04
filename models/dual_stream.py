"""
Dual-Stream Multi-View Fusion Module

将 SmolVLM 输出的 VLM 特征按视角分为静态流和动态流，
通过三种可选方式融合：Add / Concat+Linear / Cross-Attention。

静态流：agentview/front/image_0/image_1（场景语义）
动态流：wrist（末端运动信息）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    单层 Cross-Attention 融合。
    静态流特征作为 Query，动态流特征作为 Key/Value。
    """

    def __init__(self, hidden_size: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        static_feat: torch.Tensor,   # [B, T_s, D]
        dynamic_feat: torch.Tensor,  # [B, T_d, D]
    ) -> torch.Tensor:
        """返回融合后的特征 [B, T_s, D]，维度与 static_feat 相同。"""
        B, T_s, D = static_feat.shape
        T_d = dynamic_feat.shape[1]

        q = self.q_proj(static_feat).reshape(B, T_s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(dynamic_feat).reshape(B, T_d, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(dynamic_feat).reshape(B, T_d, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, T_s, D)
        out = self.out_proj(out)
        # 残差连接 + LayerNorm
        return self.norm(static_feat + out)


class DualStreamFusion(nn.Module):
    """
    双流多视角融合模块。

    将 vlm_features 按视角 token 位置分为静态流和动态流，
    融合后返回与原始 vlm_features 相同形状的特征。

    参数
    ----
    hidden_size : VLM 特征维度（SmolVLM-500M 为 576）
    fusion_type : "add" | "concat_linear" | "cross_attn"
    num_patches_per_view : 每个视角的 patch token 数量
    static_view_indices : 静态流视角索引列表，如 [0, 2, 3]
    dynamic_view_indices : 动态流视角索引列表，如 [1]
    """

    def __init__(
        self,
        hidden_size: int,
        fusion_type: str = "cross_attn",
        num_patches_per_view: int = 64,
        static_view_indices: list[int] | None = None,
        dynamic_view_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.num_patches_per_view = num_patches_per_view
        self.static_view_indices = static_view_indices or [0, 2, 3]   # front/image_0/image_1
        self.dynamic_view_indices = dynamic_view_indices or [1]        # wrist

        if fusion_type == "add":
            pass
        elif fusion_type == "concat_linear":
            self.fusion_linear = nn.Linear(hidden_size * 2, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)
        elif fusion_type == "cross_attn":
            self.cross_attn = CrossAttentionFusion(hidden_size, num_heads=8)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose from add/concat_linear/cross_attn")

    def forward(
        self,
        vlm_features: torch.Tensor,     # [B, T_enc, D]
        num_valid_views: torch.Tensor,  # [B] 每个样本的有效视角数
    ) -> torch.Tensor:
        """
        融合静态流和动态流特征。
        返回与 vlm_features 相同形状的融合特征 [B, T_enc, D]。
        文本 token 部分保持不变，只融合图像 token 部分。
        """
        B, T_enc, D = vlm_features.shape
        n = self.num_patches_per_view

        max_img_tokens = int(max(num_valid_views).item()) * n
        img_tokens = vlm_features[:, :max_img_tokens, :]
        text_tokens = vlm_features[:, max_img_tokens:, :]

        static_parts = []
        dynamic_parts = []

        for b in range(B):
            n_valid = int(num_valid_views[b].item())
            s_idx = [i for i in self.static_view_indices if i < n_valid]
            d_idx = [i for i in self.dynamic_view_indices if i < n_valid]

            if not s_idx:
                s_idx = list(range(n_valid))
            if not d_idx:
                d_idx = s_idx[:1]

            s_tokens = torch.cat([img_tokens[b, i*n:(i+1)*n] for i in s_idx], dim=0)
            d_tokens = torch.cat([img_tokens[b, i*n:(i+1)*n] for i in d_idx], dim=0)
            static_parts.append(s_tokens)
            dynamic_parts.append(d_tokens)

        max_s = max(t.shape[0] for t in static_parts)
        max_d = max(t.shape[0] for t in dynamic_parts)

        static_padded = torch.zeros(B, max_s, D, device=vlm_features.device, dtype=vlm_features.dtype)
        dynamic_padded = torch.zeros(B, max_d, D, device=vlm_features.device, dtype=vlm_features.dtype)
        for b in range(B):
            static_padded[b, :static_parts[b].shape[0]] = static_parts[b]
            dynamic_padded[b, :dynamic_parts[b].shape[0]] = dynamic_parts[b]

        if self.fusion_type == "add":
            dynamic_gap = dynamic_padded.mean(dim=1, keepdim=True)
            fused_static = static_padded + dynamic_gap
        elif self.fusion_type == "concat_linear":
            dynamic_gap = dynamic_padded.mean(dim=1, keepdim=True).expand_as(static_padded)
            fused_static = self.norm(self.fusion_linear(
                torch.cat([static_padded, dynamic_gap], dim=-1)
            ))
        else:  # cross_attn
            fused_static = self.cross_attn(static_padded, dynamic_padded)

        fused_img_tokens = img_tokens.clone()
        for b in range(B):
            n_valid = int(num_valid_views[b].item())
            s_idx = [i for i in self.static_view_indices if i < n_valid]
            if not s_idx:
                s_idx = list(range(n_valid))
            for j, i in enumerate(s_idx):
                fused_img_tokens[b, i*n:(i+1)*n] = fused_static[b, j*n:(j+1)*n]

        return torch.cat([fused_img_tokens, text_tokens], dim=1)


__all__ = ["DualStreamFusion", "CrossAttentionFusion"]
