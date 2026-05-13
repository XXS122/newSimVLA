"""
双流多视角融合模块

核心功能：
  - 将 SmolVLM 输出的 VLM 特征按视角分为静态流和动态流
  - 通过三种可选方式融合：Add / Concat+Linear / Cross-Attention
  - 支持运动引导跨视角注意力（Motion-Guided Cross-Attention）

视角分类：
  - 静态流：front/image_0/image_1（场景语义信息）
  - 动态流：wrist（末端运动信息）

创新点：
  - MotionCNN：帧差分图 → 轻量 CNN → 运动激活图 → 注入 attention bias
  - 让静态视角自动聚焦到 wrist 图中正在运动的区域
  - 参考：DeltaCNN (CVPR 2022, arXiv:2203.03996), MotionDeltaCNN (ICCV 2023, arXiv:2210.09887)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionCNN(nn.Module):
    """
    轻量结构化剪枝 CNN，将帧差分图编码为 patch 级运动激活图

    核心思想：
      - 输入：差分图 Δ = frame_t - frame_{t-1}，形状 [B, 3, H, W]
      - 输出：运动激活分数 M ∈ [0,1]，形状 [B, num_patches]
      - 架构：3 层步长卷积（通道数为标准 CNN 的 1/4，结构化剪枝）
      - 参数量：约 3M，推理额外显存 < 100MB

    动机：
      - DeltaCNN (CVPR 2022) 用帧差分省计算
      - 本模块用帧差分生成运动语义图引导 cross-attention 聚焦运动活跃区域
      - 两者目标完全不同
    """

    def __init__(self, num_patches: int, image_size: int = 384) -> None:
        """
        参数
        ----
        num_patches : int
            视觉 patch 数量（由 SmolVLM 视觉编码器决定）
        image_size : int
            输入图像尺寸（默认 384）
        """
        super().__init__()
        self.num_patches = num_patches
        grid_size = int(math.isqrt(num_patches))  # 384→24, 576→24

        # 3 层步长卷积编码器
        # 通道数为标准 CNN 的 1/4（结构化剪枝），减少参数量
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # H/8
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),            # grid×grid
        )

        # 线性层 + Sigmoid 生成运动激活分数
        self.to_score = nn.Sequential(
            nn.Flatten(),                                            # [B, 32*P]
            nn.Linear(32 * num_patches, num_patches),
            nn.Sigmoid(),  # 输出 ∈ [0, 1]
        )

        # 零初始化线性层输出，训练初期等效均匀激活，梯度驱动后逐步激活
        # 这是一个重要的设计：让模型在训练初期不依赖运动信息，随训练逐渐学会利用
        nn.init.zeros_(self.to_score[1].weight)
        nn.init.zeros_(self.to_score[1].bias)

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        diff : [B, 3, H, W]  帧差分图（pixel_t - pixel_{t-1}）

        返回
        ----
        M : [B, num_patches]  每个 patch 的运动激活分数 ∈ [0, 1]
        """
        feat = self.encoder(diff)   # [B, 32, grid, grid]
        return self.to_score(feat)  # [B, num_patches]


class CrossAttentionFusion(nn.Module):
    """
    单层 Cross-Attention 融合

    核心设计：
      - 静态流特征作为 Query，动态流特征作为 Key/Value
      - 支持可选的运动引导 attention bias（motion_map）
      - M [B, T_d] 加到 attention logits 上，让静态 patch 更关注运动活跃的 wrist patch
      - motion_bias_scale 从 0 初始化，训练初期等效原始 cross-attention

    为什么用零初始化？
      - 让模型在训练初期不依赖运动偏置，随训练逐渐学会利用
      - 避免随机初始化的运动偏置干扰正常的 attention 学习
    """

    def __init__(self, hidden_size: int, num_heads: int = 8) -> None:
        """
        参数
        ----
        hidden_size : int
            特征维度
        num_heads : int
            注意力头数
        """
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Q/K/V 投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

        # 可学习的运动 bias 缩放因子
        # 零初始化 → 训练初期不影响原始 attention
        # 这是运动引导注意力的核心：让模型学会何时以及如何利用运动信息
        self.motion_bias_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        static_feat: torch.Tensor,              # [B, T_s, D] 静态流特征
        dynamic_feat: torch.Tensor,             # [B, T_d, D] 动态流特征
        key_padding_mask: torch.Tensor | None = None,   # [B, T_d] True=屏蔽
        motion_map: torch.Tensor | None = None,         # [B, T_d] 运动激活分数
    ) -> torch.Tensor:
        """
        返回融合后的特征 [B, T_s, D]，维度与 static_feat 相同

        参数
        ----
        static_feat : torch.Tensor
            静态流特征 [B, T_s, D]（Query）
        dynamic_feat : torch.Tensor
            动态流特征 [B, T_d, D]（Key/Value）
        key_padding_mask : torch.Tensor | None
            动态流的 padding 掩码 [B, T_d]（True = 屏蔽）
        motion_map : torch.Tensor | None
            运动激活图 [B, T_d]（可选，用于运动引导注意力）
        """
        B, T_s, D = static_feat.shape
        T_d = dynamic_feat.shape[1]

        # 计算 Q/K/V
        q = self.q_proj(static_feat).reshape(B, T_s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(dynamic_feat).reshape(B, T_d, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(dynamic_feat).reshape(B, T_d, self.num_heads, self.head_dim).transpose(1, 2)

        # 手动计算 attention logits（需注入 motion bias）
        attn = (q * self.scale) @ k.transpose(-2, -1)  # [B, H, T_s, T_d]

        # 注入运动激活 bias：M [B, T_d] → [B, 1, 1, T_d]，广播到所有头和查询位置
        # 这是运动引导注意力的核心：让静态视角更关注 wrist 中运动活跃的区域
        if motion_map is not None:
            # 裁剪到 T_d（motion_map 可能因 padding 与 T_d 不等）
            m = motion_map[:, :T_d]
            attn = attn + self.motion_bias_scale * m[:, None, None, :]

        # 应用 padding 掩码
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], float('-inf')
            )

        # Softmax 归一化
        attn = attn.softmax(dim=-1)

        # 加权求和
        out = attn @ v

        # 重塑并投影
        out = out.transpose(1, 2).reshape(B, T_s, D)
        out = self.out_proj(out)

        # 残差连接 + LayerNorm
        return self.norm(static_feat + out)


class DualStreamFusion(nn.Module):
    """
    双流多视角融合模块

    核心功能：
      - 将 vlm_features 按视角 token 位置分为静态流和动态流
      - 融合后返回与原始 vlm_features 相同形状的特征
      - 文本 token 部分保持不变，只融合图像 token 部分

    视角分类：
      - 静态流：front/image_0/image_1（场景语义信息）
      - 动态流：wrist（末端运动信息）

    融合策略：
      - add：动态流均值加到静态流
      - concat_linear：静态流和动态流均值拼接后线性投影
      - cross_attn：静态流 Query，动态流 Key/Value 的跨注意力融合（默认）

    参数
    ----
    hidden_size : int
        VLM 特征维度（SmolVLM-500M 为 576）
    fusion_type : str
        融合策略："add" | "concat_linear" | "cross_attn"
    num_patches_per_view : int
        每个视角的 patch token 数量
    static_view_indices : list[int] | None
        静态流视角索引列表，如 [0, 2, 3]
    dynamic_view_indices : list[int] | None
        动态流视角索引列表，如 [1]
    use_missing_token : bool
        是否使用可学习缺失视角 token（参考 MAE arxiv:2111.06377）
    """

    def __init__(
        self,
        hidden_size: int,
        fusion_type: str = "cross_attn",
        num_patches_per_view: int = 64,
        static_view_indices: list[int] | None = None,
        dynamic_view_indices: list[int] | None = None,
        use_missing_token: bool = False,
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.num_patches_per_view = num_patches_per_view
        self.static_view_indices = static_view_indices or [0, 2, 3]   # front/image_0/image_1
        self.dynamic_view_indices = dynamic_view_indices or [1]        # wrist

        # 根据融合类型初始化对应模块
        if fusion_type == "add":
            pass  # 直接相加，无需额外模块
        elif fusion_type == "concat_linear":
            self.fusion_linear = nn.Linear(hidden_size * 2, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)
        elif fusion_type == "cross_attn":
            self.cross_attn = CrossAttentionFusion(hidden_size, num_heads=8)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose from add/concat_linear/cross_attn")

        # 可学习缺失视角 token（参考 MAE arxiv:2111.06377）
        # 零初始化 → 训练初期等效原零填充，梯度驱动后逐步激活
        # 用于处理某些视角缺失的情况（如 wrist 视角偶尔不可用）
        if use_missing_token:
            self.missing_token = nn.Parameter(torch.zeros(1, hidden_size))
        else:
            self.missing_token = None

    def _get_stream_indices(self, n_valid: int) -> tuple[list[int], list[int]]:
        """
        根据有效视角数返回 (s_idx, d_idx)

        参数
        ----
        n_valid : int
            当前样本的有效视角数

        返回
        ----
        tuple[list[int], list[int]]
            (静态流索引, 动态流索引)
        """
        s_idx = [i for i in self.static_view_indices if i < n_valid]
        d_idx = [i for i in self.dynamic_view_indices if i < n_valid]

        # 如果没有有效视角，使用所有视角
        if not s_idx:
            s_idx = list(range(n_valid))
        if not d_idx:
            d_idx = s_idx[:1]  # 至少需要一个动态视角

        return s_idx, d_idx

    def forward(
        self,
        vlm_features: torch.Tensor,              # [B, T_enc, D] VLM 特征
        num_valid_views: torch.Tensor,           # [B] 每个样本的有效视角数
        num_patches_per_view: int | None = None, # 覆盖初始化时的默认值
        motion_map: torch.Tensor | None = None,  # [B, P_wrist] 运动激活图（可选）
    ) -> torch.Tensor:
        """
        融合静态流和动态流特征

        参数
        ----
        vlm_features : torch.Tensor
            VLM 特征 [B, T_enc, D]
        num_valid_views : torch.Tensor
            每个样本的有效视角数 [B]
        num_patches_per_view : int | None
            每个视角的 patch 数量（覆盖初始化值）
        motion_map : torch.Tensor | None
            运动激活图 [B, P_wrist]（用于运动引导注意力）

        返回
        ----
        torch.Tensor
            融合后的特征 [B, T_enc, D]（与输入形状相同）
        """
        B, T_enc, D = vlm_features.shape
        n = num_patches_per_view if num_patches_per_view is not None else self.num_patches_per_view

        # 分离图像 token 和文本 token
        # 图像 token 在前，文本 token 在后
        max_img_tokens = int(max(num_valid_views).item()) * n
        img_tokens = vlm_features[:, :max_img_tokens, :]
        text_tokens = vlm_features[:, max_img_tokens:, :]

        # 为每个样本分离静态流和动态流
        static_parts = []
        dynamic_parts = []

        for b in range(B):
            n_valid = int(num_valid_views[b].item())
            s_idx, d_idx = self._get_stream_indices(n_valid)

            # 提取静态流 token（front/image_0/image_1）
            s_tokens = torch.cat([img_tokens[b, i*n:(i+1)*n] for i in s_idx], dim=0)
            # 提取动态流 token（wrist）
            d_tokens = torch.cat([img_tokens[b, i*n:(i+1)*n] for i in d_idx], dim=0)

            static_parts.append(s_tokens)
            dynamic_parts.append(d_tokens)

        # 计算最大长度（用于 padding）
        max_s = max(t.shape[0] for t in static_parts)
        max_d = max(t.shape[0] for t in dynamic_parts)

        # 缺失视角处：用可学习 missing_token 填充（若未启用则退回零填充）
        # 参考 MAE (arxiv:2111.06377) 的 mask token 思路
        if self.missing_token is not None:
            pad_vec = self.missing_token.to(dtype=vlm_features.dtype)  # [1, D]
            static_padded  = pad_vec.expand(B, max_s, D).clone()
            dynamic_padded = pad_vec.expand(B, max_d, D).clone()
        else:
            static_padded  = torch.zeros(B, max_s, D, device=vlm_features.device, dtype=vlm_features.dtype)
            dynamic_padded = torch.zeros(B, max_d, D, device=vlm_features.device, dtype=vlm_features.dtype)

        # 填充到统一长度
        for b in range(B):
            static_padded[b, :static_parts[b].shape[0]]  = static_parts[b]
            dynamic_padded[b, :dynamic_parts[b].shape[0]] = dynamic_parts[b]

        # 根据融合类型执行融合
        if self.fusion_type == "add":
            # 对每个样本单独计算有效 dynamic token 均值，避免 padding 稀释
            dynamic_means = torch.zeros(B, 1, D, device=vlm_features.device, dtype=vlm_features.dtype)
            for b in range(B):
                n_d = dynamic_parts[b].shape[0]
                dynamic_means[b, 0] = dynamic_padded[b, :n_d].mean(dim=0)
            fused_static = static_padded + dynamic_means

        elif self.fusion_type == "concat_linear":
            # 对每个样本单独计算有效 dynamic token 均值，避免 padding 稀释
            dynamic_means = torch.zeros(B, 1, D, device=vlm_features.device, dtype=vlm_features.dtype)
            for b in range(B):
                n_d = dynamic_parts[b].shape[0]
                dynamic_means[b, 0] = dynamic_padded[b, :n_d].mean(dim=0)
            dynamic_gap = dynamic_means.expand_as(static_padded)
            # 拼接后线性投影
            fused_static = self.norm(self.fusion_linear(
                torch.cat([static_padded, dynamic_gap], dim=-1)
            ))

        else:  # cross_attn
            # 构造 dynamic padding mask：True = 屏蔽 padding 位置
            dynamic_mask = torch.zeros(B, max_d, dtype=torch.bool, device=vlm_features.device)
            for b in range(B):
                n_d = dynamic_parts[b].shape[0]
                if n_d < max_d:
                    dynamic_mask[b, n_d:] = True

            # 跨注意力融合（静态 Query，动态 Key/Value）
            # motion_map 注入运动偏置，让静态视角更关注运动活跃区域
            fused_static = self.cross_attn(
                static_padded, dynamic_padded,
                key_padding_mask=dynamic_mask,
                motion_map=motion_map,
            )

        # 将融合结果写回原始位置
        fused_img_tokens = img_tokens.clone()
        for b in range(B):
            n_valid = int(num_valid_views[b].item())
            s_idx, _ = self._get_stream_indices(n_valid)
            for j, i in enumerate(s_idx):
                # 只写入有效 token，不写 padding 部分
                src_start = j * n
                src_end = src_start + n
                if src_end <= static_parts[b].shape[0]:
                    fused_img_tokens[b, i*n:(i+1)*n] = fused_static[b, src_start:src_end]

        # 拼接回文本 token（文本 token 保持不变）
        return torch.cat([fused_img_tokens, text_tokens], dim=1)


__all__ = ["DualStreamFusion", "CrossAttentionFusion", "MotionCNN"]
