"""
Dual-Stream Multi-View Fusion Module（双流多视角融合模块）

功能：将 SmolVLM 输出的 VLM 特征按视角分为静态流和动态流，
     通过三种可选方式融合：Add / Concat+Linear / Cross-Attention。

设计思路：
  - 静态流（static stream）：agentview/front/image_0/image_1 —— 提供场景全局语义
  - 动态流（dynamic stream）：wrist（手腕相机） —— 提供末端执行器的运动细节
  - 融合目的：让静态流的场景理解能力获得动态流的精细运动信息补充

调用位置：在 SmolVLMVLA.forward() 中，forward_vlm_efficient() 输出 vlm_features 后，
         如果 config.use_dual_stream=True，则调用本模块对 vlm_features 进行融合。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    单层 Cross-Attention 融合模块。

    原理：静态流特征作为 Query（"我想从动态流中获取什么信息"），
         动态流特征作为 Key/Value（"动态流能提供什么信息"）。
         通过注意力机制，静态流的每个 token 自适应地从动态流中提取相关信息。
    """

    def __init__(self, hidden_size: int, num_heads: int = 8) -> None:
        """
        参数:
            hidden_size: 特征维度，即 VLM text_model 的输出维度（SmolVLM-500M 为 960）
            num_heads: 多头注意力的头数，每个头独立计算注意力再拼接
        """
        super().__init__()
        # 确保 hidden_size 能被 num_heads 整除，否则无法均匀分头
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads                    # 注意力头数
        self.head_dim = hidden_size // num_heads      # 每个头的维度 = 960/8 = 120
        self.scale = self.head_dim ** -0.5            # 缩放因子 1/sqrt(head_dim)，防止点积过大

        # Q/K/V 线性投影层：将输入特征映射到 query/key/value 空间
        self.q_proj = nn.Linear(hidden_size, hidden_size)   # 静态流 → Query
        self.k_proj = nn.Linear(hidden_size, hidden_size)   # 动态流 → Key
        self.v_proj = nn.Linear(hidden_size, hidden_size)   # 动态流 → Value
        self.out_proj = nn.Linear(hidden_size, hidden_size) # 注意力输出的线性变换
        self.norm = nn.LayerNorm(hidden_size)               # 残差连接后的层归一化

    def forward(
        self,
        static_feat: torch.Tensor,   # [B, T_s, D] 静态流特征，T_s = 静态视角数 × patches_per_view
        dynamic_feat: torch.Tensor,  # [B, T_d, D] 动态流特征，T_d = 动态视角数 × patches_per_view
        key_padding_mask: torch.Tensor | None = None,  # [B, T_d] True 表示该位置是 padding，需要屏蔽
    ) -> torch.Tensor:
        """
        执行 cross-attention 融合。

        计算过程：
          1. 静态流 → Q，动态流 → K, V
          2. attention_score = softmax(Q @ K^T / sqrt(d))
          3. output = attention_score @ V
          4. 残差连接：output = LayerNorm(static_feat + out_proj(output))

        返回: [B, T_s, D] 融合后的静态流特征（维度不变）
        """
        B, T_s, D = static_feat.shape   # B=batch, T_s=静态流token数, D=特征维度
        T_d = dynamic_feat.shape[1]     # T_d=动态流token数

        # 线性投影 + reshape 为多头格式: [B, T, D] → [B, num_heads, T, head_dim]
        q = self.q_proj(static_feat).reshape(B, T_s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(dynamic_feat).reshape(B, T_d, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(dynamic_feat).reshape(B, T_d, self.num_heads, self.head_dim).transpose(1, 2)

        # 优先使用 PyTorch 2.0+ 的高效 SDPA 实现（支持 FlashAttention）
        if hasattr(F, "scaled_dot_product_attention"):
            attn_mask = None
            if key_padding_mask is not None:
                # 将 padding mask [B, T_d] 扩展为 [B, num_heads, T_s, T_d]
                # 每个 query token 对所有 key 位置共享同一个 mask
                attn_mask = key_padding_mask[:, None, None, :].expand(
                    B, self.num_heads, T_s, T_d
                )
                # True 位置设为 -10000（softmax 后趋近于 0），实现屏蔽
                attn_mask = attn_mask.to(dtype=static_feat.dtype) * -1e4
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            # 手动实现 attention（兼容旧版 PyTorch）
            attn = (q * self.scale) @ k.transpose(-2, -1)  # [B, heads, T_s, T_d] 注意力分数
            if key_padding_mask is not None:
                # padding 位置填充 -inf，softmax 后变为 0
                attn = attn.masked_fill(
                    key_padding_mask[:, None, None, :], float('-inf')
                )
            attn = attn.softmax(dim=-1)                    # 归一化为概率分布
            attn = self.attn_drop(attn) if hasattr(self, 'attn_drop') else attn
            out = attn @ v                                 # 加权求和 value

        # 多头拼接回原始维度: [B, heads, T_s, head_dim] → [B, T_s, D]
        out = out.transpose(1, 2).reshape(B, T_s, D)
        # 输出投影
        out = self.out_proj(out)
        # 残差连接 + LayerNorm：保留原始静态流信息，叠加从动态流学到的信息
        return self.norm(static_feat + out)


class DualStreamFusion(nn.Module):
    """
    双流多视角融合模块（核心模块）。

    整体流程：
      1. 输入 vlm_features [B, T_enc, D]，其中前面是图像 token，后面是文本 token
         布局：[view0_patches | view1_patches | view2_patches | view3_patches | text_tokens]
      2. 按视角索引将图像 token 分为静态流和动态流
      3. 用选定的融合方式将动态流信息注入静态流
      4. 将融合后的静态流 token 写回原位置，文本 token 保持不变
      5. 输出与输入形状完全相同的 [B, T_enc, D]

    参数
    ----
    hidden_size : VLM 特征维度（SmolVLM-500M 实际为 960）
    fusion_type : 融合方式
        - "add"：动态流均值直接加到静态流（最简单，无额外参数）
        - "concat_linear"：拼接后过线性层压缩（有少量参数）
        - "cross_attn"：跨注意力融合（最强表达力，参数最多）
    num_patches_per_view : 每个视角的 patch token 数量（512/16=32, 32²=1024）
    static_view_indices : 属于静态流的视角索引，默认 [0, 2, 3] 即 front/image_0/image_1
    dynamic_view_indices : 属于动态流的视角索引，默认 [1] 即 wrist
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
        self.fusion_type = fusion_type                                      # 融合方式字符串
        self.num_patches_per_view = num_patches_per_view                    # 每视角 patch 数（默认64，实际1024）
        self.static_view_indices = static_view_indices or [0, 2, 3]         # 静态流视角：front/image_0/image_1
        self.dynamic_view_indices = dynamic_view_indices or [1]              # 动态流视角：wrist

        # 根据融合方式初始化对应的可学习层
        if fusion_type == "add":
            # add 模式不需要额外参数，直接相加
            pass
        elif fusion_type == "concat_linear":
            # 将静态特征和动态均值拼接 [D+D=2D] → 线性层压缩回 [D]
            self.fusion_linear = nn.Linear(hidden_size * 2, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)
        elif fusion_type == "cross_attn":
            # 使用上面定义的 CrossAttentionFusion 模块
            self.cross_attn = CrossAttentionFusion(hidden_size, num_heads=8)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose from add/concat_linear/cross_attn")

    def _get_stream_indices(self, n_valid: int) -> tuple[list[int], list[int]]:
        """
        根据当前样本的有效视角数，返回实际可用的静态/动态视角索引。

        参数:
            n_valid: 该样本有效视角数（如 VLABench 为 4，LIBERO 填充后为 2-3）

        返回:
            (s_idx, d_idx): 静态流索引列表, 动态流索引列表

        逻辑：过滤掉超出有效范围的索引。如果过滤后为空则 fallback：
            - 静态流为空 → 所有视角都算静态流
            - 动态流为空 → 取静态流第一个视角作为动态流
        """
        s_idx = [i for i in self.static_view_indices if i < n_valid]  # 只保留有效范围内的索引
        d_idx = [i for i in self.dynamic_view_indices if i < n_valid]
        if not s_idx:
            s_idx = list(range(n_valid))   # fallback：全部视角作为静态流
        if not d_idx:
            d_idx = s_idx[:1]              # fallback：取第一个静态视角作为动态流
        return s_idx, d_idx

    def forward(
        self,
        vlm_features: torch.Tensor,     # [B, T_enc, D] VLM 输出的完整特征序列
        num_valid_views: torch.Tensor,   # [B] 每个样本的有效视角数（int tensor）
        num_patches_per_view: int | None = None,  # 运行时覆盖 patch 数（由 VLM 动态计算得到）
    ) -> torch.Tensor:
        """
        主前向传播：融合静态流和动态流特征。

        输入 vlm_features 的 token 布局（以 4 视角为例）：
          [view0: 1024 patches | view1: 1024 patches | view2: 1024 patches | view3: 1024 patches | text: ~20 tokens]
           ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ 图像 token 区域 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ →  ← 文本 token →

        返回: [B, T_enc, D] 与输入形状完全相同，文本 token 不变，图像 token 中静态流部分被融合更新
        """
        B, T_enc, D = vlm_features.shape  # B=batch_size, T_enc=总token数, D=特征维度(960)
        # n = 每个视角的 patch token 数量（运行时传入优先，否则用初始化值）
        n = num_patches_per_view if num_patches_per_view is not None else self.num_patches_per_view

        # ── 第1步：分离图像 token 和文本 token ──────────────────────────────────
        # max_img_tokens = batch 中最大有效视角数 × 每视角 patch 数
        max_img_tokens = int(max(num_valid_views).item()) * n
        img_tokens = vlm_features[:, :max_img_tokens, :]    # [B, max_views*n, D] 图像部分
        text_tokens = vlm_features[:, max_img_tokens:, :]   # [B, text_len, D] 文本部分（不参与融合）

        # ── 第2步：按视角索引拆分为静态流和动态流 ──────────────────────────────
        static_parts = []    # 每个样本的静态流 token，形状各异
        dynamic_parts = []   # 每个样本的动态流 token，形状各异

        for b in range(B):
            n_valid = int(num_valid_views[b].item())  # 该样本有效视角数
            s_idx, d_idx = self._get_stream_indices(n_valid)  # 获取该样本的静态/动态索引

            # 从 img_tokens 中按视角索引切片并拼接
            # 例如 s_idx=[0,2,3]，则取 view0、view2、view3 的 patch token 拼接
            s_tokens = torch.cat([img_tokens[b, i*n:(i+1)*n] for i in s_idx], dim=0)  # [len(s_idx)*n, D]
            d_tokens = torch.cat([img_tokens[b, i*n:(i+1)*n] for i in d_idx], dim=0)  # [len(d_idx)*n, D]
            static_parts.append(s_tokens)
            dynamic_parts.append(d_tokens)

        # ── 第3步：padding 对齐（batch 内各样本 token 数可能不同）──────────────
        max_s = max(t.shape[0] for t in static_parts)   # batch 中最长的静态流 token 数
        max_d = max(t.shape[0] for t in dynamic_parts)  # batch 中最长的动态流 token 数

        # 创建 zero-padded 张量，短的样本后面补零
        static_padded = torch.zeros(B, max_s, D, device=vlm_features.device, dtype=vlm_features.dtype)
        dynamic_padded = torch.zeros(B, max_d, D, device=vlm_features.device, dtype=vlm_features.dtype)
        for b in range(B):
            static_padded[b, :static_parts[b].shape[0]] = static_parts[b]
            dynamic_padded[b, :dynamic_parts[b].shape[0]] = dynamic_parts[b]

        # ── 第4步：执行融合 ────────────────────────────────────────────────────
        if self.fusion_type == "add":
            # 最简单的融合：计算动态流所有 token 的均值，广播加到静态流每个 token 上
            # 直觉：让静态流的每个 patch 都获得一份"手腕视角的全局运动摘要"
            dynamic_means = torch.zeros(B, 1, D, device=vlm_features.device, dtype=vlm_features.dtype)
            for b in range(B):
                n_d = dynamic_parts[b].shape[0]  # 该样本动态流的有效 token 数（不含 padding）
                dynamic_means[b, 0] = dynamic_padded[b, :n_d].mean(dim=0)  # 只对有效 token 求均值
            fused_static = static_padded + dynamic_means  # [B, max_s, D] 广播相加

        elif self.fusion_type == "concat_linear":
            # 中等复杂度：将动态流均值与静态流每个 token 拼接，过线性层压缩
            # 直觉：线性层可以学习"如何混合场景信息和运动信息"
            dynamic_means = torch.zeros(B, 1, D, device=vlm_features.device, dtype=vlm_features.dtype)
            for b in range(B):
                n_d = dynamic_parts[b].shape[0]
                dynamic_means[b, 0] = dynamic_padded[b, :n_d].mean(dim=0)
            # expand 将 [B,1,D] 扩展为 [B, max_s, D]，与 static_padded 形状一致
            dynamic_gap = dynamic_means.expand_as(static_padded)
            # 拼接: [B, max_s, 2D] → 线性层 → [B, max_s, D] → LayerNorm
            fused_static = self.norm(self.fusion_linear(
                torch.cat([static_padded, dynamic_gap], dim=-1)  # 在特征维度拼接
            ))

        else:  # cross_attn
            # 最强表达力：静态流的每个 token 通过注意力机制，自适应地从动态流中提取信息
            # 直觉：不同位置的静态 patch 可以关注动态流中不同的 patch（局部对应）
            # 构造 dynamic padding mask：True = 该位置是 padding，需要在 attention 中屏蔽
            dynamic_mask = torch.zeros(B, max_d, dtype=torch.bool, device=vlm_features.device)
            for b in range(B):
                n_d = dynamic_parts[b].shape[0]
                if n_d < max_d:
                    dynamic_mask[b, n_d:] = True  # padding 位置标记为 True
            fused_static = self.cross_attn(static_padded, dynamic_padded, key_padding_mask=dynamic_mask)

        # ── 第5步：将融合后的静态流 token 写回原始位置 ─────────────────────────
        fused_img_tokens = img_tokens.clone()  # 复制一份，避免修改原始张量
        for b in range(B):
            n_valid = int(num_valid_views[b].item())
            s_idx, _ = self._get_stream_indices(n_valid)
            for j, i in enumerate(s_idx):
                # j = 在 fused_static 中的第几个视角块
                # i = 在原始 img_tokens 中的视角索引
                src_start = j * n       # fused_static 中该视角块的起始位置
                src_end = src_start + n  # fused_static 中该视角块的结束位置
                if src_end <= static_parts[b].shape[0]:
                    # 将融合后的 token 写回对应视角的原始位置
                    fused_img_tokens[b, i*n:(i+1)*n] = fused_static[b, src_start:src_end]

        # ── 第6步：拼接图像 token 和文本 token，恢复原始形状 ──────────────────
        return torch.cat([fused_img_tokens, text_tokens], dim=1)  # [B, T_enc, D]


__all__ = ["DualStreamFusion", "CrossAttentionFusion"]


if __name__ == "__main__":
    """
    直接运行调试：从真实 VLABench TFRecord 读取数据，
    模拟 forward_vlm_efficient 的输出格式，测试 DualStreamFusion。

    用法：
        cd /home/sapi/hyj/code/newSimVLA
        python -m models.dual_stream
    或：
        python models/dual_stream.py
    """
    import sys
    import os
    import io
    import numpy as np
    from PIL import Image

    # ── 路径配置 ──────────────────────────────────────────────────────────────
    DATA_DIR = "/home/sapi/hyj/vlabench/data/1.0.0"
    SHARD    = "primitive-train.tfrecord-00000-of-00512"
    SHARD_PATH = os.path.join(DATA_DIR, SHARD)

    # SmolVLM-500M-Instruct 实际参数（从 config 读取）
    VLM_HIDDEN   = 960    # text_config.hidden_size
    NUM_PATCHES  = 64     # 调试用缩小值（真实为 1024=(512/16)^2，但 GPU 显存可能不够）
    NUM_VIEWS    = 4      # front / wrist / image_0 / image_1
    TEXT_LEN     = 20     # 模拟文本 token 长度
    BATCH_SIZE   = 2      # 取前 2 帧做 batch

    print(f"[dual_stream debug] 读取 TFRecord: {SHARD_PATH}")

    # ── 读取真实图像数据 ───────────────────────────────────────────────────────
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: tensorflow 未安装，请先 pip install tensorflow")
        sys.exit(1)

    frames = []  # 每帧：list of 4 PIL Images
    dataset = tf.data.TFRecordDataset(SHARD_PATH)
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        feat = example.features.feature

        front_bytes  = list(feat["steps/observation/front"].bytes_list.value)
        wrist_bytes  = list(feat["steps/observation/wrist"].bytes_list.value) #这个视角 是动态
        image0_bytes = list(feat["steps/observation/image_0"].bytes_list.value)
        image1_bytes = list(feat["steps/observation/image_1"].bytes_list.value)
        T = len(front_bytes) # 在这帧

        for t in range(min(T, BATCH_SIZE - len(frames))):
            imgs = []
            for buf in [front_bytes[t], wrist_bytes[t], image0_bytes[t], image1_bytes[t]]:
                imgs.append(Image.open(io.BytesIO(buf)).convert("RGB"))
            frames.append(imgs)
            if len(frames) >= BATCH_SIZE:
                break
        if len(frames) >= BATCH_SIZE:
            break

    print(f"[dual_stream debug] 读取到 {len(frames)} 帧，每帧 {len(frames[0])} 个视角")
    print(f"  图像尺寸示例: {frames[0][0].size}")  # (W, H)

    # ── 模拟 forward_vlm_efficient 的输出 ─────────────────────────────────────
    # 真实输出：vlm_features [B, V*NUM_PATCHES + TEXT_LEN, VLM_HIDDEN]
    # 这里用随机张量模拟（保持形状与真实一致）
    # 强制使用 CPU 避免 GPU OOM（其他进程可能占用显存）
    device = torch.device("cpu")
    print(f"[dual_stream debug] 使用设备: {device}（调试模式强制 CPU）")

    T_enc = NUM_VIEWS * NUM_PATCHES + TEXT_LEN
    vlm_features = torch.randn(BATCH_SIZE, T_enc, VLM_HIDDEN, device=device)
    num_valid_views = torch.tensor([4, 3], device=device)  # 第1帧4视角，第2帧3视角

    print(f"\n[输入] vlm_features shape: {vlm_features.shape}")
    print(f"[输入] num_valid_views: {num_valid_views.tolist()}")
    print(f"[输入] num_patches_per_view: {NUM_PATCHES}")

    # ── 测试三种融合方式 ───────────────────────────────────────────────────────
    for fusion_type in ["add", "concat_linear", "cross_attn"]:
        model = DualStreamFusion(
            hidden_size=VLM_HIDDEN,
            fusion_type=fusion_type,
            num_patches_per_view=NUM_PATCHES,
            static_view_indices=[0, 2, 3],   # front / image_0 / image_1
            dynamic_view_indices=[1],         # wrist
        ).to(device)

        out = model(vlm_features, num_valid_views, num_patches_per_view=NUM_PATCHES)

        shape_ok = out.shape == vlm_features.shape
        # 文本 token 部分应保持不变
        text_unchanged = torch.allclose(
            out[:, NUM_VIEWS * NUM_PATCHES:, :],
            vlm_features[:, NUM_VIEWS * NUM_PATCHES:, :],
        )
        print(f"\n[{fusion_type}]")
        print(f"  输出 shape: {out.shape}  ✓" if shape_ok else f"  输出 shape: {out.shape}  ✗ 期望 {vlm_features.shape}")
        print(f"  文本 token 不变: {'✓' if text_unchanged else '✗'}")
        print(f"  图像 token 已融合: {'✓' if not torch.allclose(out[:, :NUM_VIEWS*NUM_PATCHES], vlm_features[:, :NUM_VIEWS*NUM_PATCHES]) else '(未变化，请检查)'}")

    print("\n[dual_stream debug] 全部测试完成")
