# 双流多视角融合（Dual-Stream Multi-View Fusion）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 SmolVLM 输出的 VLM 特征层面，将静态流（agentview/front）和动态流（wrist）特征分离，通过 Cross-Attention 融合后送入 Action Transformer，提升多视角信息利用效率。

**Architecture:** SmolVLM 统一编码所有视角后，按视角 token 位置索引分离静态/动态特征，用 1 层 Cross-Attention（静态流为 Query，动态流为 Key/Value）融合，融合结果替换原始 vlm_features 送入 SmolVLMActionTransformer。消融实验支持 Add / Concat+Linear / Cross-Attention 三种融合方式。

**Tech Stack:** PyTorch, transformers, 现有 SmolVLMVLA 架构

---

## 文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `models/configuration_smolvlm_vla.py` | 新增 `use_dual_stream`、`dual_stream_fusion` 配置项 |
| 新建 | `models/dual_stream.py` | DualStreamFusion 模块（Add/Concat+Linear/CrossAttn） |
| 修改 | `models/modeling_smolvlm_vla.py` | `forward_vlm_efficient()` 记录视角 token 位置，`forward()` 调用融合模块 |
| 修改 | `models/transformer_smolvlm.py` | `SmolVLMActionTransformer.__init__` 接受可选融合特征 |
| 修改 | `train_smolvlm.py` | 新增 `--use_dual_stream`、`--dual_stream_fusion` 参数 |

---

### Task 1: 新增配置项

**Files:**
- Modify: `models/configuration_smolvlm_vla.py:29-80`

- [ ] **Step 1: 在 `SmolVLMVLAConfig.__init__` 中添加双流配置**

在 `use_adaln: bool = False,` 之后添加：

```python
        # === Dual-Stream Multi-View Fusion ===
        use_dual_stream: bool = False,
        dual_stream_fusion: str = "cross_attn",  # "add" | "concat_linear" | "cross_attn"
```

在 `self.use_adaln = use_adaln` 之后添加：

```python
        self.use_dual_stream = use_dual_stream
        self.dual_stream_fusion = dual_stream_fusion
```

- [ ] **Step 2: Commit**

```bash
git add models/configuration_smolvlm_vla.py
git commit -m "feat: add dual_stream config fields to SmolVLMVLAConfig"
```

---

### Task 2: 新建 DualStreamFusion 模块

**Files:**
- Create: `models/dual_stream.py`

- [ ] **Step 1: 创建文件**

```python
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
            # 无额外参数，直接相加
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
        vlm_features: torch.Tensor,  # [B, T_enc, D]
        num_valid_views: torch.Tensor,  # [B] 每个样本的有效视角数
    ) -> torch.Tensor:
        """
        融合静态流和动态流特征。

        返回与 vlm_features 相同形状的融合特征 [B, T_enc, D]。
        文本 token 部分保持不变，只融合图像 token 部分。
        """
        B, T_enc, D = vlm_features.shape
        n = self.num_patches_per_view

        # 分离图像 token 和文本 token
        # 图像 token 在前，文本 token 在后（见 forward_vlm_efficient 的拼接顺序）
        max_img_tokens = max(num_valid_views).item() * n
        img_tokens = vlm_features[:, :max_img_tokens, :]   # [B, V*n, D]
        text_tokens = vlm_features[:, max_img_tokens:, :]  # [B, T_text, D]

        # 提取静态流和动态流 token
        static_parts = []
        dynamic_parts = []

        for b in range(B):
            n_valid = num_valid_views[b].item()
            # 静态流：取 static_view_indices 中在有效范围内的视角
            s_idx = [i for i in self.static_view_indices if i < n_valid]
            d_idx = [i for i in self.dynamic_view_indices if i < n_valid]

            if not s_idx:
                s_idx = list(range(n_valid))  # fallback：所有视角作为静态流
            if not d_idx:
                d_idx = s_idx[:1]             # fallback：第一个视角作为动态流

            s_tokens = torch.cat([img_tokens[b, i*n:(i+1)*n] for i in s_idx], dim=0)  # [T_s, D]
            d_tokens = torch.cat([img_tokens[b, i*n:(i+1)*n] for i in d_idx], dim=0)  # [T_d, D]
            static_parts.append(s_tokens)
            dynamic_parts.append(d_tokens)

        # Pad 到相同长度以便批处理
        max_s = max(t.shape[0] for t in static_parts)
        max_d = max(t.shape[0] for t in dynamic_parts)

        static_padded = torch.zeros(B, max_s, D, device=vlm_features.device, dtype=vlm_features.dtype)
        dynamic_padded = torch.zeros(B, max_d, D, device=vlm_features.device, dtype=vlm_features.dtype)
        for b in range(B):
            static_padded[b, :static_parts[b].shape[0]] = static_parts[b]
            dynamic_padded[b, :dynamic_parts[b].shape[0]] = dynamic_parts[b]

        # 融合
        if self.fusion_type == "add":
            # 动态流 GAP 后广播相加到静态流
            dynamic_gap = dynamic_padded.mean(dim=1, keepdim=True)  # [B, 1, D]
            fused_static = static_padded + dynamic_gap               # [B, T_s, D]
        elif self.fusion_type == "concat_linear":
            dynamic_gap = dynamic_padded.mean(dim=1, keepdim=True).expand_as(static_padded)
            fused_static = self.norm(self.fusion_linear(
                torch.cat([static_padded, dynamic_gap], dim=-1)
            ))
        else:  # cross_attn
            fused_static = self.cross_attn(static_padded, dynamic_padded)  # [B, T_s, D]

        # 将融合后的静态流特征写回 img_tokens（替换静态视角位置）
        fused_img_tokens = img_tokens.clone()
        for b in range(B):
            n_valid = num_valid_views[b].item()
            s_idx = [i for i in self.static_view_indices if i < n_valid]
            if not s_idx:
                s_idx = list(range(n_valid))
            total_s = len(s_idx) * n
            for j, i in enumerate(s_idx):
                fused_img_tokens[b, i*n:(i+1)*n] = fused_static[b, j*n:(j+1)*n]

        # 拼回文本 token
        return torch.cat([fused_img_tokens, text_tokens], dim=1)


__all__ = ["DualStreamFusion", "CrossAttentionFusion"]
```

- [ ] **Step 2: Commit**

```bash
git add models/dual_stream.py
git commit -m "feat: add DualStreamFusion module (add/concat_linear/cross_attn)"
```

---

### Task 3: 修改 forward_vlm_efficient() 记录视角 token 信息

**Files:**
- Modify: `models/modeling_smolvlm_vla.py:270-321`

- [ ] **Step 1: 修改 forward_vlm_efficient() 返回值，增加 num_valid_views 和 num_patches_per_view**

在 `return {"vlm_features": vlm_features}` 替换为：

```python
        return {
            "vlm_features": vlm_features,
            "num_valid_views": valid_per_sample,          # [B] int tensor
            "num_patches_per_view": num_patches,          # int
        }
```

- [ ] **Step 2: Commit**

```bash
git add models/modeling_smolvlm_vla.py
git commit -m "feat: forward_vlm_efficient returns num_valid_views and num_patches_per_view"
```

---

### Task 4: 在 SmolVLMVLA 中集成 DualStreamFusion

**Files:**
- Modify: `models/modeling_smolvlm_vla.py:50-120`（`__init__` 和 `forward`）

- [ ] **Step 1: 在 `SmolVLMVLA.__init__` 中初始化 DualStreamFusion**

在 `self.transformer = SmolVLMActionTransformer(...)` 之后添加：

```python
        # Dual-stream fusion（可选）
        self.dual_stream_fusion = None
        if config.use_dual_stream:
            from .dual_stream import DualStreamFusion
            # VLM hidden size 从实际模型获取
            vlm_hidden = self.vlm.config.vision_config.hidden_size if hasattr(
                self.vlm.config, 'vision_config') else 576
            # 通过 connector 后的维度
            if hasattr(self.vlm.model, 'connector'):
                vlm_hidden = self.vlm.model.connector.proj.out_features
            elif hasattr(self.vlm.model, 'multi_modal_projector'):
                vlm_hidden = self.vlm.model.multi_modal_projector.proj.out_features
            self.dual_stream_fusion = DualStreamFusion(
                hidden_size=vlm_hidden,
                fusion_type=config.dual_stream_fusion,
            )
            logging.info(f"[SmolVLMVLA] Dual-stream fusion enabled: {config.dual_stream_fusion}")
```

- [ ] **Step 2: 在 `forward()` 中调用融合模块**

在 `enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)` 之后添加：

```python
        # 双流融合（可选）
        if self.dual_stream_fusion is not None:
            enc["vlm_features"] = self.dual_stream_fusion(
                enc["vlm_features"],
                enc["num_valid_views"],
            )
```

同样在 `generate_actions()` 的 `enc = self.forward_vlm_efficient(...)` 之后添加相同代码。

- [ ] **Step 3: Commit**

```bash
git add models/modeling_smolvlm_vla.py
git commit -m "feat: integrate DualStreamFusion into SmolVLMVLA forward and generate_actions"
```

---

### Task 5: 修改训练脚本，添加命令行参数

**Files:**
- Modify: `train_smolvlm.py:154-160`

- [ ] **Step 1: 在 `get_args_parser()` 中添加参数**

在 `--num_views` 参数之后添加：

```python
    parser.add_argument("--use_dual_stream", action="store_true", default=False,
                        help="启用双流多视角融合")
    parser.add_argument("--dual_stream_fusion", type=str, default="cross_attn",
                        choices=["add", "concat_linear", "cross_attn"],
                        help="双流融合方式")
```

- [ ] **Step 2: 在 `main()` 中将参数传入 config**

找到模型初始化的地方（`SmolVLMVLAConfig(...)` 调用），添加：

```python
        use_dual_stream=args.use_dual_stream,
        dual_stream_fusion=args.dual_stream_fusion,
```

- [ ] **Step 3: Commit**

```bash
git add train_smolvlm.py
git commit -m "feat: add --use_dual_stream and --dual_stream_fusion CLI args"
```

---

### Task 6: 更新 train_vlabench_debug.sh 添加双流开关

**Files:**
- Modify: `train_vlabench_debug.sh`

- [ ] **Step 1: 添加双流参数变量**

在 `USE_ADALN=false` 之后添加：

```bash
USE_DUAL_STREAM=true          # 启用双流融合
DUAL_STREAM_FUSION=cross_attn # add | concat_linear | cross_attn
```

- [ ] **Step 2: 在 ARGS 构建中添加**

在 `if [ "${USE_ADALN}" = true ]` 块之后添加：

```bash
if [ "${USE_DUAL_STREAM}" = true ]; then
    ARGS="${ARGS} --use_dual_stream --dual_stream_fusion ${DUAL_STREAM_FUSION}"
fi
```

- [ ] **Step 3: Commit**

```bash
git add train_vlabench_debug.sh
git commit -m "feat: add dual stream flags to debug training script"
```

---

### Task 7: 验证

- [ ] **Step 1: 运行 debug 训练验证无报错**

```bash
CUDA_VISIBLE_DEVICES=2 bash train_vlabench_debug.sh 8 0.1 ./runs/test_dual_stream
```

预期：训练启动，日志中出现 `[SmolVLMVLA] Dual-stream fusion enabled: cross_attn`，loss 正常下降。

- [ ] **Step 2: 验证三种融合方式均可运行**

```bash
# add
CUDA_VISIBLE_DEVICES=2 python train_smolvlm.py \
    --use_dual_stream --dual_stream_fusion add \
    --train_metas_path ./datasets/metas/vlabench_debug_train.json \
    --norm_stats_path ./norm_stats/vlabench_norm.json \
    --action_mode vlabench_joint --num_views 4 \
    --iters 50 --batch_size 2 --output_dir /tmp/test_add

# concat_linear
CUDA_VISIBLE_DEVICES=2 python train_smolvlm.py \
    --use_dual_stream --dual_stream_fusion concat_linear \
    --train_metas_path ./datasets/metas/vlabench_debug_train.json \
    --norm_stats_path ./norm_stats/vlabench_norm.json \
    --action_mode vlabench_joint --num_views 4 \
    --iters 50 --batch_size 2 --output_dir /tmp/test_concat

# cross_attn
CUDA_VISIBLE_DEVICES=2 python train_smolvlm.py \
    --use_dual_stream --dual_stream_fusion cross_attn \
    --train_metas_path ./datasets/metas/vlabench_debug_train.json \
    --norm_stats_path ./norm_stats/vlabench_norm.json \
    --action_mode vlabench_joint --num_views 4 \
    --iters 50 --batch_size 2 --output_dir /tmp/test_cross
```

预期：三种方式均无报错，loss 值合理（< 1.0）。

- [ ] **Step 3: Commit**

```bash
git commit -m "test: verify dual stream fusion runs without errors"
```
