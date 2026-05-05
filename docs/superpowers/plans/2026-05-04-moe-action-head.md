# MoE 动作头（Sparse MoE Action Head）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 SmolVLMActionTransformer 中的密集 FFN 层替换为稀疏 MoE FFN（4 专家 Top-2），使用任务 CLS token 作为路由输入，降低激活参数量和显存占用，同时提升多任务处理能力。

**Architecture:** 新建 `models/moe_transformer.py` 实现 MoEFFN 和 Router；修改 `transformer_smolvlm.py` 的 TransformerBlock/DiTBlock 支持 MoE 替换；训练损失加入负载均衡辅助损失（α=0.01）。所有改动通过 `use_moe` 配置项开关，向后兼容。

**Tech Stack:** PyTorch, 现有 SmolVLMActionTransformer 架构

---

## 文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `models/moe_transformer.py` | MoEFFN、Router、负载均衡损失 |
| 修改 | `models/transformer_smolvlm.py` | TransformerBlock/DiTBlock 支持 MoE FFN |
| 修改 | `models/configuration_smolvlm_vla.py` | 新增 `use_moe`、`num_experts`、`top_k` 配置项 |
| 修改 | `models/modeling_smolvlm_vla.py` | `forward()` 收集并返回负载均衡损失 |
| 修改 | `train_smolvlm.py` | 新增 `--use_moe`、`--num_experts`、`--moe_aux_loss_weight` 参数 |

---

### Task 1: 新增 MoE 配置项

**Files:**
- Modify: `models/configuration_smolvlm_vla.py:29-80`

- [ ] **Step 1: 添加 MoE 配置字段**

在 `use_adaln: bool = False,` 之后添加：

```python
        # === Sparse MoE Action Head ===
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2,
        moe_aux_loss_weight: float = 0.01,
```

在 `self.use_adaln = use_adaln` 之后添加：

```python
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_aux_loss_weight = moe_aux_loss_weight
```

- [ ] **Step 2: Commit**

```bash
git add models/configuration_smolvlm_vla.py
git commit -m "feat: add MoE config fields to SmolVLMVLAConfig"
```

---

### Task 2: 新建 MoE 模块

**Files:**
- Create: `models/moe_transformer.py`

- [ ] **Step 1: 创建文件**

```python
"""
Sparse Mixture-of-Experts FFN for SmolVLM Action Transformer

将 TransformerBlock/DiTBlock 中的密集 FFN 替换为稀疏 MoE FFN。
每步激活 Top-K 专家，降低计算量和显存占用。
使用任务感知路由（task-CLS token 作为路由输入）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_smolvlm import Mlp


class Router(nn.Module):
    """
    Top-K 稀疏路由器。
    输入路由 token（任务 CLS 特征），输出每个专家的权重。
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数
        ----
        x : [B, D] 路由输入（任务 CLS token 的平均池化）

        返回
        ----
        weights      : [B, top_k] 归一化后的专家权重
        indices      : [B, top_k] 选中的专家索引
        router_probs : [B, num_experts] 完整路由概率（用于负载均衡损失）
        """
        logits = self.gate(x)                          # [B, num_experts]
        router_probs = logits.softmax(dim=-1)          # [B, num_experts]
        weights, indices = router_probs.topk(self.top_k, dim=-1)  # [B, top_k]
        weights = weights / weights.sum(dim=-1, keepdim=True)     # 归一化
        return weights, indices, router_probs


class MoEFFN(nn.Module):
    """
    稀疏 MoE FFN，替换 TransformerBlock/DiTBlock 中的 Mlp。

    每步只激活 Top-K 专家，其余专家不参与计算。
    路由输入为任务 CLS token（vlm_features 的平均池化），
    而非动作 token 本身，实现任务感知路由。
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        hidden_features = int(hidden_size * mlp_ratio)

        self.experts = nn.ModuleList([
            Mlp(in_features=hidden_size, hidden_features=hidden_features)
            for _ in range(num_experts)
        ])
        self.router = Router(hidden_size, num_experts, top_k)

    def forward(
        self,
        x: torch.Tensor,          # [B, T, D] 动作 token 序列
        task_feat: torch.Tensor,  # [B, D] 任务特征（路由输入）
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回
        ----
        output       : [B, T, D] MoE FFN 输出
        router_probs : [B, num_experts] 路由概率（用于负载均衡损失）
        """
        B, T, D = x.shape
        weights, indices, router_probs = self.router(task_feat)  # [B, top_k], [B, top_k], [B, E]

        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[:, k]    # [B]
            w = weights[:, k]             # [B]
            # 对每个样本调用对应专家
            for e in range(self.num_experts):
                mask = (expert_idx == e)  # [B] bool
                if not mask.any():
                    continue
                expert_out = self.experts[e](x[mask])  # [n_mask, T, D]
                output[mask] += w[mask].view(-1, 1, 1) * expert_out

        return output, router_probs


def load_balance_loss(router_probs: torch.Tensor) -> torch.Tensor:
    """
    负载均衡辅助损失，防止所有 token 路由到同一专家。

    L_aux = num_experts × Σ_i (f_i × P_i)
    其中 f_i = 路由到专家 i 的 token 比例，P_i = 平均路由概率。

    参数
    ----
    router_probs : [B, num_experts] 或 list of [B, num_experts]
    """
    if isinstance(router_probs, list):
        return torch.stack([load_balance_loss(p) for p in router_probs]).mean()

    num_experts = router_probs.shape[-1]
    # f_i：每个专家被选中的比例（用 router_probs 近似）
    f = router_probs.mean(dim=0)   # [num_experts]
    P = router_probs.mean(dim=0)   # [num_experts]
    return num_experts * (f * P).sum()


__all__ = ["MoEFFN", "Router", "load_balance_loss"]
```

- [ ] **Step 2: Commit**

```bash
git add models/moe_transformer.py
git commit -m "feat: add MoEFFN, Router, load_balance_loss modules"
```

---

### Task 3: 修改 TransformerBlock 支持 MoE

**Files:**
- Modify: `models/transformer_smolvlm.py:162-179`

- [ ] **Step 1: 修改 TransformerBlock**

将原始 `TransformerBlock` 替换为：

```python
class TransformerBlock(nn.Module):
    """Standard Transformer block (pre-LN)，支持可选 MoE FFN。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.use_moe = use_moe
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        if use_moe:
            from .moe_transformer import MoEFFN
            self.mlp = MoEFFN(hidden_size, mlp_ratio=mlp_ratio,
                              num_experts=num_experts, top_k=top_k)
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                drop=0.1,
            )

    def forward(
        self,
        x: torch.Tensor,
        task_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        返回 (output, router_probs_or_None)
        """
        x = x + self.attn(self.norm1(x))
        if self.use_moe:
            assert task_feat is not None, "task_feat required for MoE block"
            mlp_out, router_probs = self.mlp(self.norm2(x), task_feat)
            x = x + mlp_out
            return x, router_probs
        else:
            x = x + self.mlp(self.norm2(x))
            return x, None
```

- [ ] **Step 2: 修改 DiTBlock 支持 MoE**

将原始 `DiTBlock` 的 `self.mlp` 初始化部分替换为：

```python
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_moe = use_moe
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        if use_moe:
            from .moe_transformer import MoEFFN
            self.mlp = MoEFFN(hidden_size, mlp_ratio=mlp_ratio,
                              num_experts=num_experts, top_k=top_k)
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                drop=0.1,
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
```

DiTBlock 的 `forward` 方法中，将 `self.mlp` 调用改为：

```python
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        if self.use_moe:
            mlp_out, router_probs = self.mlp(x_norm, c)  # c 作为 task_feat
            x = x + gate_mlp.unsqueeze(1) * mlp_out
            return x, router_probs
        else:
            x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
            return x, None
```

- [ ] **Step 3: Commit**

```bash
git add models/transformer_smolvlm.py
git commit -m "feat: TransformerBlock and DiTBlock support optional MoE FFN"
```

---

### Task 4: 修改 SmolVLMActionTransformer 传递 task_feat

**Files:**
- Modify: `models/transformer_smolvlm.py:254-443`

- [ ] **Step 1: 修改 `__init__` 接受 MoE 参数**

在 `SmolVLMActionTransformer.__init__` 的参数列表中添加：

```python
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2,
```

将 blocks 初始化改为传入 MoE 参数：

```python
        # Concat 模式
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                             use_moe=use_moe, num_experts=num_experts, top_k=top_k)
            for _ in range(depth)
        ])
        # AdaLN 模式同理
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     use_moe=use_moe, num_experts=num_experts, top_k=top_k)
            for _ in range(depth)
        ])
```

- [ ] **Step 2: 修改 `_forward_concat` 传递 task_feat 并收集 router_probs**

将 Transformer backbone 循环改为：

```python
        # task_feat：vlm_features 的全局平均池化，作为路由输入
        task_feat = vlm_features.mean(dim=1)  # [B, D_vlm] → 需要投影到 hidden_size
        # 注意：vlm_features 维度是 vlm_hidden_size，需要先投影
        task_feat_proj = self.vlm_proj(vlm_features).mean(dim=1)  # [B, H]

        all_router_probs = []
        for block in self.blocks:
            x, router_probs = block(x, task_feat=task_feat_proj)
            if router_probs is not None:
                all_router_probs.append(router_probs)

        self._last_router_probs = all_router_probs  # 供 modeling_smolvlm_vla.py 读取
```

AdaLN 模式的 `_forward_adaln` 同理，用 `c`（条件向量）作为 task_feat。

- [ ] **Step 3: Commit**

```bash
git add models/transformer_smolvlm.py
git commit -m "feat: SmolVLMActionTransformer passes task_feat to MoE blocks, collects router_probs"
```

---

### Task 5: 在训练损失中加入负载均衡损失

**Files:**
- Modify: `models/modeling_smolvlm_vla.py:324-384`

- [ ] **Step 1: 修改 `forward()` 收集并返回负载均衡损失**

在 `return {"velocity_loss": velocity_loss}` 之前添加：

```python
        loss_dict = {"velocity_loss": velocity_loss}

        # MoE 负载均衡损失
        if (hasattr(self.transformer, '_last_router_probs') and
                self.transformer._last_router_probs and
                self.config.moe_aux_loss_weight > 0):
            from .moe_transformer import load_balance_loss
            aux_loss = load_balance_loss(self.transformer._last_router_probs)
            loss_dict["moe_aux_loss"] = self.config.moe_aux_loss_weight * aux_loss

        return loss_dict
```

将原来的 `return {"velocity_loss": velocity_loss}` 删除。

- [ ] **Step 2: Commit**

```bash
git add models/modeling_smolvlm_vla.py
git commit -m "feat: add MoE load balance loss to training forward pass"
```

---

### Task 6: 修改训练脚本

**Files:**
- Modify: `train_smolvlm.py`

- [ ] **Step 1: 添加 CLI 参数**

在 `--dual_stream_fusion` 之后添加：

```python
    parser.add_argument("--use_moe", action="store_true", default=False,
                        help="启用稀疏 MoE 动作头")
    parser.add_argument("--num_experts", type=int, default=4,
                        help="MoE 专家数量")
    parser.add_argument("--moe_aux_loss_weight", type=float, default=0.01,
                        help="MoE 负载均衡损失权重")
```

- [ ] **Step 2: 将参数传入 config**

```python
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        moe_aux_loss_weight=args.moe_aux_loss_weight,
```

- [ ] **Step 3: 在训练循环中处理多损失**

找到 `loss = sum(loss_dict.values())` 确认已正确求和（现有代码已支持多损失）。

- [ ] **Step 4: Commit**

```bash
git add train_smolvlm.py
git commit -m "feat: add --use_moe, --num_experts, --moe_aux_loss_weight CLI args"
```

---

### Task 7: 验证

- [ ] **Step 1: 运行 50 步验证无报错**

```bash
CUDA_VISIBLE_DEVICES=2 python train_smolvlm.py \
    --use_moe --num_experts 4 --moe_aux_loss_weight 0.01 \
    --train_metas_path ./datasets/metas/vlabench_debug_train.json \
    --norm_stats_path ./norm_stats/vlabench_norm.json \
    --action_mode vlabench_joint --num_views 4 \
    --iters 50 --batch_size 2 --output_dir /tmp/test_moe \
    --log_interval 1
```

预期：日志中出现 `velocity_loss` 和 `moe_aux_loss` 两个损失项，均为正数且合理（velocity_loss < 1.0，moe_aux_loss < 0.1）。

- [ ] **Step 2: 验证显存占用低于密集模型**

```bash
# 密集模型
CUDA_VISIBLE_DEVICES=2 python -c "
import torch
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.configuration_smolvlm_vla import SmolVLMVLAConfig
cfg = SmolVLMVLAConfig(smolvlm_model_path='/datasets/models/smolvlm/SmolVLM-500M-Instruct',
                       action_mode='vlabench_joint', num_views=4)
m = SmolVLMVLA(cfg).cuda()
print('Dense params:', sum(p.numel() for p in m.transformer.parameters()))
"

# MoE 模型
CUDA_VISIBLE_DEVICES=2 python -c "
import torch
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.configuration_smolvlm_vla import SmolVLMVLAConfig
cfg = SmolVLMVLAConfig(smolvlm_model_path='/datasets/models/smolvlm/SmolVLM-500M-Instruct',
                       action_mode='vlabench_joint', num_views=4, use_moe=True, num_experts=4)
m = SmolVLMVLA(cfg).cuda()
print('MoE params:', sum(p.numel() for p in m.transformer.parameters()))
"
```

预期：MoE 模型参数量约为密集模型的 4 倍，但每步激活参数约为 2/4 = 50%。

- [ ] **Step 3: Commit**

```bash
git commit -m "test: verify MoE training runs and produces dual loss terms"
```
