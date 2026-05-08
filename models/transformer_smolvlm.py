"""
SmolVLM Action Transformer

Action Transformer specifically designed for SmolVLM-VLA.
Key difference from the original transformer:
  - No aux_visual_inputs: all views are processed together by SmolVLM
  - VLM outputs a single unified feature for all views
  - Simpler architecture: x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)
"""

from __future__ import annotations

import math
from functools import partial
from typing import Final, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- Small utils ----------------------------------

def _to_2tuple(x) -> Tuple:
    """Minimal replacement for timm.layers.to_2tuple."""
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


def _has_sdp_attention() -> bool:
    """Check if we can use PyTorch fused scaled_dot_product_attention."""
    return hasattr(F, "scaled_dot_product_attention")


# ---------------------------------- MLP --------------------------------------

class Mlp(nn.Module):
    """MLP used in ViT-style blocks."""

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
    """Multi-Head Self-Attention with optional fused SDPA fallback."""

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
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ------------------------------- Utilities -----------------------------------

def basic_init(module: nn.Module) -> None:
    """Apply basic initialization to Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
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


# ------------------------------- Core Layers ----------------------------------

class TransformerBlock(nn.Module):
    """Standard Transformer block (pre-LN)."""

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


# ------------------------------- DiT Layers (AdaLN) ----------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """DiT Block with Adaptive Layer Normalization (AdaLN)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        modulation_params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation_params.chunk(6, dim=-1)
        )
        
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm)
        
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class FinalLayer(nn.Module):
    """DiT Final Layer with AdaLN."""
    
    def __init__(self, hidden_size: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# --------------------------- Main Model (SmolVLM Version) ---------------------------------------

class SmolVLMActionTransformer(nn.Module):
    """
    Flow Matching Transformer for action prediction - SmolVLM Version.

    Key difference from ActionTransformer:
      - No aux_visual_inputs: SmolVLM processes all views together
      - Simpler forward: x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)
      - Only one visual input stream
    
    Supports two modes:
    - Concat mode (use_adaln=False): Original architecture
    - AdaLN mode (use_adaln=True): DiT style conditioning
    """

    def __init__(
        self,
        hidden_size: int = 768,
        vlm_hidden_size: int = 576,  # Will be overridden by actual model config
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
            # ====== Hybrid Mode: AdaLN（time+proprio）+ Concat（VLM token 保留在序列里）======
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
            # ========== DiT Mode: AdaLN ==========
            self.blocks = nn.ModuleList(
                [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
            )

            # Condition encoders
            self.time_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            # VLM pooling projection (no aux_visual needed)
            self.vlm_cond_proj = nn.Linear(vlm_hidden_size, hidden_size)
            # Proprio conditioning: GRU（K>1）或线性映射（K=1）
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
            
            # Position encoding
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
            nn.init.normal_(self.pos_emb, std=0.02)
            
            # Final layer
            self.final_layer = FinalLayer(hidden_size, dim_action)
        else:
            # ========== Concat Mode: Original architecture ==========
            self.blocks = nn.ModuleList(
                [TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
            )

            # VLM projection only (no aux_visual_proj needed for SmolVLM)
            self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)

            self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
            nn.init.normal_(self.pos_emb, std=0.02)

            self.norm = nn.LayerNorm(hidden_size)
            
            # Action encoder/decoder
            action_input_dim = dim_action + dim_time + dim_propio
            self.action_encoder = nn.Linear(action_input_dim, hidden_size)
            self.action_decoder = nn.Linear(hidden_size, dim_action)

        self.apply(basic_init)

    def forward(
        self,
        vlm_features: torch.Tensor,  # [B, T_vlm, D] - unified features from SmolVLM
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for SmolVLM Action Transformer.

        Inputs
        ------
        vlm_features : [B, T_vlm, D] - Unified features from SmolVLM (all views processed together)
        action_with_noise : [B, T_action, dim_action]
        proprio : [B, dim_proprio]
        t : [B]

        Returns
        -------
        Tensor: Predicted velocity, [B, T_action, dim_action]
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
        Concat mode forward pass.
        
        Simplified: x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)
        No aux_visual_inputs needed.
        """
        B, num_actions = action_with_noise.shape[:2]

        # Encode (action + proprio + time) → tokens
        # Concat 模式只用最新帧 proprio（K>1 时取最后一帧，向后兼容）
        if proprio.dim() == 3:
            proprio = proprio[:, -1, :]  # [B, K, D] → [B, D]
        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions, proprio.shape[-1])
        
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens)  # [B, T_action, H]

        # Project VLM features and concatenate (no aux_visual needed)
        x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)

        # Add positional embeddings
        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}."
            )
        x = x + self.pos_emb[:, :seq_len, :]

        # Transformer backbone
        for block in self.blocks:
            x = block(x)

        # Decode only the action segment
        return self.action_decoder(self.norm(x[:, :num_actions]))
    
    def _forward_adaln(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        DiT/AdaLN mode forward pass.
        
        Conditions (time, vlm, proprio) injected via AdaLN.
        No aux_visual needed for SmolVLM.
        """
        B, num_actions = action_with_noise.shape[:2]
        
        # ========== 1. Build global condition c ==========
        # Time embedding
        t_emb = timestep_embedding(t, self.hidden_size)
        t_emb = self.time_proj(t_emb)  # [B, H]
        
        # VLM condition: Global Average Pooling
        vlm_cond = self.vlm_cond_proj(vlm_features.mean(dim=1))  # [B, H]
        
        # Proprio condition：GRU 历史编码（K>1）或单帧线性映射（K=1）
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

        # Fuse all conditions
        c = t_emb + vlm_cond + proprio_cond  # [B, H]
        
        # ========== 2. Encode action sequence ==========
        x = self.action_encoder(action_with_noise)  # [B, T_action, H]
        
        # Add position encoding
        x = x + self.pos_emb[:, :num_actions, :]
        
        # ========== 3. DiT Blocks with AdaLN ==========
        for block in self.blocks:
            x = block(x, c)
        
        # ========== 4. Final Layer with AdaLN ==========
        return self.final_layer(x, c)

    def _forward_hybrid(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        混合模式 Forward：AdaLN（time+proprio）+ Concat（VLM token 保留在序列里）。

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
