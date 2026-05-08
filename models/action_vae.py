"""
ActionVAE: Variational Autoencoder for action chunks.

Based on RoLD (arxiv:2403.07312): encodes action trajectory [B,T,D_a] to
compact latent code z [B,d_z], enabling flow matching in a low-dimensional
latent space rather than the full action sequence space.

Components:
  - ActionVAE: Transformer encoder → (μ,log σ²) → z; cross-attn decoder → â
  - LatentFlowNet: MLP-based flow matching head in latent space [B,d_z]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── helpers ───────────────────────────────────────

def _sinusoidal_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(100) * torch.arange(half, dtype=t.dtype, device=t.device) / half
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ─────────────────────────── attention blocks ───────────────────────────────

class _SelfAttn(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=True)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class _CrossAttn(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=True)
        self.kv_proj = nn.Linear(kv_dim, 2 * hidden, bias=True)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(ctx).reshape(B, -1, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, C))


class _EncBlock(nn.Module):
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
    def __init__(self, hidden: int, heads: int, vlm_dim: int):
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
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), ctx)
        x = x + self.mlp(self.norm3(x))
        return x


# ─────────────────────────── ActionVAE ──────────────────────────────────────

class ActionVAE(nn.Module):
    """
    VAE for action chunks (RoLD arxiv:2403.07312).

    Encoder
    -------
    Input: action [B, T, D_a] (normalized)
    1. Linear project → [B, T, H]
    2. Prepend learnable CLS token → [B, T+1, H]
    3. Transformer self-attention (enc_depth layers)
    4. CLS output → Linear → [μ, log σ²] ∈ R^{2·d_z}
    5. Reparameterize: z = μ + ε·exp(0.5·log σ²)

    Decoder
    -------
    Input: z [B, d_z] + VLM features [B, L, D_vlm]
    1. Linear z → [B, T, H] (T action slots)
    2. dec_depth blocks of (self-attn → cross-attn to VLM → FFN)
    3. Linear → [B, T, D_a] reconstructed actions
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

        # ── Encoder ──
        self.enc_action_proj = nn.Linear(dim_action, hidden)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.cls_token, std=0.02)
        self.enc_pos = nn.Parameter(torch.zeros(1, seq_len + 1, hidden))
        nn.init.normal_(self.enc_pos, std=0.02)
        self.enc_blocks = nn.ModuleList([_EncBlock(hidden, heads) for _ in range(enc_depth)])
        self.enc_norm = nn.LayerNorm(hidden)
        self.to_mu_logvar = nn.Linear(hidden, 2 * latent_dim)

        # ── Decoder ──
        self.z_to_slots = nn.Linear(latent_dim, seq_len * hidden)
        self.dec_pos = nn.Parameter(torch.zeros(1, seq_len, hidden))
        nn.init.normal_(self.dec_pos, std=0.02)
        self.dec_blocks = nn.ModuleList([_DecBlock(hidden, heads, vlm_hidden) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(hidden)
        self.action_out = nn.Linear(hidden, dim_action)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def encode(
        self, action: torch.Tensor, vlm_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode action chunk → (μ, log σ², z).

        Parameters
        ----------
        action : [B, T, D_a]  normalized action sequence
        vlm_features : [B, L, D_vlm]  (unused in encoder; kept for API symmetry)

        Returns
        -------
        mu, log_var : [B, d_z]
        z           : [B, d_z]  (reparameterized sample at training, = mu at eval)
        """
        B = action.shape[0]
        x = self.enc_action_proj(action)                          # [B, T, H]
        cls = self.cls_token.expand(B, -1, -1)                    # [B, 1, H]
        x = torch.cat([cls, x], dim=1) + self.enc_pos             # [B, T+1, H]

        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)

        mu_logvar = self.to_mu_logvar(x[:, 0])                    # [B, 2·d_z]
        mu, log_var = mu_logvar.chunk(2, dim=-1)

        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        else:
            z = mu
        return mu, log_var, z

    def decode(self, z: torch.Tensor, vlm_features: torch.Tensor) -> torch.Tensor:
        """
        Decode latent z → action chunk, cross-attending to VLM features.

        Parameters
        ----------
        z            : [B, d_z]
        vlm_features : [B, L, D_vlm]

        Returns
        -------
        action : [B, T, D_a]
        """
        B = z.shape[0]
        slots = self.z_to_slots(z).reshape(B, self.seq_len, self.hidden)  # [B, T, H]
        slots = slots + self.dec_pos

        for blk in self.dec_blocks:
            slots = blk(slots, vlm_features)
        slots = self.dec_norm(slots)
        return self.action_out(slots)                               # [B, T, D_a]

    def forward(
        self, action: torch.Tensor, vlm_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z, mu, log_var, recon_action)."""
        mu, log_var, z = self.encode(action, vlm_features)
        recon = self.decode(z, vlm_features)
        return z, mu, log_var, recon


# ─────────────────────────── LatentFlowNet ──────────────────────────────────

class LatentFlowNet(nn.Module):
    """
    Flow matching velocity head operating in the VAE latent space.

    Input  : z_t [B,d_z],  t [B],  vlm_features [B,L,D_vlm],  proprio [B,D_p]
    Output : velocity v_t [B,d_z]

    Architecture: MLP with residual blocks conditioned on VLM global pool,
    proprio, and sinusoidal time embedding.  Compact because d_z=32 << T·D_a.
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

        self.vlm_proj    = nn.Linear(vlm_hidden, hidden)
        self.proprio_proj = nn.Linear(dim_proprio, hidden)
        self.time_proj   = nn.Sequential(
            nn.Linear(dim_time, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        # Fuse: z_t || vlm_pool || proprio || time → hidden
        self.input_proj = nn.Linear(latent_dim + hidden * 3, hidden)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 2),
                nn.SiLU(),
                nn.Linear(hidden * 2, hidden),
            )
            for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, latent_dim)
        # Zero-init output so initial velocity ≈ 0 (stable training start)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        z_t: torch.Tensor,           # [B, d_z]
        t: torch.Tensor,             # [B]
        vlm_features: torch.Tensor,  # [B, L, D_vlm]
        proprio: torch.Tensor,       # [B, D_p] or [B, K, D_p]
    ) -> torch.Tensor:
        # Global VLM representation
        vlm_pool = self.vlm_proj(vlm_features.mean(dim=1))    # [B, H]

        # Proprio: take last frame if history window
        if proprio.dim() == 3:
            proprio = proprio[:, -1, :]
        prop_emb = self.proprio_proj(proprio)                  # [B, H]

        # Sinusoidal time embedding
        t_emb = self.time_proj(_sinusoidal_emb(t, self.dim_time))  # [B, H]

        x = self.input_proj(torch.cat([z_t, vlm_pool, prop_emb, t_emb], dim=-1))

        for blk in self.blocks:
            x = x + blk(x)

        return self.output(self.out_norm(x))                   # [B, d_z]


__all__ = ["ActionVAE", "LatentFlowNet"]
