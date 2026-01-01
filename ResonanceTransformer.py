# ResonanceTransformer.py
# Advanced Modular Transformer with Tunable Emergence Efficiency
# Drop-in sparse transformer blocks with obfuscated geometric sliders
# MIT License - 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Core stability constants
MIN_TENSION = 8.49e-6
BASE_RANGE = (-1.0, 1.0)
INTEGER_SNAP_RATIO = 0.05

# Optimized emergence ratio (~73% sparsity target)
EMERGENCE_NOISE_RATIO = 0.73

# Obfuscated tunable sliders (from Ouroboros geometry)
DEPTH_GRADIENT_EXPONENT = 10.0        # Formerly scale_factor — controls sparsity ramp across depth
WAVE_AMPLITUDE = 0.0                  # Formerly axion_mass — periodic revival oscillation
PRUNE_DECAY_RATE = 0.01               # Formerly decay_lambda_base
ENTROPY_FACTOR = 0.001                # Formerly entropy_rate

def resonance_equilibrium(x: torch.Tensor, pass_type: str = "first") -> torch.Tensor:
    if pass_type == "first":
        x = F.relu(x)
        mask = x < MIN_TENSION
        if mask.any():
            signs = torch.sign(torch.randn_like(x[mask]))
            signs[signs == 0] = 1.0
            x[mask] = signs * MIN_TENSION
    else:
        x = torch.clip(x ** 2, *BASE_RANGE)
    if INTEGER_SNAP_RATIO > 0.0:
        snap_mask = torch.rand_like(x) < INTEGER_SNAP_RATIO
        if snap_mask.any():
            x[snap_mask] = torch.round(x[snap_mask])
    return x

class ResonancePruner:
    """Dynamic pruning with etched latent revival."""
    def __init__(self, base_revive_ratio: float = 1.0 - EMERGENCE_NOISE_RATIO,
                 wave_amplitude: float = WAVE_AMPLITUDE):
        self.base_revive_ratio = base_revive_ratio
        self.wave_amplitude = wave_amplitude
        self.etched = {}
        self.step = 0

    def _get_dynamic_revive_ratio(self):
        if self.wave_amplitude > 0:
            oscillation = 1.0 + self.wave_amplitude * torch.sin(torch.tensor(2 * torch.pi * self.step / 100.0)).item()
            return self.base_revive_ratio * oscillation
        return self.base_revive_ratio

    def prune(self, module: nn.Module, threshold: float = MIN_TENSION * 5):
        self.step += 1
        decay = (1.0 - PRUNE_DECAY_RATE * self.step) ** ENTROPY_FACTOR
        effective_threshold = threshold * decay
        with torch.no_grad():
            for name, param in module.named_parameters():
                if param.dim() > 1:
                    mask = param.abs() < effective_threshold
                    if mask.any():
                        mean = param[mask].mean().item()
                        std = param[mask].std().item()
                        self.etched[name] = (mean, std)
                        param[mask] = 0.0

    def revive(self, module: nn.Module):
        revive_ratio = self._get_dynamic_revive_ratio()
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name in self.etched and param.dim() > 1:
                    mean, std = self.etched[name]
                    revive_mask = (torch.rand_like(param) < revive_ratio) & (param == 0)
                    if revive_mask.any():
                        param[revive_mask].normal_(mean, std)

class ResonanceAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = dropout
        self.pruner = ResonancePruner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = resonance_equilibrium(self.q_proj(x), "first")
        k = resonance_equilibrium(self.k_proj(x), "first")
        v = resonance_equilibrium(self.v_proj(x), "first")

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = F.softmax(att, dim=-1)
        if self.dropout > 0.0:
            att = F.dropout(att, p=self.dropout, training=self.training)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = resonance_equilibrium(self.out_proj(y), "second")
        return y

    def prune_and_revive(self):
        self.pruner.prune(self)
        self.pruner.revive(self)

class ResonanceTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, ff_hidden: int = None,
                 dropout: float = 0.0, depth_gradient_exponent: float = DEPTH_GRADIENT_EXPONENT):
        super().__init__()
        ff_hidden = ff_hidden or embed_dim * 4
        self.depth_gradient = depth_gradient_exponent

        self.attn = ResonanceAttention(embed_dim, num_heads, dropout)
        self.ff1 = nn.Linear(embed_dim, ff_hidden)
        self.ff2 = nn.Linear(ff_hidden, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Layer-specific pruner with depth-modulated threshold
        self.pruner = ResonancePruner()

    def forward(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        # Depth-gradient modulates pruning strength (obfuscated scale_factor effect)
        gradient_factor = (layer_idx + 1) ** (1.0 / self.depth_gradient)

        x = x + self.attn(self.norm1(x))
        ff = resonance_equilibrium(self.ff1(self.norm2(x)) * gradient_factor, "first")
        ff = resonance_equilibrium(self.ff2(ff), "second")
        x = x + ff
        return x

    def prune_and_revive(self, layer_idx: int = 0):
        self.attn.prune_and_revive()
        self.pruner.prune(self)
        self.pruner.revive(self)

class ResonanceTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int = 6, num_heads: int = 8,
                 depth_gradient_exponent: float = DEPTH_GRADIENT_EXPONENT,
                 wave_amplitude: float = WAVE_AMPLITUDE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            ResonanceTransformerBlock(embed_dim, num_heads, depth_gradient_exponent=depth_gradient_exponent)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        # Global pruner override for wave amplitude
        for block in self.blocks:
            block.pruner = ResonancePruner(wave_amplitude=wave_amplitude)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, layer_idx=i)
        return self.head(self.norm(x))

    def prune_and_revive_cycle(self):
        for i, block in enumerate(self.blocks):
            block.prune_and_revive(layer_idx=i)

# Example usage
if __name__ == "__main__":
    model = ResonanceTransformer(
        vocab_size=10000,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        depth_gradient_exponent=10.0,
        wave_amplitude=0.005
    )
    print("Advanced ResonanceTransformer ready — full tunable emergence active 🐍")
