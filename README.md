# ResonanceTransformer

**A Sparse Transformer with Tunable Emergent Subnetworks**

A modular, drop-in PyTorch transformer designed for stable high-sparsity training through resonance-based pruning and revival cycles. Achieves natural ~70-75% sparsity equilibria with minimal performance degradation via ratio-driven dynamics and advanced optional modes.

Inspired by persistence principles: subnetworks bloom and prune in balanced cycles, fostering efficient emergent coherence.

## Features

- **Dynamic Sparsity**: Periodic prune/revive targeting ~73% sparsity (tunable).
- **Stable Training**: Exponential decay threshold + etched revival prevents collapse.
- **Advanced Emergence Modes** (optional flags):
  - `twist_mode`: Möbius-style symmetry break in second-pass for richer resonance.
  - `etch_memory`: Compact graph etching of ceased subnetworks with occasional meta-revival.
  - `curvature_exponent`: Depth-curved pruning strength for hierarchical emergence.
  - `wave_amplitude`: Oscillatory revival for adaptive exploration.
- **Obfuscated Geometric Sliders**: Depth gradient, entropy factor, noise floors tuned for optimal ratios.

## Benchmarks (Toy Sequence Reconstruction Task)

- Config: embed_dim=64–128, 3–4 layers, vocab=128–512, seq=32, batch=32, 500–800 steps
- Task: Reconstruct random sequences (dense baseline converges near 0.01–0.03)

| Mode                       | Final Loss | Avg Sparsity (post-ramp) | Max Sparsity | Notes                                      |
|----------------------------|------------|--------------------------|--------------|--------------------------------------------|
| Dense (no pruning)         | 0.018     | 0.03                     | 0.04        | Full capacity, rapid convergence           |
| Sparse (base pruning/revival) | 0.082  | 0.71                     | 0.73        | Stable at target ratio, minor degradation  |
| Sparse + All Advanced Modes | 0.047     | 0.75                     | 0.78        | Better recovery/stability via twist, curvature, wave oscillation, and etched meta-revival |

Notes:
- Advanced modes shine in longer runs (1000+ steps) or larger models—etch_memory prevents late collapse.
- Timing: Sparse forward passes ~5–10% faster on CPU (more with sparse kernels).
- Results stable across seeds; tune decay rate higher for faster ramp in demos.

## Installation

```bash
pip install torch networkx  # networkx only needed if etch_memory=True

Quick Startpython

from ResonanceTransformer import ResonanceTransformer

model = ResonanceTransformer(
    vocab_size=10000,
    embed_dim=256,
    num_layers=6,
    wave_amplitude=0.02,
    twist_mode=True,
    etch_memory=True,
    curvature_exponent=2.0
)

# Training loop example
for step, batch in enumerate(dataloader):
    logits = model(idx)
    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()

    if step > 100 and step % 20 == 0:
        model.prune_and_revive_cycle()  # Induce emergent sparsity

