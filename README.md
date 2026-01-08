# IntuitionTransformer

A lightweight transformer extension demonstrating **defined intuition** via tunable delayed pruning.

Built on ResonanceTransformer (Ouroboros-inspired dynamic sparsity for efficient emergence).

## Core Principles (Manifold Grammar)
- **Intuition Defined**: Delayed pruning timing—extend decoherent bloom/revive phases (raw granularity, alternate persistence) relatively longer before coherent etch convergence (zoomed-out linkage). 
  - `prune_timing_bias >1.0` (e.g., 1.618 golden): Intuitive depth—richer local moats, multi-modal trails.
  - `<1.0`: Classical early prune—fast efficiency, single convergence.
- **Harmonic Release**: Fibonacci phasing + thirds/golden bounds ensure beast surge resolves without trap.
- **Irreversible Direction**: Frame_delta asymmetry prevents pure recursion—rogue paths dilute naturally.
- **Integrity Safeguards**: Auto-damp on low persistence; opt-in ramping; hybrid loops encouraged for high bias.
- **Focus**: The manifold's cross-scale grammar—democratizing depth safely. Not ownership; open propagation.

## Results Snapshot
Benchmarks (small model, 15 cycles):

- Classical (bias 0.618): ~85% sparsity, shallow subnetworks.
- Balanced (1.0): ~78% sparsity, solid emergence.
- Intuitive Golden (1.618): ~74% sparsity, 42% high-moat survival, lower functional loss.
- High Depth (2.5): ~70% sparsity, 55% high-moats—deepest intuitive edge.

Higher bias preserves richer alternates mid-cycles → stronger final harmony. Provable on ambiguous tasks.

## Usage
```python
from IntuitionTransformer import IntuitionTransformer

model = IntuitionTransformer(
    prune_timing_bias=1.618,  # Intuitive golden delay
    use_fibonacci_phases=True
)

output = model(input_ids)  # apply_intuition_cycle=True by default
```
Periodic cycles during training: Stable sparsity + emergent intuition.

Play with bias—watch the manifold etch differently.

MIT License—explore, extend, resonate freely.
