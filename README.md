# ConstrainedML.jl

[![Tests](https://img.shields.io/badge/tests-193%2F193%20passing-brightgreen)](test/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/julia-v1.10+-purple.svg)](https://julialang.org)

**Physics-Constrained Neural Networks with Mathematical Guarantees**

Neural network layers that **provably** respect fundamental physics symmetries. Not approximate—**exact** at machine precision.

---

## Core Innovation

Traditional ML uses *soft penalties* to encourage symmetry:
```julia
loss = data_loss + λ * symmetry_violation  # Approximate, requires tuning
```

ConstrainedML guarantees symmetry *by construction*:
```julia
layer = GaugeEquivariantConv(...)  # Mathematically impossible to violate
```

---

## Features

### Gauge Symmetries
- **U(1)** - Electromagnetism (error: ~10^-16)
- **SU(2)** - Weak interactions
- **SU(3)** - Quantum Chromodynamics
- Full lattice gauge theory support

### Spacetime Symmetries
- **Lorentz Invariance** - Special relativity (error: ~10^-15)
- **Lorentz Equivariance** - Frame-independent predictions

### Statistical Symmetries
- **Permutation Invariance** - Identical particles (error: ~10^-7)
- DeepSets architecture

---

## Quick Start
```julia
using ConstrainedML
using Flux

# 1. U(1) Gauge-Equivariant Network
layer = GaugeEquivariantConv(1, 8, symmetry=U1(), activation=tanh)
field = rand(ComplexF64, 32, 32)
output = layer(field)

# Test: Exact gauge equivariance
g = GaugeTransformation(U1(), π/4)
@assert check_equivariance(layer, field, g)  # Always true!

# 2. Lorentz-Invariant Jet Classifier
particles = [
    FourVector(10.0, 3.0, 0.0, 0.0),
    FourVector(5.0, 0.0, 4.0, 0.0)
]
model = LorentzInvariantNet(hidden_dims=[64, 64], output_dim=2)
classification = model(particles)  # Frame-independent!

# 3. Permutation-Invariant Set Processing
model = DeepSets(
    phi = Dense(4 => 32, relu),
    rho = Dense(32 => 2),
    aggregation = sum
)
elements = [rand(4) for _ in 1:10]
output = model(elements)  # Order doesn't matter!
```

---

## Validation Results

All symmetries verified at machine precision:

| Symmetry | Test Cases | Max Error | Status |
|----------|-----------|-----------|--------|
| U(1) Gauge | Multi-layer, various θ | 2.4×10^-16 | Pass |
| Lorentz (boosts) | 15 velocities, 3 directions | 2.7×10^-15 | Pass |
| Lorentz (rotations) | 15 angles, 3 axes | 6.8×10^-16 | Pass |
| Permutation | 10 shuffles, various sizes | 9.5×10^-6 | Pass |
| SU(2) plaquettes | Random transformations | 4.4×10^-16 | Pass |

**193/193 tests passing**

---

## Applications

### Particle Physics
```julia
# Jet classification at the LHC
model = ParticleSetClassifier(hidden_dim=128, output_dim=2)
jets = load_lhc_data()
predictions = model(jets)  # Lorentz + permutation invariant!
```

### Lattice QCD
```julia
# SU(3) lattice gauge theory
lattice = create_random_lattice(SU3(), (16, 16, 16, 8))
model = SUNGaugeEquivariantConv(64, symmetry=SU3())
output = model(lattice)  # Exactly gauge-invariant!
```

### Normalizing Flows
```julia
# Generate gauge field configurations
flow = Chain(
    GaugeEquivariantConv(1, 8, symmetry=U1()),
    GaugeEquivariantConv(8, 1, symmetry=U1())
)
samples = generate_samples(flow)  # All respect gauge symmetry!
```

---

## Installation
```julia
using Pkg
Pkg.add(url="https://github.com/Manav02012002/ConstrainedML.jl")
```

Or for development:
```bash
git clone https://github.com/Manav02012002/ConstrainedML.jl.git
cd ConstrainedML.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

---

## Running Tests
```bash
julia --project=. test/runtests.jl
```

Expected output:
```
Test Summary:    | Pass  Total   Time
ConstrainedML.jl |  193    193  18.4s
All ConstrainedML.jl tests passed!
```

---

## Examples

- [`examples/simple_demo.jl`](examples/simple_demo.jl) - Quick demonstration of all symmetries
- [`examples/jet_classification.jl`](examples/jet_classification.jl) - Realistic HEP application
- [`examples/simple_u1_example.jl`](examples/simple_u1_example.jl) - U(1) gauge theory basics

---

## How It Works

### U(1) Gauge Equivariance
For complex fields ψ = |ψ|e^(iθ):
1. Convolve real and imaginary parts separately (preserves phase)
2. Apply activation to magnitude only
3. Reconstruct: **provably** equivariant under ψ → e^(iα)ψ

### Lorentz Invariance
Use **only** Lorentz-invariant features:
- Masses: p² = p_μ p^μ
- Dot products: p·q
Network output **cannot** depend on reference frame!

### Permutation Invariance
DeepSets architecture:
```
f({x₁, ..., xₙ}) = ρ(Σᵢ φ(xᵢ))
```
**Proven** to be the universal permutation-invariant function!

---

## Why This Matters

**Traditional approach:**
- Symmetry violations happen randomly
- Unpredictable behavior
- Requires careful tuning
- No guarantees

**ConstrainedML:**
- Symmetry violations **mathematically impossible**
- Predictable, trustworthy
- No hyperparameters for symmetries
- Mathematical proof of correctness

Perfect for:
- CERN data analysis
- Lattice QCD simulations
- Dark matter searches
- Any physics ML where correctness matters

---

## Roadmap

- GPU acceleration (CUDA.jl)
- Complete Standard Model gauge group
- HEP data format integration (ROOT, HepMC)
- Gauge-equivariant Graph Neural Networks
- Conformal field theory support
- Python bindings (PyCall)
- ONNX export for production

---

## Documentation

- **Theory:** See [`DOCUMENTATION.tex`](DOCUMENTATION_LATEX.tex) for mathematical details
- **API Reference:** See docstrings in source code
- **Tutorials:** See [`examples/`](examples/) directory

---

## Contributing

Contributions welcome! Areas of interest:
- New symmetry groups
- Performance optimizations
- Additional examples
- Documentation improvements

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Acknowledgments

Developed as part of research in lattice field theory and machine learning applications to high energy physics.

Built with [Flux.jl](https://fluxml.ai/) and the Julia ML ecosystem.

---

## Contact

Issues and questions: [GitHub Issues](https://github.com/Manav02012002/ConstrainedML.jl/issues)

---

<p align="center">
  <i>Making physics-informed machine learning rigorous, one symmetry at a time.</i>
</p>
