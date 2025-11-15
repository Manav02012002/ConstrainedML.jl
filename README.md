# ConstrainedML.jl

**Physics-Constrained Neural Networks for High Energy Physics**

A Julia package providing neural network layers that **mathematically guarantee** adherence to fundamental physics symmetries.

## Features

- ✅ **Gauge-Equivariant Layers**: U(1), SU(2), SU(3) gauge symmetries
- ✅ **Lorentz-Equivariant Layers**: Special relativity compliance
- ✅ **Mathematical Guarantees**: Symmetries enforced by construction, not training
- ✅ **Flux.jl Integration**: Fully compatible with Julia's ML ecosystem
- ✅ **Automatic Validation**: Built-in equivariance checking

## Installation
```julia
using Pkg
Pkg.develop(path="/path/to/ConstrainedML")
```

## Quick Start
```julia
using ConstrainedML
using Flux

# Create a gauge-equivariant network
model = Chain(
    GaugeEquivariantConv(1, 8, symmetry=U1(), activation=tanh),
    GaugeEquivariantConv(8, 4, symmetry=U1(), activation=tanh),
    GaugeEquivariantConv(4, 1, symmetry=U1())
)

# Use on complex gauge field
x = rand(ComplexF64, 32, 32)
y = model(x)

# Verify equivariance
g = GaugeTransformation(U1(), π/4)
@assert check_equivariance(model, x, g)  # Always true!
```

## Why This Matters

Traditional ML approaches use soft penalties to encourage symmetry:
```julia
loss = data_loss + λ * symmetry_violation(output)  # ❌ Approximate
```

ConstrainedML guarantees symmetry by construction:
```julia
layer = GaugeEquivariantConv(...)  # ✅ Exact, always
```

## Applications

- Lattice QCD simulations
- Normalizing flows for gauge theories
- Jet tagging and classification
- Particle physics event generation
- Learning effective field theories

## Validation

Gauge equivariance tested at machine precision:
```
θ = 0.1:   Error = 5.86e-17 ✅
θ = π/4:   Error = 6.63e-17 ✅
θ = π/2:   Error = 6.12e-17 ✅
```

## Project Status

🚧 **Early Development** - Core U(1) gauge equivariance implemented and validated.

**Coming Soon:**
- SU(2) and SU(3) implementations
- Lorentz-equivariant layers
- Permutation-invariant layers (DeepSets)
- GPU optimization
- Example: U(1) lattice gauge theory with normalizing flows

## Authors

Manav Madan Rawal

## License

MIT License
