"""
    ConstrainedML

A Julia package for building neural networks with guaranteed physics symmetries.
"""
module ConstrainedML

using Flux
using LinearAlgebra
using StaticArrays

# Include submodules
include("symmetries/groups.jl")
include("utils/physics_types.jl")
include("symmetries/transformations.jl")
include("utils/validation.jl")
include("layers/gauge_equivariant.jl")
include("layers/lorentz_equivariant.jl")
include("layers/permutation_invariant.jl")

# Exports - Groups
export Symmetry, LieGroup
export U1, SU2, SU3, LorentzGroup, PermutationGroup
export dim

# Exports - Physics Types
export PhysicsObject
export FourVector, ⋅, norm2, mass
export GaugeField, LatticeConfig

# Exports - Transformations
export Transformation
export GaugeTransformation, LorentzTransformation
export transform, boost, rotation
export identity_transform

# Exports - Layers
export EquivariantLayer
export GaugeEquivariantConv
export LorentzEquivariantLayer, LorentzInvariantNet  # NEW!

# Exports - Validation
export check_equivariance, equivariance_error
export DeepSets, EquivariantDeepSets, ParticleSetClassifier

# Add to exports section
export name, isnon_abelian

export SUNGaugeTransformation, random_SU2, random_SU3, compute_plaquette
export SUNGaugeEquivariantLayer
# Gauge theory functions
export compute_plaquette, plaquette_strength
export LatticeGaugeField, create_random_lattice, parallel_transport
export SUNGaugeEquivariantConv


end # module ConstrainedML