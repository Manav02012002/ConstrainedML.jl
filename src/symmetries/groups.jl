"""
    groups.jl

Defines abstract and concrete symmetry groups used in physics.
"""

# ============================================================================
# Abstract Types
# ============================================================================

"""
    Symmetry

Abstract supertype for all symmetries.
"""
abstract type Symmetry end

"""
    LieGroup <: Symmetry

Abstract type for continuous Lie groups.
"""
abstract type LieGroup <: Symmetry end

# ============================================================================
# Concrete Groups
# ============================================================================

"""
    U1 <: LieGroup

The U(1) gauge group (circle group, electromagnetism).
"""
struct U1 <: LieGroup end

"""
    SU2 <: LieGroup

The SU(2) gauge group (weak interactions, isospin).
"""
struct SU2 <: LieGroup end

"""
    SU3 <: LieGroup

The SU(3) gauge group (QCD, color charge).
"""
struct SU3 <: LieGroup end

"""
    LorentzGroup <: LieGroup

The Lorentz group SO(1,3).
"""
struct LorentzGroup <: LieGroup
    metric::Symbol
    
    function LorentzGroup(; metric::Symbol=:minkowski)
        metric ∈ [:minkowski, :euclidean] || 
            throw(ArgumentError("metric must be :minkowski or :euclidean"))
        new(metric)
    end
end

"""
    PermutationGroup <: Symmetry

The symmetric group S_n of all permutations.
"""
struct PermutationGroup <: Symmetry
    n::Int
    
    function PermutationGroup(n::Int)
        n > 0 || throw(ArgumentError("n must be positive"))
        new(n)
    end
end

# ============================================================================
# Group Properties
# ============================================================================

"""
    dim(group::LieGroup)

Return the dimension of the Lie group (number of parameters).
"""
dim(::U1) = 1
dim(::SU2) = 3
dim(::SU3) = 8
dim(::LorentzGroup) = 6

# ============================================================================
# Display
# ============================================================================

Base.show(io::IO, ::U1) = print(io, "U(1)")
Base.show(io::IO, ::SU2) = print(io, "SU(2)")
Base.show(io::IO, ::SU3) = print(io, "SU(3)")
Base.show(io::IO, g::LorentzGroup) = print(io, "Lorentz($(g.metric))")
Base.show(io::IO, g::PermutationGroup) = print(io, "S_$(g.n)")

# ============================================================================
# Helper Functions
# ============================================================================

"""
    name(group::LieGroup)

Return the mathematical name of the group.
"""
name(::U1) = "U(1)"
name(::SU2) = "SU(2)"
name(::SU3) = "SU(3)"
name(::LorentzGroup) = "SO(1,3)"

"""
    isnon_abelian(group::LieGroup)

Check if the group is non-abelian (elements don't commute).
"""
isnon_abelian(::U1) = false
isnon_abelian(::SU2) = true
isnon_abelian(::SU3) = true
isnon_abelian(::LorentzGroup) = true  # Boosts don't commute with rotations