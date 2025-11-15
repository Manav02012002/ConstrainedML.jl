"""
    physics_types.jl

Defines physics data types like four-vectors and gauge fields.
"""

using StaticArrays
using LinearAlgebra

# ============================================================================
# Abstract Types
# ============================================================================

"""
    PhysicsObject

Abstract supertype for all physics objects.
"""
abstract type PhysicsObject end

# ============================================================================
# Four-Vectors
# ============================================================================

"""
    FourVector{T<:Real} <: PhysicsObject

A four-vector in Minkowski spacetime (t, x, y, z).

# Fields
- `t::T`: Time component
- `x::T`: x-component
- `y::T`: y-component  
- `z::T`: z-component

# Examples
```julia
p = FourVector(10.0, 1.0, 2.0, 3.0)  # Energy-momentum
p.t  # Energy
norm2 = p ⋅ p  # Invariant mass squared
```
"""
struct FourVector{T<:Real} <: PhysicsObject
    t::T
    x::T
    y::T
    z::T
end

# Convenience constructor
FourVector(components::AbstractVector) = FourVector(components[1], components[2], components[3], components[4])

"""
    ⋅(p::FourVector, q::FourVector)

Minkowski inner product: p⋅q = p.t*q.t - p.x*q.x - p.y*q.y - p.z*q.z
"""
function LinearAlgebra.dot(p::FourVector, q::FourVector)
    return p.t * q.t - p.x * q.x - p.y * q.y - p.z * q.z
end

# Make ⋅ available
⋅(p::FourVector, q::FourVector) = dot(p, q)

"""
    norm2(p::FourVector)

Minkowski norm squared: p² = p⋅p
"""
norm2(p::FourVector) = p ⋅ p

"""
    mass(p::FourVector)

Invariant mass: m = √(E² - |p|²)
"""
mass(p::FourVector) = sqrt(abs(norm2(p)))

# Addition
Base.:+(p::FourVector, q::FourVector) = FourVector(p.t + q.t, p.x + q.x, p.y + q.y, p.z + q.z)

# Scalar multiplication
Base.:*(a::Real, p::FourVector) = FourVector(a*p.t, a*p.x, a*p.y, a*p.z)
Base.:*(p::FourVector, a::Real) = a * p

# Display
Base.show(io::IO, p::FourVector) = print(io, "FourVector($(p.t), $(p.x), $(p.y), $(p.z))")

# ============================================================================
# Gauge Fields
# ============================================================================

"""
    GaugeField{G<:LieGroup, T<:Number, N} <: PhysicsObject

A gauge field configuration on a lattice.

# Fields
- `group::G`: The gauge group (U1, SU2, SU3, etc.)
- `data::Array{T,N}`: Field values on lattice
"""
struct GaugeField{G<:LieGroup, T<:Number, N} <: PhysicsObject
    group::G
    data::Array{T,N}
end

# Constructor with group inference

# Display
Base.show(io::IO, field::GaugeField{G}) where G = print(io, "GaugeField{$(G)} with size $(size(field.data))")

# Access underlying data
Base.getindex(field::GaugeField, args...) = getindex(field.data, args...)
Base.setindex!(field::GaugeField, val, args...) = setindex!(field.data, val, args...)
Base.size(field::GaugeField) = size(field.data)

# ============================================================================
# Lattice Configuration
# ============================================================================

"""
    LatticeConfig{G<:LieGroup, T<:Number}

A complete gauge field configuration on a spacetime lattice.

# Fields
- `group::G`: Gauge group
- `lattice_size::NTuple{4,Int}`: (Nx, Ny, Nz, Nt)
- `links::Array{T,5}`: Link variables [μ, x, y, z, t] where μ ∈ {1,2,3,4}
"""
struct LatticeConfig{G<:LieGroup, T<:Number} <: PhysicsObject
    group::G
    lattice_size::NTuple{4,Int}
    links::Array{T,5}  # [direction, x, y, z, t]
    
    function LatticeConfig(group::G, lattice_size::NTuple{4,Int}, links::Array{T,5}) where {G<:LieGroup, T<:Number}
        # Validate dimensions
        expected_size = (4, lattice_size...)
        size(links) == expected_size || throw(DimensionMismatch("Expected links of size $expected_size, got $(size(links))"))
        
        new{G,T}(group, lattice_size, links)
    end
end

# Constructor that initializes links
function LatticeConfig(group::G, lattice_size::NTuple{4,Int}) where {G<:LieGroup}
    # Initialize to identity (for U(1), this is 1.0 + 0.0im)
    if group isa U1
        links = ones(ComplexF64, 4, lattice_size...)
    else
        error("Auto-initialization not yet implemented for $(typeof(group))")
    end
    
    return LatticeConfig(group, lattice_size, links)
end

# Display
Base.show(io::IO, config::LatticeConfig{G}) where G = 
    print(io, "LatticeConfig{$(G)} on $(config.lattice_size) lattice")