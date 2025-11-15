"""
    transformations.jl

Defines how symmetry groups act on physics objects (transformations).
"""

using LinearAlgebra

# ============================================================================
# Abstract Types
# ============================================================================

"""
    Transformation

Abstract supertype for all group transformations.
"""
abstract type Transformation end

# ============================================================================
# Gauge Transformations
# ============================================================================

"""
    GaugeTransformation{G<:LieGroup, T}

A gauge transformation for group G.

# Fields
- `group::G`: The gauge group
- `parameters::T`: Group-specific parameters (angles for U(1), matrices for SU(N))
"""
struct GaugeTransformation{G<:LieGroup, T} <: Transformation
    group::G
    parameters::T
end

# Convenience constructors
GaugeTransformation(group::U1, θ::Real) = GaugeTransformation{U1, typeof(θ)}(group, θ)
GaugeTransformation(group::SU2, params::AbstractVector) = GaugeTransformation{SU2, typeof(params)}(group, params)
GaugeTransformation(group::SU3, params::AbstractVector) = GaugeTransformation{SU3, typeof(params)}(group, params)

"""
    transform(g::GaugeTransformation{U1}, field::Complex)

Apply U(1) gauge transformation: ψ → e^{iθ} ψ
"""
function transform(g::GaugeTransformation{U1}, field::Number)
    return exp(im * g.parameters) * field
end

"""
    transform(g::GaugeTransformation{U1}, field::AbstractArray{<:Complex})

Apply U(1) gauge transformation to array of complex numbers: ψ → e^{iθ} ψ
"""
function transform(g::GaugeTransformation{U1}, field::AbstractArray{<:Complex})
    return exp(im * g.parameters) .* field
end

"""
    transform(g::GaugeTransformation{U1}, field::GaugeField{U1})

Apply U(1) gauge transformation to entire field.
"""
function transform(g::GaugeTransformation{U1}, field::GaugeField{U1})
    # If parameters is a single angle, apply uniformly
    if g.parameters isa Real
        transformed_data = exp(im * g.parameters) .* field.data
    # If parameters is an array (local transformation), apply pointwise
    elseif g.parameters isa AbstractArray
        transformed_data = exp.(im .* g.parameters) .* field.data
    else
        error("Unexpected parameter type: $(typeof(g.parameters))")
    end
    
    return GaugeField(field.group, transformed_data)
end

"""
    transform(g::GaugeTransformation{U1}, config::LatticeConfig{U1})

Apply U(1) gauge transformation to lattice configuration.
For gauge theories, links transform as: U_μ(x) → e^{iα(x)} U_μ(x) e^{-iα(x+μ̂)}
"""
function transform(g::GaugeTransformation{U1}, config::LatticeConfig{U1})
    # For simplicity, implement global transformation first
    # (local transformation requires parallel transport)
    if g.parameters isa Real
        # Global U(1) transformation
        transformed_links = exp(im * g.parameters) .* config.links
        return LatticeConfig(config.group, config.lattice_size, transformed_links)
    else
        error("Local gauge transformations not yet implemented")
    end
end

# ============================================================================
# Lorentz Transformations
# ============================================================================

"""
    LorentzTransformation{T<:Real}

A Lorentz transformation (boost or rotation).

# Fields
- `matrix::SMatrix{4,4,T}`: The 4×4 Lorentz transformation matrix Λ
"""
struct LorentzTransformation{T<:Real} <: Transformation
    matrix::SMatrix{4,4,T}
    
    function LorentzTransformation(Λ::AbstractMatrix{T}) where T<:Real
        size(Λ) == (4, 4) || throw(ArgumentError("Lorentz transformation must be 4×4"))
        # TODO: Add check that Λ^T η Λ = η (preserves metric)
        new{T}(SMatrix{4,4,T}(Λ))
    end
end

"""
    transform(Λ::LorentzTransformation, p::FourVector)

Apply Lorentz transformation to four-vector: p^μ → Λ^μ_ν p^ν
"""
function transform(Λ::LorentzTransformation, p::FourVector)
    # Convert to vector, apply transformation, convert back
    p_vec = SVector(p.t, p.x, p.y, p.z)
    p_transformed = Λ.matrix * p_vec
    return FourVector(p_transformed[1], p_transformed[2], p_transformed[3], p_transformed[4])
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    boost(β::Real; direction=:x)

Create a Lorentz boost transformation with velocity β in given direction.

# Arguments
- `β::Real`: Velocity (in units where c=1), must satisfy |β| < 1
- `direction::Symbol`: Direction of boost (:x, :y, or :z)

# Returns
- `LorentzTransformation`: The boost matrix

# Examples
```julia
Λ = boost(0.5, direction=:x)  # Boost in x-direction with v=0.5c
p_boosted = transform(Λ, p)
```
"""
function boost(β::Real; direction::Symbol=:x)
    abs(β) < 1 || throw(ArgumentError("Velocity must satisfy |β| < 1"))
    
    γ = 1 / sqrt(1 - β^2)
    
    if direction == :x
        Λ = [
            γ      -γ*β    0      0;
            -γ*β    γ      0      0;
            0       0      1      0;
            0       0      0      1
        ]
    elseif direction == :y
        Λ = [
            γ       0     -γ*β    0;
            0       1      0      0;
            -γ*β    0      γ      0;
            0       0      0      1
        ]
    elseif direction == :z
        Λ = [
            γ       0      0     -γ*β;
            0       1      0      0;
            0       0      1      0;
            -γ*β    0      0      γ
        ]
    else
        throw(ArgumentError("Direction must be :x, :y, or :z"))
    end
    
    return LorentzTransformation(Λ)
end

"""
    rotation(θ::Real; axis=:z)

Create a spatial rotation transformation.

# Arguments
- `θ::Real`: Rotation angle in radians
- `axis::Symbol`: Axis of rotation (:x, :y, or :z)

# Examples
```julia
Λ = rotation(π/4, axis=:z)  # 45° rotation around z-axis
```
"""
function rotation(θ::Real; axis::Symbol=:z)
    c, s = cos(θ), sin(θ)
    
    if axis == :x
        Λ = [
            1   0    0    0;
            0   1    0    0;
            0   0    c   -s;
            0   0    s    c
        ]
    elseif axis == :y
        Λ = [
            1   0    0    0;
            0   c    0    s;
            0   0    1    0;
            0  -s    0    c
        ]
    elseif axis == :z
        Λ = [
            1   0    0    0;
            0   c   -s    0;
            0   s    c    0;
            0   0    0    1
        ]
    else
        throw(ArgumentError("Axis must be :x, :y, or :z"))
    end
    
    return LorentzTransformation(Λ)
end

"""
    identity_transform(::Type{LorentzGroup})

Return the identity Lorentz transformation.
"""
function identity_transform(::Type{LorentzGroup})
    return LorentzTransformation(SMatrix{4,4}(I))
end

"""
    identity_transform(::Type{U1})

Return the identity U(1) gauge transformation (angle = 0).
"""
function identity_transform(::Type{U1})
    return GaugeTransformation(U1(), 0.0)
end

# ============================================================================
# Display
# ============================================================================

Base.show(io::IO, g::GaugeTransformation{G}) where G = 
    print(io, "GaugeTransformation{$(G)}($(g.parameters))")

Base.show(io::IO, Λ::LorentzTransformation) = 
    print(io, "LorentzTransformation($(Λ.matrix[1,1]), ...)")

    # ============================================================================
# SU(N) Gauge Transformations
# ============================================================================

"""
    SUNGaugeTransformation{N, T}

Gauge transformation for SU(N) group.

For SU(2): 2×2 unitary matrix with det=1
For SU(3): 3×3 unitary matrix with det=1
"""
struct SUNGaugeTransformation{N, T<:Number} <: Transformation
    group::Union{SU2, SU3}
    matrix::Matrix{T}  # N×N unitary matrix
    
    function SUNGaugeTransformation(group::Union{SU2, SU3}, U::Matrix{T}) where T
        N = group isa SU2 ? 2 : 3
        
        # Validate dimensions
        size(U) == (N, N) || error("Matrix must be $(N)×$(N)")
        
        # Check unitarity (U† U ≈ I)
        if !isapprox(U' * U, I(N), rtol=1e-6)
            @warn "Matrix is not unitary"
        end
        
        # Check determinant ≈ 1
        if !isapprox(det(U), 1.0, rtol=1e-6)
            @warn "Determinant is not 1: det(U) = $(det(U))"
        end
        
        new{N, T}(group, U)
    end
end

"""
    random_SU2()

Generate random SU(2) matrix using Euler angles parametrization.
"""
function random_SU2()
    # SU(2) = unit quaternions
    # Parametrize as: exp(i σ·θ/2)
    θ = randn(3) * π  # Random angles
    
    # Pauli matrices
    σ1 = Complex.([0 1; 1 0])
    σ2 = Complex.([0 -im; im 0])
    σ3 = Complex.([1 0; 0 -1])
    
    # exp(i σ·θ/2) using matrix exponential
    H = im * (θ[1]*σ1 + θ[2]*σ2 + θ[3]*σ3) / 2
    U = exp(H)
    
    return SUNGaugeTransformation(SU2(), U)
end

"""
    random_SU3()

Generate random SU(3) matrix using Gell-Mann matrices.
"""
function random_SU3()
    # Use QR decomposition of random complex matrix
    A = randn(ComplexF64, 3, 3)
    Q, R = qr(A)
    
    # Normalize to det = 1
    Q = Matrix(Q) / det(Q)^(1/3)
    
    return SUNGaugeTransformation(SU3(), Q)
end

"""
    transform(g::SUNGaugeTransformation, ψ::AbstractVector)

Apply SU(N) gauge transformation to field in fundamental representation.
ψ → U ψ (matrix-vector multiplication)
"""
function transform(g::SUNGaugeTransformation{N}, ψ::AbstractVector) where N
    return g.matrix * ψ
end

"""
    transform(g::SUNGaugeTransformation, U_link::AbstractMatrix)

Transform a gauge link: U → g U g†
"""
function transform(g::SUNGaugeTransformation, U_link::AbstractMatrix)
    return g.matrix * U_link * g.matrix'
end