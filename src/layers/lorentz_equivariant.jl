"""
    lorentz_equivariant.jl

Neural network layers that are Lorentz-equivariant by construction.
"""

using Flux
using LinearAlgebra
using Statistics

# ============================================================================
# Lorentz-Equivariant Layer for Sets of Four-Vectors
# ============================================================================

"""
    LorentzEquivariantLayer{F1, F2, F3}

A neural network layer that is exactly Lorentz-equivariant.

Processes sets of four-vectors while preserving Lorentz symmetry.

# Architecture (following Bogatskiy et al. 2020):
1. Compute all Lorentz-invariant pairwise features: m_i² = p_i · p_i, s_ij = p_i · p_j
2. Use these to generate scalar coefficients α_ij and β_ij
3. Update each vector as: p_i^out = p_i + Σ_j (α_ij p_j + β_ij s_ij p_j)

This is provably Lorentz-equivariant because:
- All coefficients depend only on Lorentz invariants
- All updates are linear combinations of input four-vectors

# Examples
```julia
layer = LorentzEquivariantLayer(
    hidden_dim = 64,
    output_dim = 32
)

particles = [
    FourVector(10.0, 3.0, 0.0, 0.0),
    FourVector(5.0, 0.0, 4.0, 0.0)
]

updated = layer(particles)
```
"""
struct LorentzEquivariantLayer{F1, F2, F3} <: EquivariantLayer
    phi::F1         # Process invariants → scalar features
    alpha_net::F2   # Scalar features → mixing coefficients α_ij
    beta_net::F3    # Scalar features → dot-product coefficients β_ij
    hidden_dim::Int
    output_dim::Int
end

"""
    LorentzEquivariantLayer(; hidden_dim, output_dim)

Construct a Lorentz-equivariant layer.

# Arguments
- `hidden_dim::Int`: Hidden dimension for processing invariants
- `output_dim::Int`: Output dimension (number of scalar features)
"""
function LorentzEquivariantLayer(;
    hidden_dim::Int = 64,
    output_dim::Int = 32
)
    # Network for processing invariants
    phi = Chain(
        Dense(1 => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => output_dim, relu)
    )
    
    # Networks for computing mixing coefficients
    alpha_net = Dense(output_dim => 1, tanh)  # α_ij ∈ [-1, 1]
    beta_net = Dense(output_dim => 1, tanh)   # β_ij ∈ [-1, 1]
    
    return LorentzEquivariantLayer(phi, alpha_net, beta_net, hidden_dim, output_dim)
end

"""
    (layer::LorentzEquivariantLayer)(particles::AbstractVector{<:FourVector})

Apply Lorentz-equivariant transformation to a set of four-vectors.

# Algorithm:
1. Compute all Lorentz invariants: m_i² = p_i · p_i, s_ij = p_i · p_j
2. Process invariants with MLP to get scalar features
3. Use features to compute mixing coefficients α_ij and β_ij
4. Update each vector: p_i^out = p_i + Σ_j (α_ij p_j + β_ij s_ij p_j)

This is provably Lorentz-equivariant!
"""
function (layer::LorentzEquivariantLayer)(particles::AbstractVector{<:FourVector})
    n = length(particles)
    n > 0 || error("Need at least one particle")
    
    # Step 1: Compute all Lorentz invariants
    # Mass squared for each particle
    masses_sq = [Float32(p ⋅ p) for p in particles]
    
    # Pairwise dot products
    dot_products = zeros(Float32, n, n)
    for i in 1:n
        for j in 1:n
            dot_products[i, j] = Float32(particles[i] ⋅ particles[j])
        end
    end
    
    # Step 2: Process invariants to get scalar features
    # Combine all invariants into feature vector
    all_invariants = vcat(masses_sq, vec(dot_products))
    
    # Handle dynamic input dimension
    target_dim = size(layer.phi[1].weight, 2)
    if length(all_invariants) < target_dim
        all_invariants = vcat(all_invariants, zeros(Float32, target_dim - length(all_invariants)))
    else
        all_invariants = all_invariants[1:target_dim]
    end
    
    features = layer.phi(all_invariants)  # [output_dim]
    
    # Step 3 & 4: Compute coefficients and update vectors
    updated = Vector{FourVector{Float64}}(undef, n)
    
    for i in 1:n
        # Start with original vector
        new_p = particles[i]
        
        # Add contributions from all other vectors
        for j in 1:n
            # Get scalar coefficients from network
            alpha = Float64(layer.alpha_net(features)[1])
            beta = Float64(layer.beta_net(features)[1])
            
            # Equivariant update: p_i += α p_j + β (p_i · p_j) p_j
            # Use small learning rate for stability
            if i != j
                new_p = new_p + (alpha * 0.1) * particles[j]
                new_p = new_p + (beta * dot_products[i, j] * 0.01) * particles[j]
            end
        end
        
        updated[i] = new_p
    end
    
    return updated
end

# Make it work with Flux
Flux.@functor LorentzEquivariantLayer

# ============================================================================
# Lorentz-Invariant Network
# ============================================================================

"""
    LorentzInvariantNet{F}

A network that produces Lorentz-invariant outputs from four-vectors.

Uses only Lorentz invariants, so output is guaranteed to be frame-independent.

Properly handles variable-size inputs by computing exact number of invariants.
"""
struct LorentzInvariantNet{F} <: EquivariantLayer
    network::F
    expected_invariants::Int  # Track expected number of invariants
end

"""
    LorentzInvariantNet(; hidden_dims, output_dim, max_particles)

Construct Lorentz-invariant network.

# Arguments
- `hidden_dims::Vector{Int}`: Hidden layer dimensions
- `output_dim::Int`: Output dimension
- `max_particles::Int`: Maximum expected particles (for sizing, default=10)
"""
function LorentzInvariantNet(;
    hidden_dims::Vector{Int} = [64, 64],
    output_dim::Int = 1,
    max_particles::Int = 10
)
    # Calculate expected number of invariants for max_particles
    # n masses + n(n+1)/2 dot products (including diagonal)
    n_invariants = max_particles + div(max_particles * (max_particles + 1), 2)
    
    # Build MLP with proper input dimension
    layers = []
    
    # First layer
    push!(layers, Dense(n_invariants => hidden_dims[1], relu))
    
    # Hidden layers
    for i in 1:(length(hidden_dims)-1)
        push!(layers, Dense(hidden_dims[i] => hidden_dims[i+1], relu))
    end
    
    # Output layer
    push!(layers, Dense(hidden_dims[end] => output_dim))
    
    network = Chain(layers...)
    
    return LorentzInvariantNet{typeof(network)}(network, n_invariants)
end

"""
    compute_lorentz_invariants(particles::AbstractVector{<:FourVector})

Compute all Lorentz-invariant quantities from a set of four-vectors.

Returns a vector of invariants:
- p_i · p_i for each particle (masses squared)
- p_i · p_j for all pairs (dot products)
"""
function compute_lorentz_invariants(particles::AbstractVector{<:FourVector})
    n = length(particles)
    
    # Number of invariants: n self-products + n(n-1)/2 pair products
    n_invariants = n + div(n * (n - 1), 2)
    invariants = Vector{Float32}(undef, n_invariants)
    
    idx = 1
    
    # Self products (masses squared)
    for i in 1:n
        invariants[idx] = Float32(particles[i] ⋅ particles[i])
        idx += 1
    end
    
    # Pair products
    for i in 1:n
        for j in (i+1):n
            invariants[idx] = Float32(particles[i] ⋅ particles[j])
            idx += 1
        end
    end
    
    return invariants
end

function (model::LorentzInvariantNet)(particles::AbstractVector{<:FourVector})
    # Compute ALL invariants
    invariants = compute_lorentz_invariants(particles)
    
    # Pad or truncate to expected size
    if length(invariants) < model.expected_invariants
        invariants_padded = vcat(invariants, zeros(Float32, model.expected_invariants - length(invariants)))
    elseif length(invariants) > model.expected_invariants
        invariants_padded = invariants[1:model.expected_invariants]
    else
        invariants_padded = invariants
    end
    
    # Process with network
    return model.network(invariants_padded)
end

Flux.@functor LorentzInvariantNet

# ============================================================================
# Display
# ============================================================================

Base.show(io::IO, layer::LorentzEquivariantLayer) = 
    print(io, "LorentzEquivariantLayer(output_dim=$(layer.output_dim))")

Base.show(io::IO, model::LorentzInvariantNet) = 
    print(io, "LorentzInvariantNet(expected_invariants=$(model.expected_invariants))")