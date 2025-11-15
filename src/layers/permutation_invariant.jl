"""
    permutation_invariant.jl

Neural network layers that are permutation-invariant by construction.
"""

using Flux

# ============================================================================
# DeepSets Layer
# ============================================================================

"""
    DeepSets{F1, F2, A}

A permutation-invariant neural network layer (DeepSets architecture).

Processes unordered sets of inputs, guaranteed to be permutation-invariant.

# Fields
- `phi::F1`: Per-element feature extractor
- `rho::F2`: Set-level processor
- `aggregation::A`: Aggregation function (sum, mean, max)

# Mathematical Guarantee
For any permutation σ and input set {x₁, ..., xₙ}:
    layer({x_σ(1), ..., x_σ(n)}) = layer({x₁, ..., xₙ})

# Reference
Zaheer et al. "Deep Sets" (NeurIPS 2017)

# Examples
```julia
# Jet classification
model = DeepSets(
    phi = Chain(Dense(4 => 64, relu), Dense(64 => 64, relu)),
    rho = Chain(Dense(64 => 32, relu), Dense(32 => 2)),
    aggregation = sum
)

particles = [
    [10.0, 3.0, 0.0, 0.0],  # 4-momentum components
    [5.0, 0.0, 4.0, 0.0],
    [8.0, 1.0, 1.0, 1.0]
]

classification = model(particles)  # [prob_quark, prob_gluon]
```
"""
struct DeepSets{F1, F2, A} <: EquivariantLayer
    phi::F1          # Per-element network
    rho::F2          # Aggregation network
    aggregation::A   # Aggregation function
end

"""
    DeepSets(; phi, rho, aggregation=sum)

Construct a DeepSets (permutation-invariant) layer.

# Arguments
- `phi`: Neural network for per-element processing
- `rho`: Neural network for set-level processing
- `aggregation`: Aggregation function (sum, mean, maximum, minimum)

The output is guaranteed to be permutation-invariant.
"""
function DeepSets(;
    phi::F1,
    rho::F2,
    aggregation::A = sum
) where {F1, F2, A}
    return DeepSets{F1, F2, A}(phi, rho, aggregation)
end

"""
    (layer::DeepSets)(elements::AbstractVector)

Apply DeepSets to a set of elements.

# Algorithm:
1. Apply phi to each element independently
2. Aggregate features using aggregation function
3. Process aggregated features with rho
"""
function (layer::DeepSets)(elements::AbstractVector)
    isempty(elements) && error("Cannot process empty set")
    
    # Step 1: Per-element processing
    features = [layer.phi(e) for e in elements]
    
    # Step 2: Aggregation (permutation-invariant!)
    if layer.aggregation === sum
        aggregated = sum(features)
    elseif layer.aggregation === mean
        aggregated = sum(features) ./ length(features)
    elseif layer.aggregation === maximum
        aggregated = reduce((a, b) -> max.(a, b), features)
    elseif layer.aggregation === minimum
        aggregated = reduce((a, b) -> min.(a, b), features)
    else
        # Custom aggregation function
        aggregated = layer.aggregation(features)
    end
    
    # Step 3: Set-level processing
    output = layer.rho(aggregated)
    
    return output
end

# Make it work with Flux
Flux.@functor DeepSets

# ============================================================================
# Equivariant DeepSets (outputs a set)
# ============================================================================

"""
    EquivariantDeepSets{F1, F2, A}

DeepSets variant that outputs a set (permutation-equivariant).

Each output element can depend on all input elements, but the overall
transformation is equivariant to permutations.

# Examples
```julia
# Particle flow: update each particle based on all others
model = EquivariantDeepSets(
    phi = Dense(4 => 32, relu),
    psi = Dense(64 => 4),  # 32 (features) + 32 (global) = 64
    aggregation = mean
)

particles_in = [...]
particles_out = model(particles_in)  # Same length, updated
```
"""
struct EquivariantDeepSets{F1, F2, A} <: EquivariantLayer
    phi::F1          # Per-element feature extractor
    psi::F2          # Per-element update network
    aggregation::A   # Aggregation function
end

function EquivariantDeepSets(;
    phi::F1,
    psi::F2,
    aggregation::A = mean
) where {F1, F2, A}
    return EquivariantDeepSets{F1, F2, A}(phi, psi, aggregation)
end

"""
    (layer::EquivariantDeepSets)(elements::AbstractVector)

Apply equivariant DeepSets transformation.

Each output depends on all inputs, but transformation is equivariant.
"""
function (layer::EquivariantDeepSets)(elements::AbstractVector)
    isempty(elements) && error("Cannot process empty set")
    
    # Extract features from each element
    features = [layer.phi(e) for e in elements]
    
    # Compute global context (permutation-invariant)
    if layer.aggregation === sum
        global_context = sum(features)
    elseif layer.aggregation === mean
        global_context = sum(features) ./ length(features)
    elseif layer.aggregation === maximum
        global_context = reduce((a, b) -> max.(a, b), features)
    else
        global_context = layer.aggregation(features)
    end
    
    # Update each element using global context + local features
    # Concatenate: [local_features, global_context]
    outputs = similar(features)
    
    for i in 1:length(elements)
        # Concatenate local features with global context
        combined = vcat(features[i], global_context)
        outputs[i] = layer.psi(combined)
    end
    
    return outputs
end

Flux.@functor EquivariantDeepSets

# ============================================================================
# Convenience: DeepSets for FourVectors
# ============================================================================

"""
    ParticleSetClassifier(; hidden_dim, output_dim, aggregation=sum)

Convenience constructor for classifying sets of particles.

# Examples
```julia
# Jet tagging: particles → quark/gluon classification
model = ParticleSetClassifier(
    hidden_dim = 64,
    output_dim = 2,  # [quark, gluon]
    aggregation = sum
)

particles = [
    FourVector(10.0, 3.0, 0.0, 0.0),
    FourVector(5.0, 0.0, 4.0, 0.0),
    ...
]

probs = model(particles)  # [p_quark, p_gluon]
```
"""
function ParticleSetClassifier(;
    hidden_dim::Int = 64,
    output_dim::Int = 2,
    aggregation = sum
)
    # Per-particle network
    phi = Chain(
        # Convert FourVector to array, then process
        Dense(4 => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu)
    )
    
    # Set-level network
    rho = Chain(
        Dense(hidden_dim => hidden_dim÷2, relu),
        Dense(hidden_dim÷2 => output_dim)
    )
    
    return DeepSets(phi=phi, rho=rho, aggregation=aggregation)
end

# Helper to convert FourVector to array for network input
function (ds::DeepSets)(particles::AbstractVector{<:FourVector})
    # Convert particles to arrays
    arrays = [[p.t, p.x, p.y, p.z] for p in particles]
    return ds(arrays)
end

# ============================================================================
# Display
# ============================================================================

Base.show(io::IO, layer::DeepSets) = 
    print(io, "DeepSets(aggregation=$(layer.aggregation))")

Base.show(io::IO, layer::EquivariantDeepSets) = 
    print(io, "EquivariantDeepSets(aggregation=$(layer.aggregation))")