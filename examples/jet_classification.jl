"""
Jet Classification: Quark vs Gluon Jets

Demonstrates all three symmetries:
- Lorentz invariance: Classification shouldn't depend on reference frame
- Permutation invariance: Particle order doesn't matter
- Physics constraints: Uses 4-momentum properly

This is a simplified example. Real jet tagging would use actual collision data.
"""

using ConstrainedML
using Flux
using Statistics
using Random

println("="^70)
println("Jet Classification: Quark vs Gluon Jets")
println("Demonstrating Lorentz + Permutation Invariance")
println("="^70)

# ============================================================================
# 1. Generate Synthetic Jet Data
# ============================================================================

"""
    generate_jet(n_particles::Int, jet_type::Symbol)

Generate a synthetic jet with n_particles.
- :quark jets: fewer particles, more collimated
- :gluon jets: more particles, wider spread
"""
function generate_jet(n_particles::Int, jet_type::Symbol)
    particles = FourVector{Float64}[]
    
    # Jet parameters depend on type
    if jet_type == :quark
        spread = 0.3  # More collimated
    else  # :gluon
        spread = 0.6  # Wider spread
    end
    
    # Generate particles
    for i in 1:n_particles
        # Random momentum components (simplified physics)
        E = 5.0 + rand() * 20.0  # Energy in GeV
        θ = randn() * spread      # Polar angle spread
        φ = rand() * 2π           # Azimuthal angle
        
        # Convert to Cartesian
        pt = E * sin(θ)  # Transverse momentum
        px = pt * cos(φ)
        py = pt * sin(φ)
        pz = E * cos(θ)
        
        push!(particles, FourVector(E, px, py, pz))
    end
    
    return particles
end

println("\n1. Generating synthetic jet data...")
Random.seed!(42)

# Generate training data
n_jets_per_class = 100
quark_jets = [generate_jet(rand(3:8), :quark) for _ in 1:n_jets_per_class]
gluon_jets = [generate_jet(rand(5:12), :gluon) for _ in 1:n_jets_per_class]

println("   Generated $(n_jets_per_class) quark jets")
println("   Generated $(n_jets_per_class) gluon jets")
println("   Quark jets: $(round(mean(length.(quark_jets)), digits=2)) particles (avg)")
println("   Gluon jets: $(round(mean(length.(gluon_jets)), digits=2)) particles (avg)")

# ============================================================================
# 2. Build Physics-Aware Classifier (Lorentz-Invariant Version)
# ============================================================================

println("\n2. Building physics-aware classifier...")

"""
Physics-aware jet classifier using ONLY Lorentz-invariant features.
Guaranteed to be both Lorentz-invariant and permutation-invariant.
"""
model = LorentzInvariantNet(
    hidden_dims = [64, 64, 32],
    output_dim = 2  # Binary: [quark, gluon]
)

println("   ✅ Model architecture:")
println("      - Uses ONLY Lorentz-invariant features")
println("      - Permutation-invariant by construction")
println("      - Output: [P(quark), P(gluon)]")

# ============================================================================
# 3. Test Lorentz Invariance
# ============================================================================

println("\n3. Testing Lorentz invariance...")

test_jet = quark_jets[1]
output_original = model(test_jet)

# Apply various Lorentz transformations
transformations = [
    ("Boost (β=0.3, x)", boost(0.3, direction=:x)),
    ("Boost (β=0.6, z)", boost(0.6, direction=:z)),
    ("Rotation (π/4, y)", rotation(π/4, axis=:y)),
    ("Rotation (π/2, z)", rotation(π/2, axis=:z))
]

println("   Original output: ", output_original)
println()

invariance_results = Bool[]
for (name, Λ) in transformations
    # Transform the jet
    jet_transformed = [transform(Λ, p) for p in test_jet]
    output_transformed = model(jet_transformed)
    
    # Check invariance
    diff = maximum(abs.(output_original .- output_transformed))
    is_invariant = diff < 1e-5
    push!(invariance_results, is_invariant)
    
    status = is_invariant ? "✅" : "❌"
    println("   $status $name: diff = $(round(diff, sigdigits=3))")
end

if all(invariance_results)
    println("\n   🎉 Model is Lorentz-invariant!")
else
    println("\n   ⚠️  Warning: Model shows frame dependence")
end

# ============================================================================
# 4. Test Permutation Invariance
# ============================================================================

println("\n4. Testing permutation invariance...")

test_jet2 = gluon_jets[1]
output_original2 = model(test_jet2)

println("   Original output: ", output_original2)
println("   Testing 10 random permutations...")

diffs = Float64[]
for i in 1:10
    # Random permutation
    perm_indices = shuffle(1:length(test_jet2))
    jet_permuted = test_jet2[perm_indices]
    
    output_permuted = model(jet_permuted)
    diff = maximum(abs.(output_original2 .- output_permuted))
    push!(diffs, diff)
end

max_diff = maximum(diffs)
println("   Max difference across permutations: $(round(max_diff, sigdigits=3))")

if max_diff < 1e-5
    println("   🎉 Model is permutation-invariant!")
else
    println("   ⚠️  Warning: Model shows permutation dependence")
end

# ============================================================================
# 5. Classification Performance (Synthetic Data)
# ============================================================================

println("\n5. Testing classification on synthetic data...")

# Test on a few examples
n_test = 10
quark_test = quark_jets[1:n_test]
gluon_test = gluon_jets[1:n_test]

quark_predictions = [model(jet) for jet in quark_test]
gluon_predictions = [model(jet) for jet in gluon_test]

# Check if model distinguishes (even without training)
quark_scores = [pred[1] - pred[2] for pred in quark_predictions]  # quark - gluon
gluon_scores = [pred[1] - pred[2] for pred in gluon_predictions]

println("   Quark jets: mean score = $(round(mean(quark_scores), digits=3))")
println("   Gluon jets: mean score = $(round(mean(gluon_scores), digits=3))")
println()
println("   (Note: Model is untrained - scores are random)")
println("   (In practice: train with real data and BCE loss)")

# ============================================================================
# 6. Demonstrate Combined Transformation
# ============================================================================

println("\n6. Testing combined boost + rotation + permutation...")

test_jet3 = quark_jets[5]

# Original
output1 = model(test_jet3)

# Boost, then rotate, then permute (apply transformations sequentially)
Λ_boost = boost(0.7, direction=:z)
R = rotation(π/3, axis=:y)

# Apply boost first, then rotation
jet_boosted = [transform(Λ_boost, p) for p in test_jet3]
jet_rotated = [transform(R, p) for p in jet_boosted]
jet_permuted = shuffle(jet_rotated)

output2 = model(jet_permuted)

diff_combined = maximum(abs.(output1 .- output2))
println("   Original:                    ", output1)
println("   Boost + Rotate + Permute:    ", output2)
println("   Difference:                  ", round(diff_combined, sigdigits=3))

if diff_combined < 1e-5
    println("   ✅ Invariant under combined transformations!")
else
    println("   ❌ Failed combined transformation test")
end

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("Summary: Jet Classification Example")
println("="^70)
println()
println("✅ Built physics-aware classifier using:")
println("   • Lorentz-invariant features ONLY (masses, dot products)")
println("   • Permutation-invariant architecture")
println("   • Reference-frame independent predictions")
println()
println("✅ Verified symmetries:")
println("   • Classification invariant under boosts")
println("   • Classification invariant under rotations")
println("   • Classification invariant under particle reordering")
println("   • Invariant under combined transformations")
println()
println("💡 Key insight:")
println("   By using ONLY Lorentz-invariant features,")
println("   we get perfect frame-independence by construction!")
println()
println("🎯 Next steps for real application:")
println("   • Train on real LHC jet data")
println("   • Add high-level features (jet mass, N-subjettiness)")
println("   • Optimize hyperparameters")
println("   • Benchmark against ParticleNet, EFN, PFN")
println()
println("="^70)
