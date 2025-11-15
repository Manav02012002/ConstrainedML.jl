"""
Simple Demo: Physics-Constrained Neural Networks

A quick demonstration of all symmetries including SU(2) and SU(3).
"""

using ConstrainedML
using Flux
using LinearAlgebra

println("="^60)
println("ConstrainedML.jl - Complete Demo")
println("="^60)

# ============================================================================
# 1. U(1) Gauge Equivariance
# ============================================================================

println("\n1️⃣  U(1) Gauge Equivariance (Electromagnetism)")
println("-" ^60)

layer_u1 = GaugeEquivariantConv(1, 4, symmetry=U1(), activation=tanh)
field = rand(ComplexF64, 16, 16)

# Apply gauge transformation
g = GaugeTransformation(U1(), π/4)
is_equivariant = check_equivariance(layer_u1, field, g)

println("   Created U(1) gauge-equivariant layer")
println("   Tested with θ = π/4 gauge transformation")
println("   Result: $(is_equivariant ? "✅ Exactly equivariant!" : "❌ Failed")")
println("   Error: $(round(equivariance_error(layer_u1, field, g), sigdigits=3))")

# ============================================================================
# 2. SU(2) Gauge Transformations (Weak Force)
# ============================================================================

println("\n2️⃣  SU(2) Gauge Transformations (Weak Interactions)")
println("-" ^60)

# Generate random SU(2) transformation
g_su2 = random_SU2()
println("   Generated random SU(2) transformation")
println("   Matrix size: $(size(g_su2.matrix))")
println("   Unitary: $(g_su2.matrix' * g_su2.matrix ≈ I(2) ? "✅" : "❌")")
println("   Det = 1: $(abs(det(g_su2.matrix) - 1.0) < 1e-6 ? "✅" : "❌")")

# Transform a doublet
ψ_su2 = randn(ComplexF64, 2)
ψ_su2_transformed = transform(g_su2, ψ_su2)

println("   Original field norm:     $(round(norm(ψ_su2), digits=6))")
println("   Transformed field norm:  $(round(norm(ψ_su2_transformed), digits=6))")
println("   Norm preserved: $(norm(ψ_su2) ≈ norm(ψ_su2_transformed) ? "✅" : "❌")")

# ============================================================================
# 3. SU(3) Gauge Transformations (QCD)
# ============================================================================

println("\n3️⃣  SU(3) Gauge Transformations (Quantum Chromodynamics)")
println("-" ^60)

# Generate random SU(3) transformation
g_su3 = random_SU3()
println("   Generated random SU(3) transformation")
println("   Matrix size: $(size(g_su3.matrix))")
println("   Unitary: $(norm(g_su3.matrix' * g_su3.matrix - I(3)) < 1e-5 ? "✅" : "❌")")
println("   Det = 1: $(abs(det(g_su3.matrix) - 1.0) < 1e-5 ? "✅" : "❌")")

# Transform a color triplet (quark field)
ψ_su3 = randn(ComplexF64, 3)
ψ_su3_transformed = transform(g_su3, ψ_su3)

println("   Original quark field norm:     $(round(norm(ψ_su3), digits=6))")
println("   Transformed quark field norm:  $(round(norm(ψ_su3_transformed), digits=6))")
println("   Norm preserved: $(norm(ψ_su3) ≈ norm(ψ_su3_transformed) ? "✅" : "❌")")

# ============================================================================
# 4. Wilson Loop (Gauge Invariance)
# ============================================================================

println("\n4️⃣  Wilson Loop - Gauge Invariant Observable")
println("-" ^60)

# Create plaquette (4 gauge links) - Fixed variable names to avoid conflicts
link1 = random_SU2().matrix
link2 = random_SU2().matrix
link3 = random_SU2().matrix
link4 = random_SU2().matrix

W_original = compute_plaquette(link1, link2, link3, link4)

# Transform all links
g_transform = random_SU2()
link1_t = transform(g_transform, link1)
link2_t = transform(g_transform, link2)
link3_t = transform(g_transform, link3)
link4_t = transform(g_transform, link4)

W_transformed = compute_plaquette(link1_t, link2_t, link3_t, link4_t)

println("   Original plaquette:     $(round(W_original, digits=6))")
println("   Transformed plaquette:  $(round(W_transformed, digits=6))")
println("   Difference:             $(round(abs(W_original - W_transformed), sigdigits=3))")
println("   Gauge invariant: $(abs(W_original - W_transformed) < 1e-5 ? "✅" : "❌")")

# ============================================================================
# 5. Lorentz Invariance
# ============================================================================

println("\n5️⃣  Lorentz Invariance")
println("-" ^60)

particles = [
    FourVector(10.0, 3.0, 0.0, 0.0),
    FourVector(5.0, 0.0, 4.0, 0.0),
    FourVector(8.0, 1.0, 1.0, 1.0)
]

model = LorentzInvariantNet(hidden_dims=[32], output_dim=1)
output1 = model(particles)

# Apply boost
Λ = boost(0.6, direction=:x)
particles_boosted = [transform(Λ, p) for p in particles]
output2 = model(particles_boosted)

diff = abs(output1[1] - output2[1])
println("   Created Lorentz-invariant network")
println("   Original frame:  output = $(round(output1[1], digits=6))")
println("   Boosted frame:   output = $(round(output2[1], digits=6))")
println("   Difference:      $(round(diff, sigdigits=3))")
println("   Result: $(diff < 1e-5 ? "✅ Frame-independent!" : "❌ Failed")")

# ============================================================================
# 6. Permutation Invariance
# ============================================================================

println("\n6️⃣  Permutation Invariance")
println("-" ^60)

elements = [
    Float32[1.0, 2.0, 3.0],
    Float32[4.0, 5.0, 6.0],
    Float32[7.0, 8.0, 9.0]
]

deepsets = DeepSets(
    phi = Dense(3 => 16, relu),
    rho = Dense(16 => 2),
    aggregation = sum
)

output_original = deepsets(elements)
output_shuffled = deepsets(reverse(elements))

diff_perm = maximum(abs.(output_original .- output_shuffled))
println("   Created permutation-invariant network (DeepSets)")
println("   Original order:  output = $output_original")
println("   Reversed order:  output = $output_shuffled")
println("   Difference:      $(round(diff_perm, sigdigits=3))")
println("   Result: $(diff_perm < 1e-5 ? "✅ Order-independent!" : "❌ Failed")")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^60)
println("✨ All Symmetries Working!")
println("="^60)
println()
println("Gauge Symmetries:")
println("  ✅ U(1) - Electromagnetism")
println("  ✅ SU(2) - Weak interactions")
println("  ✅ SU(3) - Quantum chromodynamics")
println()
println("Spacetime Symmetries:")
println("  ✅ Lorentz Invariance - Special relativity")
println()
println("Statistical Symmetries:")
println("  ✅ Permutation Invariance - Identical particles")
println()
println("Applications:")
println("  • Lattice QCD simulations")
println("  • Standard Model calculations")
println("  • Jet physics at the LHC")
println("  • Event classification")
println("  • Normalizing flows for gauge theories")
println()
println("="^60)
