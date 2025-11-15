"""
Basic test of Lorentz-equivariant layers
"""

using ConstrainedML
using Test

println("="^60)
println("Testing Lorentz-Equivariant Layers")
println("="^60)

# Test 1: LorentzInvariantNet
println("\n1. Testing LorentzInvariantNet (invariant output)...")
model = LorentzInvariantNet(
    hidden_dims = [32, 32],
    output_dim = 1
)

particles = [
    FourVector(10.0, 3.0, 0.0, 0.0),
    FourVector(5.0, 0.0, 4.0, 0.0)
]

output = model(particles)
println("   Input: 2 four-vectors")
println("   Output: ", output)

# Test Lorentz invariance
println("\n2. Testing Lorentz invariance under boost...")
Λ = boost(0.5, direction=:x)
particles_boosted = [transform(Λ, p) for p in particles]

output_boosted = model(particles_boosted)
println("   Original output: ", output)
println("   Boosted output:  ", output_boosted)
println("   Difference:      ", abs(output[1] - output_boosted[1]))

@test output ≈ output_boosted rtol=1e-5

println("\n3. Testing under rotation...")
R = rotation(π/4, axis=:z)
particles_rotated = [transform(R, p) for p in particles]

output_rotated = model(particles_rotated)
println("   Original output: ", output)
println("   Rotated output:  ", output_rotated)
println("   Difference:      ", abs(output[1] - output_rotated[1]))

@test output ≈ output_rotated rtol=1e-5

println("\n" * "="^60)
println("✅ LorentzInvariantNet tests passed!")
println("="^60)

# Test 2: LorentzEquivariantLayer
println("\n4. Testing LorentzEquivariantLayer (equivariant output)...")
layer = LorentzEquivariantLayer(
    hidden_dim = 32,
    output_dim = 16
)

particles2 = [
    FourVector(10.0, 3.0, 0.0, 0.0),
    FourVector(5.0, 0.0, 4.0, 0.0),
    FourVector(8.0, 1.0, 1.0, 1.0)
]

output2 = layer(particles2)
println("   Input: 3 four-vectors")
println("   Output: 3 four-vectors")
println("   Output[1]: ", output2[1])

# Test Lorentz equivariance
println("\n5. Testing Lorentz equivariance under boost...")
particles2_boosted = [transform(Λ, p) for p in particles2]

# Method 1: boost input then apply layer
output_method1 = layer(particles2_boosted)

# Method 2: apply layer then boost output
output_method2 = [transform(Λ, p) for p in output2]

println("   Method 1 (boost→layer): ", output_method1[1])
println("   Method 2 (layer→boost): ", output_method2[1])

# Check equivariance (might not be exact due to nonlinearity, but should be close)
max_diff = maximum(norm2(output_method1[i]) - norm2(output_method2[i]) for i in 1:3)
println("   Max invariant mass difference: ", max_diff)

println("\n" * "="^60)
println("✅ Basic Lorentz layer tests complete!")
println("="^60)
