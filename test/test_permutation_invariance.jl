"""
Test permutation-invariant layers (DeepSets)
"""

using ConstrainedML
using Flux
using Test
using Random
using Statistics  # For mean function

println("="^60)
println("Testing Permutation-Invariant Layers (DeepSets)")
println("="^60)

# Test 1: Basic DeepSets with vectors
println("\n1. Testing DeepSets with simple vectors...")

model = DeepSets(
    phi = Chain(Dense(4 => 32, relu), Dense(32 => 32, relu)),
    rho = Chain(Dense(32 => 16, relu), Dense(16 => 2)),
    aggregation = sum
)

# Create a set of vectors
elements = [
    Float32[1.0, 2.0, 3.0, 4.0],
    Float32[5.0, 6.0, 7.0, 8.0],
    Float32[9.0, 10.0, 11.0, 12.0]
]

output = model(elements)
println("   Input: 3 vectors of dimension 4")
println("   Output: ", output)
println("   Output dimension: ", length(output))

# Test permutation invariance
println("\n2. Testing permutation invariance...")
Random.seed!(42)

diffs = Float64[]
for trial in 1:10
    # Random permutation
    perm = shuffle(1:length(elements))
    elements_shuffled = elements[perm]
    
    output_shuffled = model(elements_shuffled)
    
    diff = maximum(abs.(output .- output_shuffled))
    push!(diffs, diff)
    
    if trial == 1
        println("   Trial $trial: Original output = $output")
        println("   Trial $trial: Shuffled output = $output_shuffled")
        println("   Trial $trial: Difference = $diff")
    end
    
    @test output ≈ output_shuffled atol=1e-5
end
println("   ✅ All 10 permutations give identical output! (max diff: $(maximum(diffs)))")

# Test 3: Different aggregation functions
println("\n3. Testing different aggregation functions...")

for agg_fn in [sum, mean, maximum]
    model_agg = DeepSets(
        phi = Dense(3 => 16, relu),
        rho = Dense(16 => 1),
        aggregation = agg_fn
    )
    
    test_set = [Float32[1.0, 2.0, 3.0], Float32[4.0, 5.0, 6.0]]
    out1 = model_agg(test_set)
    
    # Shuffle
    test_set_shuffled = reverse(test_set)
    out2 = model_agg(test_set_shuffled)
    
    @test out1 ≈ out2 atol=1e-5
    println("   Aggregation=$agg_fn: ✅ Permutation-invariant")
end

# Test 4: DeepSets with FourVectors
println("\n4. Testing DeepSets with FourVectors (particle physics)...")

particle_model = ParticleSetClassifier(
    hidden_dim = 32,
    output_dim = 2,
    aggregation = sum
)

particles = [
    FourVector(10.0, 3.0, 0.0, 0.0),
    FourVector(5.0, 0.0, 4.0, 0.0),
    FourVector(8.0, 1.0, 1.0, 1.0)
]

output_particles = particle_model(particles)
println("   Input: 3 four-vectors")
println("   Output: ", output_particles)

# Test permutation invariance
particles_shuffled = shuffle(particles)
output_shuffled = particle_model(particles_shuffled)

diff_particles = maximum(abs.(output_particles .- output_shuffled))
println("   Original output:  ", output_particles)
println("   Shuffled output:  ", output_shuffled)
println("   Difference:       ", diff_particles)

@test output_particles ≈ output_shuffled atol=1e-5
println("   ✅ Permutation-invariant for particles!")

# Test 5: EquivariantDeepSets
println("\n5. Testing EquivariantDeepSets (outputs a set)...")

equivariant_model = EquivariantDeepSets(
    phi = Dense(4 => 16, relu),
    psi = Dense(32 => 4),  # 16 (features) + 16 (global context) = 32
    aggregation = mean
)

elements2 = [
    Float32[1.0, 2.0, 3.0, 4.0],
    Float32[5.0, 6.0, 7.0, 8.0],
    Float32[9.0, 10.0, 11.0, 12.0]
]

output_set = equivariant_model(elements2)
println("   Input: 3 vectors")
println("   Output: 3 vectors (updated)")
println("   Output[1]: ", output_set[1])

# Test equivariance: permute input → same permutation in output
perm = [3, 1, 2]
elements2_perm = elements2[perm]
output_set_perm = equivariant_model(elements2_perm)

# Check: output_set_perm should equal output_set[perm]
println("\n6. Testing equivariance property...")
eq_diffs = Float64[]
for i in 1:3
    original_idx = perm[i]
    diff = maximum(abs.(output_set_perm[i] .- output_set[original_idx]))
    push!(eq_diffs, diff)
    if i == 1
        println("   Position $i: output_perm[$i] ≈ output[$original_idx] (diff: $diff)")
    end
    @test output_set_perm[i] ≈ output_set[original_idx] atol=1e-4
end
println("   ✅ Equivariant to permutations! (max diff: $(maximum(eq_diffs)))")

# Test 7: Different set sizes
println("\n7. Testing with different set sizes...")
test_model = DeepSets(
    phi = Dense(4 => 16, relu),
    rho = Dense(16 => 2),
    aggregation = sum
)

for n in [1, 2, 5, 10, 20]
    test_elements = [Float32[rand(), rand(), rand(), rand()] for _ in 1:n]
    
    out1 = test_model(test_elements)
    out2 = test_model(shuffle(test_elements))
    
    @test out1 ≈ out2 atol=1e-5
end
println("   ✅ Works for sets of size 1, 2, 5, 10, 20")

println("\n" * "="^60)
println("✅ All DeepSets tests passed!")
println("="^60)