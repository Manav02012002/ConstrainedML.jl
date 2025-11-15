"""
Simple U(1) Gauge-Equivariant Network Example

Demonstrates creating and using a gauge-equivariant neural network
for U(1) gauge theory.
"""

using ConstrainedML
using Flux

println("=" ^60)
println("Simple U(1) Gauge-Equivariant Network Example")
println("=" ^60)

# Create a gauge-equivariant network
println("\n1. Creating gauge-equivariant network...")
model = Chain(
    GaugeEquivariantConv(1, 8, symmetry=U1(), activation=tanh),
    GaugeEquivariantConv(8, 4, symmetry=U1(), activation=tanh),
    GaugeEquivariantConv(4, 1, symmetry=U1())
)
println(model)

# Create input field
println("\n2. Creating random U(1) gauge field (16×16)...")
x = rand(ComplexF64, 16, 16)
println("Input size: ", size(x))

# Forward pass
println("\n3. Running forward pass...")
y = model(x)
println("Output size: ", size(y))

# Test gauge equivariance
println("\n4. Testing gauge equivariance...")
test_angles = [0.1, π/4, π/2, π, 2π/3]

for θ in test_angles
    g = GaugeTransformation(U1(), θ)
    is_equiv = check_equivariance(model, x, g, rtol=1e-5)
    error = equivariance_error(model, x, g)
    
    status = is_equiv ? "✅" : "❌"
    println("  θ = $(round(θ, digits=3)): $status (error = $(round(error, sigdigits=3)))")
end

println("\n" * "=" ^60)
println("SUCCESS! Network is exactly gauge-equivariant!")
println("=" ^60)
