"""
Comprehensive tests for U(1) gauge-equivariant layers
"""

using Test
using ConstrainedML
using Flux

@testset "Single Layer Equivariance" begin
    layer = GaugeEquivariantConv(1, 8, symmetry=U1(), activation=tanh)
    x = rand(ComplexF64, 16, 16)
    
    # Test multiple angles
    for θ in [0.1, π/6, π/4, π/3, π/2, 2π/3, π, 4π/3, 3π/2, 2π]
        g = GaugeTransformation(U1(), θ)
        @test check_equivariance(layer, x, g, rtol=1e-10)
    end
end

@testset "Multi-Layer Equivariance" begin
    model = Chain(
        GaugeEquivariantConv(1, 8, symmetry=U1(), activation=tanh),
        GaugeEquivariantConv(8, 4, symmetry=U1(), activation=tanh),
        GaugeEquivariantConv(4, 1, symmetry=U1())
    )
    
    x = rand(ComplexF64, 16, 16)
    
    for θ in [0.1, π/4, π/2, π, 2π/3]
        g = GaugeTransformation(U1(), θ)
        @test check_equivariance(model, x, g, rtol=1e-10)
    end
end

@testset "Edge Cases" begin
    # Single pixel
    layer = GaugeEquivariantConv(1, 4, symmetry=U1())
    x = rand(ComplexF64, 1, 1)
    g = GaugeTransformation(U1(), π/4)
    @test check_equivariance(layer, x, g, rtol=1e-10)
    
    # Large field
    x_large = rand(ComplexF64, 64, 64)
    @test check_equivariance(layer, x_large, g, rtol=1e-10)
end

@testset "Gradient Flow" begin
    # Test that gradients flow through the layer
    layer = GaugeEquivariantConv(1, 4, symmetry=U1(), activation=tanh)
    x = rand(ComplexF64, 8, 8)
    
    loss(m, x) = sum(abs2, m(x))
    
    # Compute gradient
    grads = Flux.gradient(m -> loss(m, x), layer)
    
    # Gradients should exist and be non-zero
    @test grads[1] !== nothing
    @test !iszero(grads[1].weight)
end
