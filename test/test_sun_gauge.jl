"""
Test SU(2) and SU(3) gauge transformations
"""

using Test
using ConstrainedML
using LinearAlgebra

@testset "SU(2) Transformations" begin
    # Generate random SU(2) transformation
    g = random_SU2()
    
    @test size(g.matrix) == (2, 2)
    @test g.matrix' * g.matrix ≈ I(2) rtol=1e-6  # Unitary
    @test det(g.matrix) ≈ 1.0 rtol=1e-6          # Det = 1
    
    # Test on field in fundamental representation
    ψ = randn(ComplexF64, 2)
    ψ_transformed = transform(g, ψ)
    
    @test length(ψ_transformed) == 2
    @test norm(ψ_transformed) ≈ norm(ψ) rtol=1e-6  # Preserves norm
end

@testset "SU(3) Transformations" begin
    # Generate random SU(3) transformation
    g = random_SU3()
    
    @test size(g.matrix) == (3, 3)
    @test g.matrix' * g.matrix ≈ I(3) rtol=1e-5  # Unitary
    @test abs(det(g.matrix) - 1.0) < 1e-5        # Det = 1
    
    # Test on field (color triplet)
    ψ = randn(ComplexF64, 3)
    ψ_transformed = transform(g, ψ)
    
    @test length(ψ_transformed) == 3
    @test norm(ψ_transformed) ≈ norm(ψ) rtol=1e-6
end

@testset "Gauge Link Transformation" begin
    # SU(2) gauge link
    g1 = random_SU2()
    g2 = random_SU2()
    
    U_link = g1.matrix  # Some SU(2) matrix
    
    # Transform: U → g U g†
    U_transformed = transform(g2, U_link)
    
    @test size(U_transformed) == (2, 2)
    @test U_transformed' * U_transformed ≈ I(2) rtol=1e-5
end

@testset "Plaquette Gauge Invariance" begin
    # Create 4 SU(2) links forming a plaquette
    U1 = random_SU2().matrix
    U2 = random_SU2().matrix
    U3 = random_SU2().matrix
    U4 = random_SU2().matrix
    
    # Compute plaquette
    W_original = compute_plaquette(U1, U2, U3, U4)
    
    # Apply gauge transformation
    g = random_SU2()
    U1_t = transform(g, U1)
    U2_t = transform(g, U2)
    U3_t = transform(g, U3)
    U4_t = transform(g, U4)
    
    W_transformed = compute_plaquette(U1_t, U2_t, U3_t, U4_t)
    
    # Plaquette should be gauge-invariant!
    @test W_original ≈ W_transformed rtol=1e-5
end
