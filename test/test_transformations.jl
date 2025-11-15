"""
Test transformations (boosts, rotations, gauge)
"""

using Test
using ConstrainedML
using LinearAlgebra

@testset "Lorentz Boosts" begin
    p = FourVector(10.0, 0.0, 0.0, 0.0)
    
    # Boost in x-direction
    for β in [0.1, 0.3, 0.5, 0.7, 0.9]
        Λ = boost(β, direction=:x)
        p_boosted = transform(Λ, p)
        
        # Mass should be invariant
        @test p ⋅ p ≈ p_boosted ⋅ p_boosted rtol=1e-10
        
        # Energy should increase
        @test p_boosted.t > p.t
    end
    
    # Test all directions
    for dir in [:x, :y, :z]
        Λ = boost(0.5, direction=dir)
        p_boosted = transform(Λ, p)
        @test p ⋅ p ≈ p_boosted ⋅ p_boosted rtol=1e-10
    end
end

@testset "Rotations" begin
    p = FourVector(10.0, 3.0, 0.0, 0.0)
    
    # Test all axes
    for axis in [:x, :y, :z]
        for θ in [0.1, π/4, π/2, π]
            R = rotation(θ, axis=axis)
            p_rotated = transform(R, p)
            
            # Mass invariant
            @test p ⋅ p ≈ p_rotated ⋅ p_rotated rtol=1e-10
            
            # Time component unchanged
            @test p_rotated.t ≈ p.t rtol=1e-10
            
            # Spatial magnitude unchanged
            @test sqrt(p.x^2 + p.y^2 + p.z^2) ≈ 
                  sqrt(p_rotated.x^2 + p_rotated.y^2 + p_rotated.z^2) rtol=1e-10
        end
    end
end

@testset "U(1) Gauge Transformations" begin
    # Test gauge transformation on complex field
    ψ = rand(ComplexF64, 8, 8)
    
    for θ in [0.1, π/4, π/2, π]
        g = GaugeTransformation(U1(), θ)
        ψ_transformed = transform(g, ψ)
        
        # Magnitude should be preserved
        @test abs.(ψ) ≈ abs.(ψ_transformed) rtol=1e-10
        
        # Phase should shift by θ
        # Check by transforming back
        g_inv = GaugeTransformation(U1(), -θ)
        ψ_back = transform(g_inv, ψ_transformed)
        @test ψ ≈ ψ_back rtol=1e-10
    end
end

@testset "Composition of Transformations" begin
    p = FourVector(10.0, 3.0, 0.0, 0.0)
    
    # Boost then rotate
    Λ1 = boost(0.5, direction=:x)
    R = rotation(π/4, axis=:z)
    
    p1 = transform(R, transform(Λ1, p))
    
    # Should preserve mass
    @test p ⋅ p ≈ p1 ⋅ p1 rtol=1e-10
    
    # Multiple transformations
    for _ in 1:10
        β = rand() * 0.9
        θ = rand() * 2π
        dir = rand([:x, :y, :z])
        axis = rand([:x, :y, :z])
        
        Λ = boost(β, direction=dir)
        R = rotation(θ, axis=axis)
        
        p_transformed = transform(R, transform(Λ, p))
        
        @test p ⋅ p ≈ p_transformed ⋅ p_transformed rtol=1e-9
    end
end
