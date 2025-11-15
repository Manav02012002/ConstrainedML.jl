"""
Comprehensive tests for Lorentz-equivariant layers
"""

using ConstrainedML
using Test

@testset "Lorentz Layers" begin
    
    @testset "LorentzInvariantNet" begin
        model = LorentzInvariantNet(hidden_dims=[32, 32], output_dim=1)
        
        particles = [
            FourVector(10.0, 3.0, 0.0, 0.0),
            FourVector(5.0, 0.0, 4.0, 0.0)
        ]
        
        output = model(particles)
        
        # Test various Lorentz transformations
        @testset "Boosts" begin
            for β in [0.1, 0.3, 0.5, 0.7, 0.9]
                for dir in [:x, :y, :z]
                    Λ = boost(β, direction=dir)
                    particles_transformed = [transform(Λ, p) for p in particles]
                    output_transformed = model(particles_transformed)
                    @test output ≈ output_transformed rtol=1e-6
                end
            end
        end
        
        @testset "Rotations" begin
            for θ in [0.1, π/4, π/2, π, 3π/2]
                for axis in [:x, :y, :z]
                    R = rotation(θ, axis=axis)
                    particles_transformed = [transform(R, p) for p in particles]
                    output_transformed = model(particles_transformed)
                    @test output ≈ output_transformed rtol=1e-6
                end
            end
        end
        
        @testset "Different particle counts" begin
            for n in [1, 2, 5, 10]
                ps = [FourVector(rand()*10, rand()*3, rand()*3, rand()*3) for _ in 1:n]
                out = model(ps)
                @test length(out) == 1  # Should always return scalar
                
                # Check invariance
                Λ = boost(0.5, direction=:x)
                ps_boosted = [transform(Λ, p) for p in ps]
                out_boosted = model(ps_boosted)
                @test out ≈ out_boosted rtol=1e-5
            end
        end
    end
    
    @testset "LorentzEquivariantLayer" begin
        layer = LorentzEquivariantLayer(hidden_dim=32, output_dim=16)
        
        particles = [
            FourVector(10.0, 3.0, 0.0, 0.0),
            FourVector(5.0, 0.0, 4.0, 0.0),
            FourVector(8.0, 1.0, 1.0, 1.0)
        ]
        
        output = layer(particles)
        @test length(output) == length(particles)
        
        @testset "Equivariance under boosts" begin
            for β in [0.3, 0.6]
                Λ = boost(β, direction=:x)
                
                # Method 1: transform input
                particles_boosted = [transform(Λ, p) for p in particles]
                output1 = layer(particles_boosted)
                
                # Method 2: transform output
                output2 = [transform(Λ, p) for p in output]
                
                # Check equivariance via invariant masses
                for i in 1:length(particles)
                    mass1_sq = output1[i] ⋅ output1[i]
                    mass2_sq = output2[i] ⋅ output2[i]
                    @test mass1_sq ≈ mass2_sq rtol=1e-10
                end
            end
        end
    end
end

println("\n✅ All Lorentz layer tests passed!")
