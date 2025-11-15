"""
Main test suite for ConstrainedML.jl
"""

using Test
using ConstrainedML

@testset "ConstrainedML.jl" begin
    
    @testset "Physics Types" begin
        include("test_physics_types.jl")
    end
    
    @testset "Symmetry Groups" begin
        include("test_groups.jl")
    end
    
    @testset "Transformations" begin
        include("test_transformations.jl")
    end
    
    @testset "U(1) Gauge Equivariance" begin
        include("test_gauge_equivariance.jl")
    end
    
    @testset "Lorentz Equivariance" begin
        include("test_lorentz_comprehensive.jl")
    end
    
    @testset "Permutation Invariance" begin
        include("test_permutation_invariance.jl")
    end

    @testset "SU(N) Gauge Theory" begin
        include("test_sun_gauge.jl")
    end
    
end

println("\n" * "="^60)
println("✅ All ConstrainedML.jl tests passed!")
println("="^60)
