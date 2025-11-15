"""
Test Lie group implementations
"""

using Test
using ConstrainedML

@testset "U(1) Group" begin
    g = U1()
    @test dim(g) == 1
    @test name(g) == "U(1)"
    @test !isnon_abelian(g)
end

@testset "SU(2) Group" begin
    g = SU2()
    @test dim(g) == 3
    @test name(g) == "SU(2)"
    @test isnon_abelian(g)
end

@testset "SU(3) Group" begin
    g = SU3()
    @test dim(g) == 8
    @test name(g) == "SU(3)"
    @test isnon_abelian(g)
end

@testset "Lorentz Group" begin
    g = LorentzGroup()
    @test dim(g) == 6  # 3 boosts + 3 rotations
    @test name(g) == "SO(1,3)"
end
