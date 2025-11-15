"""
Test physics type system
"""

using Test
using ConstrainedML

@testset "FourVector" begin
    # Construction
    p = FourVector(10.0, 3.0, 4.0, 0.0)
    @test p.t == 10.0
    @test p.x == 3.0
    @test p.y == 4.0
    @test p.z == 0.0
    
    # Minkowski dot product
    mass_sq = p ⋅ p
    @test mass_sq ≈ 10.0^2 - 3.0^2 - 4.0^2 - 0.0^2
    @test mass_sq ≈ 75.0
    
    # Norm (mass squared, not mass!)
    @test norm2(p) ≈ 75.0  # Fixed: norm2 returns mass^2
    
    # Addition
    q = FourVector(5.0, 1.0, 0.0, 0.0)
    pq = p + q
    @test pq.t == 15.0
    @test pq.x == 4.0
    
    # Scalar multiplication
    p2 = 2.0 * p
    @test p2.t == 20.0
    @test p2.x == 6.0
end

@testset "GaugeField" begin
    # U(1) gauge field
    field = GaugeField(U1(), rand(ComplexF64, 8, 8))
    @test field.group isa U1
    @test size(field.data) == (8, 8)
end

@testset "LatticeConfig" begin
    # Lattice configuration (4D spacetime)
    config = LatticeConfig(U1(), (8, 8, 8, 4))  # Fixed: use 4D tuple
    @test config.group isa U1
    @test config.lattice_size == (8, 8, 8, 4)
end
