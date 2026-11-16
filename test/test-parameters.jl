@testitem "TOVParameters" begin
    @testset "construction and promotion" begin
        p = TOVParameters(1, 0.5)

        @test p isa TOVParameters{Float64}
        @test p.omega === 1.0
        @test p.x === 0.5
        @test parameter_tuple(p) == (1.0, 0.5)

        q = TOVParameters(big"1.0", 0.5)

        @test q isa TOVParameters{BigFloat}
        @test q.omega == big"1.0"
        @test q.x == big"0.5"
    end

    @testset "validation" begin
        @test_throws ArgumentError TOVParameters(0.0, 1.0)
        @test_throws ArgumentError TOVParameters(-1.0, 1.0)
        @test_throws ArgumentError TOVParameters(1.0, 0.0)
        @test_throws ArgumentError TOVParameters(1.0, -1.0)

        @test_throws ArgumentError TOVParameters(NaN, 1.0)
        @test_throws ArgumentError TOVParameters(1.0, NaN)
        @test_throws ArgumentError TOVParameters(Inf, 1.0)
        @test_throws ArgumentError TOVParameters(1.0, Inf)
    end
end
