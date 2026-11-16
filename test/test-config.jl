@testitem "TOVSolverConfig defaults" begin
    config = TOVSolverConfig()

    @test config.abstol == 1.0e-10
    @test config.reltol == 1.0e-10
    @test config.horizon_tolerance == DEFAULT_HORIZON_TOLERANCE
    @test config.origin_tolerance == DEFAULT_ORIGIN_TOLERANCE
    @test config.maxiters == 1_000_000
    @test config.save_everystep === true
end

@testitem "TOVSolverConfig custom values and conversion" begin
    algorithm = TOVSolvers.Tsit5()

    config = TOVSolverConfig(
        ode_algorithm = algorithm,
        abstol = Float32(1.0e-6),
        reltol = 1 // 1_000_000,
        horizon_tolerance = 1.0e-8,
        origin_tolerance = 1.0e-12,
        maxiters = 42,
        save_everystep = false,
    )

    @test config.ode_algorithm === algorithm
    @test isapprox(config.abstol, 1.0e-6)
    @test isapprox(config.reltol, 1.0e-6)
    @test isapprox(config.horizon_tolerance, 1.0e-8)
    @test isapprox(config.origin_tolerance, 1.0e-12)
    @test config.maxiters == 42
    @test config.save_everystep === false

    # TODO: change this, if config becomes type preserving
    @test config.abstol isa Float64
    @test config.reltol isa Float64
    @test config.horizon_tolerance isa Float64
    @test config.origin_tolerance isa Float64
end

#=
@testitem "Invalid input" begin
  @test_throws ArgumentError TOVSolverConfig(abstol=0.0)
  @test_throws ArgumentError TOVSolverConfig(reltol=-1.0)
  @test_throws ArgumentError TOVSolverConfig(horizon_tolerance=Inf)
  @test_throws ArgumentError TOVSolverConfig(maxiters=0)
end
=#
