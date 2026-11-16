@testitem "coordinate names" begin
    coord = MMprimeSystem()

    @test coordinate_names(coord) == (:r, :m, :m_prime)
end

@testitem "TOV equation" begin
    coord = MMprimeSystem()
    params = TOVParameters(2.0, 4.0)

    equation = TOVEquation(coord)
    du = zeros(2)

    # r = 3, m = 1, m′ = 2 gives H = -336 and denominator = 12.
    result = equation(du, [1.0, 2.0], params, 3.0)

    @test result === nothing
    @test du == [2.0, -28.0]
end

@testitem "regular-origin validation" begin
    coord = MMprimeSystem()

    @test_nowarn RegularOrigin(
        coord,
        0.0,
        0.0,
        0.0,
    )

    @test_logs (:warn, r"Both m and m_prime are non-zero") begin
        RegularOrigin(coord, 0.0, 1.0, 1.0)
    end
end

@testitem "callbacks" begin
    import OrdinaryDiffEq 

    coord = MMprimeSystem()
    params = TOVParameters(1.0, 1.0)
    tolerance = 1e-4

    # Keep m = u[1] = 1 constant while r advances from 1 to 3.
    # The MMprime horizon callback should trigger near r = 2.
    function constant_mass!(du, u, p, r)
        du[1] = 0.0
        du[2] = 0.0
        return nothing
    end

    ode_prob = OrdinaryDiffEq.ODEProblem(
        constant_mass!,
        [1.0, 0.0],
        (1.0, 3.0),
    )

    callback = create_horizon_callback(
        coord,
        params,
        tolerance,
    )

    ode_sol = OrdinaryDiffEq.solve(
        ode_prob,
        OrdinaryDiffEq.Tsit5();
        callback,
        abstol=1e-10,
        reltol=1e-10,
    )

    # Current condition:
    # r - 2m - tolerance * r = 0
    # hence r = 2 / (1 - tolerance), with m = 1.
    @test isapprox(ode_sol.t[end],  2.0 / (1.0 - tolerance),atol=1e-8)
    @test ode_sol.t[end] < 3.0
end
