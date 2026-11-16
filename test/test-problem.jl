@testsnippet MockSystem begin
    struct TestCoordSystem <: AbstractTOVCoordSystem
        id::Int
    end

    struct TestInitialCondition <: AbstractInitialCondition
        coord_sys::TestCoordSystem
        arg::Float64
        direction::Symbol
    end

    TOVSolvers.coord_system(ic::TestInitialCondition) = ic.coord_sys
    TOVSolvers.initial_argument(ic::TestInitialCondition) = ic.arg
    TOVSolvers.default_direction(ic::TestInitialCondition) = ic.direction
end

@testitem "Construction and validation" setup = [MockSystem] begin

    params = TOVParameters(1.0, 0.5)
    coord = TestCoordSystem(1)
    ic = TestInitialCondition(coord, 0.0, :forward)

    prob = TOVProblem(params, coord, ic, (0.0, 10.0); direction = :auto)

    @test parameters(prob) === params
    @test coord_system(prob) === coord
    @test initial_condition(prob) === ic
    @test argument_span(prob) == (0.0, 10.0)
    @test integration_direction(prob) === :forward

    @test_throws ArgumentError TOVProblem(
        params,
        TestCoordSystem(2),
        ic,
        (0.0, 10.0),
    )

    @test_throws ArgumentError TOVProblem(
        params,
        coord,
        ic,
        (0.0, 10.0);
        direction = :sideways,
    )
end

@testitem "infer arg span from ic" setup = [MockSystem] begin

    coord = TestCoordSystem(2)
    ic = TestInitialCondition(coord, 10.0, :backward)

    prob = TOVProblem(TOVParameters(1.0, 0.5), ic, 0.0)


    @test argument_span(prob) == (10.0, 0.0)
    @test integration_direction(prob) === :backward
end
