@testsnippet MockSolutionSetup begin
    struct TestCoordSystem <: TOVSolvers.AbstractTOVCoordSystem
        id::Int
    end

    TOVSolvers.coordinate_names(::TestCoordSystem) = (:t, :u1, :u2)

    struct TestInitialCondition <: TOVSolvers.AbstractInitialCondition
        coord_sys::TestCoordSystem
    end

    TOVSolvers.coord_system(ic::TestInitialCondition) = ic.coord_sys

    """
    Minimal stand-in for the DifferentialEquations solution protocol used by
    TOVSolution: `.t` and `solution[component, :]`.
    """
    struct FakeODESolution{T, U}
        t::Vector{T}
        u::Vector{U}
    end

    Base.getindex(sol::FakeODESolution, component::Int, ::Colon) =
        getindex.(sol.u, component)


end

@testitem "TOVSolution collection and coordinate access" setup = [MockSolutionSetup] begin
    coord = TestCoordSystem(1)
    ic = TestInitialCondition(coord)

    raw_solution = FakeODESolution(
        [0.0, 1.0, 2.0],
        [
            [0.0, 1.0],
            [1.0, 3.0],
            [2.0, 5.0],
        ],
    )

    params = TOVParameters(1.0, 0.5)

    sol = TOVSolution(
        raw_solution,
        params,
        coord,
        ic,
        true,
        false,
    )

    @test parameters(sol) === params
    @test coord_system(sol) === coord

    @test length(sol) == 3

    @test sol.t == [0.0, 1.0, 2.0]
    @test sol.u1 == [0.0, 1.0, 2.0]
    @test sol.u2 == [1.0, 3.0, 5.0]

    @test sol[2] == (1.0, 1.0, 3.0)
    @test collect(sol) == [
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 3.0),
        (2.0, 2.0, 5.0),
    ]
end
