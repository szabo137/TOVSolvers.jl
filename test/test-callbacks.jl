@testsnippet MockCallbackSetup begin
  
    mutable struct TestCoordSystem <: TOVSolvers.AbstractTOVCoordSystem
        requested_callbacks::Vector{Symbol}
    end

    struct TestProblem{C,P} <: TOVSolvers.AbstractTOVProblem
        params::P
        coord_sys::C
        direction::Symbol
    end

    TOVSolvers.parameters(prob::TestProblem) = prob.params
    TOVSolvers.coord_system(prob::TestProblem) = prob.coord_sys
    TOVSolvers.integration_direction(prob::TestProblem) = prob.direction

    # The callback is intentionally inert. This test verifies callback
    # selection, not event behavior.
    function TOVSolvers.create_horizon_callback(
        coord::TestCoordSystem,
        params,
        tolerance,
    )
        push!(coord.requested_callbacks, :horizon)

        return TOVSolvers.ContinuousCallback(
            (u, t, integrator) -> 1.0,
            integrator -> nothing,
        )
    end

    function TOVSolvers.create_origin_callback(
        coord::TestCoordSystem,
        params,
        tolerance,
    )
        push!(coord.requested_callbacks, :origin)

        return TOVSolvers.ContinuousCallback(
            (u, t, integrator) -> 1.0,
            integrator -> nothing,
        )
    end
end

@testitem "integration direction" setup = [MockCallbackSetup] begin

    params = TOVSolvers.TOVParameters(1.0, 0.5)
    config = TOVSolvers.TOVSolverConfig()

    forward_coord = TestCoordSystem(Symbol[])
    forward_prob = TestProblem(params, forward_coord, :forward)

    callbacks = TOVSolvers.create_callbacks(forward_prob, config)

    @test callbacks isa TOVSolvers.CallbackSet
    @test forward_coord.requested_callbacks == [:horizon]

    backward_coord = TestCoordSystem(Symbol[])
    backward_prob = TestProblem(params, backward_coord, :backward)

    TOVSolvers.create_callbacks(backward_prob, config)

    @test backward_coord.requested_callbacks == [:origin, :horizon]

    invalid_coord = TestCoordSystem(Symbol[])
    invalid_prob = TestProblem(params, invalid_coord, :sideways)

    @test_throws ArgumentError TOVSolvers.create_callbacks(invalid_prob, config)
end
