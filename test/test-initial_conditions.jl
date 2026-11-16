@testsnippet MockCoordSystems begin

    struct MockCoordSystem <: AbstractTOVCoordSystem end

    TOVSolvers.coordinate_names(::MockCoordSystem) = (:u1, :u2)

    TOVSolvers.reached_horizon(
        ::MockCoordSystem,
        arg,
        state;
        kwargs...,
    ) = false

    TOVSolvers.reached_origin(
        ::MockCoordSystem,
        arg,
        state;
        kwargs...,
    ) = iszero(arg)

    TOVSolvers.assert_regular_origin(
        ::MockCoordSystem,
        arg,
        u1,
        u2;
        kwargs...,
    ) = iszero(arg) || throw(ArgumentError("mock origin must have arg == 0"))


end

@testitem "RegularOrigin stores a valid initial condition" setup = [MockCoordSystems] begin
    coord = MockCoordSystem()
    ic = RegularOrigin(coord, 0.0, 1.0, 2.0)

    @test coord_system(ic) === coord
    @test initial_argument(ic) === 0.0
    @test initial_state(ic) == [1.0, 2.0]
    @test default_direction(ic) === :forward
end
