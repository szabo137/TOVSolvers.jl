@testsnippet MockCoordSystem begin
  struct TestCoordSystem <: TOVSolvers.AbstractTOVCoordSystem end

    function TOVSolvers._get_tov_equation(::TestCoordSystem)
        function rhs!(du, u, params, arg)
            du[1] = params.omega * u[1] + arg
            du[2] = u[2] - params.x
            return nothing
        end

        return rhs!
    end
end


@testitem "TOVEquation delegates to the coordinate-system RHS" setup = [MockCoordSystem] begin

    coord = TestCoordSystem()
    equation = TOVSolvers.TOVEquation(coord)
    params = TOVSolvers.TOVParameters(2.0, 5.0)

    u = [3.0, 7.0]
    du = zeros(2)

    result = equation(du, u, params, 0.5)

    @test TOVSolvers.coord_system(equation) === coord
    @test result === nothing
    @test du == [6.5, 2.0]
end
