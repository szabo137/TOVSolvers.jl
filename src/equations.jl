struct TOVEquation{C <: AbstractTOVCoordSystem, F} <: AbstractTOVEquation
    coord_system::C
    equation::F

    function TOVEquation(coord_system::C) where {C <: AbstractTOVCoordSystem}
        tov = _get_tov_equation(coord_system)

        return new{C, typeof(tov)}(coord_system, tov)
    end
end

@inline coord_system(eq::TOVEquation) = eq.coord_system

function (eq::TOVEquation)(du, u, params, arg)
    return eq.equation(du, u, params, arg)::Nothing
end
