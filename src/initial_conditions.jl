### Regular Origin

struct RegularOrigin{T <: Real, C} <: AbstractInitialCondition
    coord_sys::C
    arg::T
    u1::T
    u2::T

    function RegularOrigin(coord_sys::C, arg::T, u1::T, u2::T; kwargs...) where {C <: AbstractTOVCoordSystem, T <: Real}
        assert_regular_origin(coord_sys, arg, u1, u2; kwargs...)
        return new{T, C}(coord_sys, arg, u1, u2)
    end
end

coord_system(ic::RegularOrigin) = ic.coord_sys
initial_state(ic::RegularOrigin) = [ic.u1, ic.u2]
initial_argument(ic::RegularOrigin) = ic.arg
default_direction(::RegularOrigin) = :forward
