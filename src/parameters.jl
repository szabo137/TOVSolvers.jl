# TODO:
# - implement Base.show
# - consider Base.@kw_only
struct TOVParameters <: AbstractTOVParameters
    omega::Float64
    x::Float64
    
    function TOVParameters(omega::Real, x::Real)
        omega > 0 || throw(ArgumentError("omega must be positive, got $omega"))
        x > 0 || throw(ArgumentError("x must be positive, got $x"))
        new(Float64(omega), Float64(x))
    end
end

parameter_tuple(p::TOVParameters) = (p.omega, p.x)


# TODO: is this necessary?
# Convenient accessor functions
Base.getproperty(p::TOVParameters, s::Symbol) = getfield(p, s)

"""
    slow_manifold(r, params::TOVParameters)

Compute the mass value on the slow manifold at radius r.
The slow manifold is defined as m(r) = (omega/x) * r³.
"""
function slow_manifold(r::Real, params::TOVParameters)
    return (params.omega / params.x) * r^3
end
