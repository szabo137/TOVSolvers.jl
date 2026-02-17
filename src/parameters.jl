"""
    TOVParameters(omega, x)

Parameters for the TOV equation with linear equation of state.

# Fields
- `omega::Float64`: Positive constant, typically in range [10^-3, 1.0]
- `x::Float64`: Positive constant, typically 3.0

# Example
```julia
params = TOVParameters(0.1, 3.0)
```
"""
struct TOVParameters
    omega::Float64
    x::Float64
    
    function TOVParameters(omega::Real, x::Real)
        omega > 0 || throw(ArgumentError("omega must be positive, got $omega"))
        x > 0 || throw(ArgumentError("x must be positive, got $x"))
        new(Float64(omega), Float64(x))
    end
end

parameter_tuple(p::TOVParameters) = (p.omega, p.x)

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
