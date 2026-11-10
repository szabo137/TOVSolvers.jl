
### Regular Origin

struct RegularOrigin{T<:Real,C} <: AbstractInitialCondition
    coord_sys::C
    arg::T
    u1::T
    u2::T
    
    function RegularOrigin(coord_sys::C,arg::T,u1::T,u2::T; kwargs...) where {C<:AbstractTOVCoordSystem, T<:Real}
        _assert_regular_origin(coord_sys, arg, u1, u2; kwargs...)
        new{T,C}(coord_sys, arg, u1, u2)
    end
end

coord_system(ic::RegularOrigin) = ic.coord_sys
initial_state(ic::RegularOrigin) = [ic.u1, ic.u2]
initial_arg(ic::RegularOrigin) = ic.arg
default_direction(::RegularOrigin) = :forward

"""
    AdmissibleCrossing(r_h; crossing=:both, params::TOVParameters)

Initial condition at the horizon r=2m with admissible crossing.

The horizon is located at r_h where m(r_h) = r_h/2. The crossing slopes
m'(r_h) are computed from the regularity condition at the horizon.

# Fields
- `r_h::Float64`: Horizon radius
- `m_h::Float64`: Mass at horizon, equal to r_h/2
- `crossing::Symbol`: Which crossing to use (:plus, :minus, or :both)

# Examples
```julia
params = TOVParameters(0.1, 3.0)

# Use both crossings (default)
ic_both = AdmissibleCrossing(1.0, params=params)

# Use only the plus crossing
ic_plus = AdmissibleCrossing(1.0, crossing=:plus, params=params)

# Use only the minus crossing
ic_minus = AdmissibleCrossing(1.0, crossing=:minus, params=params)
```
"""
struct AdmissibleCrossing <: AbstractInitialCondition
  # TODO: refactor this!
    r_h::Float64
    m_h::Float64
    crossing::Symbol
    
    function AdmissibleCrossing(r_h::Real; crossing::Symbol=:both, params::Union{TOVParameters,Nothing}=nothing)
        r_h > 0 || throw(ArgumentError("r_h must be positive, got $r_h"))
        crossing in [:plus, :minus, :both] || throw(ArgumentError("crossing must be :plus, :minus, or :both"))
        
        m_h = r_h / 2.0
        
        # If params are provided, check if crossing is admissible
        if params !== nothing
            Δ = compute_discriminant(r_h, m_h, params)
            if Δ < 0
                throw(ArgumentError("No real admissible crossing exists at r_h=$r_h (discriminant=$Δ < 0)"))
            end
        end
        
        new(Float64(r_h), Float64(m_h), crossing)
    end
end

