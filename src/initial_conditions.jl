"""
Abstract type for initial conditions in the TOV equation.
"""
abstract type AbstractInitialCondition end

"""
    RegularOrigin(; m=0.0, m_prime=0.0)

Initial condition at the origin r=0 that ensures regularity.

For regularity at r=0, at least one of the following must hold:
- m(0) = 0 and m'(0) ≠ 0 (Case 1)
- m'(0) = 0 and m(0) ≠ 0 (Case 2)  
- m(0) = 0 and m'(0) = 0 (Case 3)

Both m and m' being non-zero will result in a singularity.

# Fields
- `r0::Float64`: Initial radius (always 0.0 for origin)
- `m::Float64`: Initial mass m(0)
- `m_prime::Float64`: Initial mass derivative m'(0)

# Examples
```julia
# Case 1: Zero mass, non-zero derivative
ic1 = RegularOrigin(m=0.0, m_prime=1.0)

# Case 2: Non-zero mass, zero derivative
ic2 = RegularOrigin(m=0.5, m_prime=0.0)

# Case 3: Both zero
ic3 = RegularOrigin(m=0.0, m_prime=0.0)
```
"""
struct RegularOrigin <: AbstractInitialCondition
    r0::Float64
    m::Float64
    m_prime::Float64
    
    function RegularOrigin(; m::Real=0.0, m_prime::Real=0.0)
        m_val = Float64(m)
        mp_val = Float64(m_prime)
        
        # Check for regularity condition
        if m_val != 0.0 && mp_val != 0.0
            @warn "Both m and m_prime are non-zero. This will create a singularity at r=0."
        end
        
        new(DEFAULT_R_ZERO, m_val, mp_val)
    end
end

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

"""
    get_initial_state(ic::RegularOrigin)

Get the initial state vector [m, m'] for a regular origin condition.
"""
function get_initial_state(ic::RegularOrigin)
    return [ic.m, ic.m_prime]
end

"""
    get_initial_radius(ic::AbstractInitialCondition)

Get the initial radius for integration.
"""
get_initial_radius(ic::RegularOrigin) = ic.r0
get_initial_radius(ic::AdmissibleCrossing) = ic.r_h
