"""
    TOVProblem(params, initial_condition, r_span; direction=:forward)

Defines a TOV problem to be solved.

# Fields
- `params::TOVParameters`: Physical parameters (omega, x)
- `initial_condition::AbstractInitialCondition`: Initial conditions
- `r_span::Tuple{Float64,Float64}`: Integration interval (r_start, r_end)
- `direction::Symbol`: Integration direction (:forward or :backward)

# Examples
```julia
# Problem starting from origin, integrating forward
params = TOVParameters(0.1, 3.0)
ic = RegularOrigin(m=0.0, m_prime=1.0)
prob = TOVProblem(params, ic, (0.0, 5.0))

# Problem starting from horizon, shooting backward
ic_horizon = AdmissibleCrossing(2.0, crossing=:plus, params=params)
prob_backward = TOVProblem(params, ic_horizon, (2.0, 0.0), direction=:backward)
```
"""
struct TOVProblem
    params::TOVParameters
    initial_condition::AbstractInitialCondition
    r_span::Tuple{Float64, Float64}
    direction::Symbol
    
    function TOVProblem(
        params::TOVParameters,
        initial_condition::AbstractInitialCondition,
        r_span::Tuple{Real, Real};
        direction::Symbol=:forward
    )
        direction in [:forward, :backward] || throw(ArgumentError("direction must be :forward or :backward"))
        
        r_start, r_end = Float64.(r_span)
        
        # Validate direction matches span
        if direction == :forward && r_end < r_start
            @warn "Forward integration requested but r_end < r_start. Setting direction to :backward"
            direction = :backward
        elseif direction == :backward && r_end > r_start
            @warn "Backward integration requested but r_end > r_start. Setting direction to :forward"
            direction = :forward
        end
        
        new(params, initial_condition, (r_start, r_end), direction)
    end
end

"""
    TOVProblem(params, initial_condition, r_end; direction=:auto)

Convenience constructor that automatically determines r_start from initial condition.

# Arguments
- `params`: TOV parameters
- `initial_condition`: Initial conditions
- `r_end`: Final radius for integration
- `direction`: :auto (default), :forward, or :backward

If direction is :auto, it is determined based on the initial condition type:
- RegularOrigin: integrates forward from 0 to r_end
- AdmissibleCrossing: integrates backward from r_h to r_end
"""
function TOVProblem(
    params::TOVParameters,
    initial_condition::AbstractInitialCondition,
    r_end::Real;
    direction::Symbol=:auto
)
    r_start = get_initial_radius(initial_condition)
    
    if direction == :auto
        # Determine direction based on initial condition type
        if initial_condition isa RegularOrigin
            direction = :forward
        elseif initial_condition isa AdmissibleCrossing
            direction = :backward
        end
    end
    
    return TOVProblem(params, initial_condition, (r_start, r_end), direction=direction)
end

# TODO:
# - add TOVProblem constructor, which takes r_span defaults from the initial_condition
# conditions
# - for RegularOrigin, the min_r can be infered from the ic, and the direction can default
# to :forward
# - for AdmissibleCrossing, the max_r and min_r can be infered from the ic and the direction can
# default to :backward
