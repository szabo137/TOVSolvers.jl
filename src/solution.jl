"""
    TOVSolution

Container for the solution of a TOV problem.

# Fields
- `r`: Radius values
- `m`: Mass values
- `m_prime`: Mass derivative values
- `params::TOVParameters`: Parameters used
- `initial_condition::AbstractInitialCondition`: Initial conditions used
- `hit_horizon::Bool`: Whether the solution reached the horizon
- `reached_origin::Bool`: Whether the solution reached the origin (for backward integration)
- `retcode`: Return code from the ODE solver
- `sol`: Raw ODE solution object (optional, for advanced use)

# Methods
- `length(sol)`: Number of solution points
- `sol[i]`: Get (r, m, m') at index i
"""
struct TOVSolution{S}
    ode_sol::S # Store the raw ODE solution for interpolation if needed
    params::TOVParameters
    initial_condition::AbstractInitialCondition
    hit_horizon::Bool
    reached_origin::Bool
    
    function TOVSolution(
        ode_sol::S,
        params::TOVParameters,
        initial_condition::AbstractInitialCondition,
        hit_horizon::Bool,
        reached_origin::Bool,
    ) where S
        
        new{S}(
            ode_sol,
            params,
            initial_condition,
            hit_horizon,
            reached_origin,
        )
    end
end

function Base.getproperty(sol::TOVSolution, s::Symbol)
    if s==:r
        return sol.ode_sol.t
    elseif s==:m 
        return sol.ode_sol[1,:]
    elseif s==:m_prime 
        return sol.ode_sol[2,:]
    else
        return getfield(sol,s)
    end
end

# Convenience methods
Base.length(sol::TOVSolution) = length(sol.r)
Base.getindex(sol::TOVSolution, i::Int) = (sol.r[i], sol.m[i], sol.m_prime[i])
Base.iterate(sol::TOVSolution, state=1) = state > length(sol) ? nothing : (sol[state], state+1)

"""
    classify_solution(sol::TOVSolution)

Classify a TOV solution based on its properties.

# Returns
A symbol indicating the solution type:
- `:regular_origin_to_horizon`: Started at origin, reached horizon
- `:regular_origin_incomplete`: Started at origin, did not reach horizon
- `:horizon_crossing_to_origin`: Started at horizon, reached origin
- `:horizon_crossing_to_horizon`: Started at horizon, reached horizon again
- `:horizon_crossing_incomplete`: Started at horizon, incomplete
"""
function classify_solution(sol::TOVSolution)
    ic = sol.initial_condition
    
    if ic isa RegularOrigin
        if sol.hit_horizon
            return :regular_origin_to_horizon
        else
            return :regular_origin_incomplete
        end
    elseif ic isa AdmissibleCrossing
        if sol.reached_origin
            return :horizon_crossing_to_origin
        elseif sol.hit_horizon
            return :horizon_crossing_to_horizon
        else
            return :horizon_crossing_incomplete
        end
    end
    
    return :unknown
end

"""
    is_on_slow_manifold(sol::TOVSolution; rtol=1e-3)

Check if the solution stays on or near the slow manifold.

# Arguments
- `sol`: TOV solution
- `rtol`: Relative tolerance for slow manifold detection

# Returns
- Tuple (is_on, max_deviation) where is_on is true if the solution stays within tolerance
"""
function is_on_slow_manifold(sol::TOVSolution; rtol::Real=1e-3)
    max_deviation = 0.0
    
    for i in 1:length(sol)
        r = sol.r[i]
        m = sol.m[i]
        
        if r > 0
            m_slow = slow_manifold(r, sol.params)
            rel_dev = abs(m - m_slow) / (abs(m_slow) + 1e-14)
            max_deviation = max(max_deviation, rel_dev)
        end
    end
    
    return (max_deviation < rtol, max_deviation)
end

"""
    Base.show(io::IO, sol::TOVSolution)

Custom display for TOVSolution.
"""
function Base.show(io::IO, sol::TOVSolution)
    classification = classify_solution(sol)
    n_points = length(sol)
    r_range = (sol.r[1], sol.r[end])
    m_range = (minimum(sol.m), maximum(sol.m))
    
    println(io, "TOVSolution with $n_points points")
    println(io, "  Classification: $classification")
    println(io, "  Parameters: omega=$(sol.params.omega), x=$(sol.params.x)")
    println(io, "  r ∈ [$(r_range[1]), $(r_range[2])]")
    println(io, "  m ∈ [$(m_range[1]), $(m_range[2])]")
    println(io, "  Hit horizon: $(sol.hit_horizon)")
    println(io, "  Reached origin: $(sol.reached_origin)")
    #println(io, "  Return code: $(sol.retcode)")
end
