"""
    create_horizon_callback(params::TOVParameters, horizon_tolerance::Float64)

Create a callback that terminates integration when approaching the horizon r = 2m.

The callback triggers when |r - 2m|/r < horizon_tolerance.

# Arguments
- `params`: TOV parameters
- `horizon_tolerance`: Relative tolerance for horizon detection

# Returns
- A DifferentialEquations.jl callback that terminates at the horizon
"""
function create_horizon_callback(params::TOVParameters, horizon_tolerance::Float64)
    condition(u, r, integrator) = begin
        m = u[1]
        # Return negative when approaching horizon (triggers when crossing zero)
        return r - 2*m - horizon_tolerance * r
    end
    
    affect!(integrator) = begin
        @info "Horizon reached at r=$(integrator.t), m=$(integrator.u[1]). Terminating integration."
        terminate!(integrator)
    end
    
    return ContinuousCallback(condition, affect!)
end

"""
    create_negative_mass_callback()

Create a callback that terminates integration if mass becomes negative.

Physical solutions should have m ≥ 0.
"""
function create_negative_mass_callback()
    condition(u, r, integrator) = u[1]  # Triggers when m crosses zero
    
    affect!(integrator) = begin
        @warn "Mass became negative at r=$(integrator.t). Terminating integration."
        terminate!(integrator)
    end
    
    return ContinuousCallback(condition, affect!)
end

"""
    create_origin_callback(params::TOVParameters, origin_tolerance::Float64)

Create a callback that terminates integration when reaching the origin r = 0.

Used for backward integration from the horizon.

# Arguments
- `params`: TOV parameters
- `origin_tolerance`: Absolute tolerance for origin detection

# Returns
- A DifferentialEquations.jl callback that terminates at origin
"""
function create_origin_callback(params::TOVParameters, origin_tolerance::Float64)
    condition(u, r, integrator) = r - origin_tolerance
    
    affect!(integrator) = begin
        m = integrator.u[1]
        m_prime = integrator.u[2]
        @info "Origin reached at r=$(integrator.t), m=$m, m'=$m_prime. Terminating integration."
        terminate!(integrator)
    end
    
    return ContinuousCallback(condition, affect!)
end

"""
    create_callbacks(prob::TOVProblem, config::TOVSolverConfig)

Create all necessary callbacks for a TOV problem based on the problem setup and solver config.

# Arguments
- `prob`: TOV problem definition
- `config`: Solver configuration

# Returns
- CallbackSet containing all relevant callbacks
"""
function create_callbacks(prob::TOVProblem, config)
    callbacks = []
    
    # Always add negative mass callback
    push!(callbacks, create_negative_mass_callback())
    
    if prob.direction == :forward
        # For forward integration from origin, add horizon callback
        push!(callbacks, create_horizon_callback(prob.params, config.horizon_tolerance))
    elseif prob.direction == :backward
        # For backward integration from horizon, add origin callback
        push!(callbacks, create_origin_callback(prob.params, config.origin_tolerance))
        
        # Also add horizon callback in case we reach it again
        push!(callbacks, create_horizon_callback(prob.params, config.horizon_tolerance))
    end
    
    return CallbackSet(callbacks...)
end
