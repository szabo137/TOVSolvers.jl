"""
    TOVSolverConfig

Configuration options for the TOV solver.

# Fields
- `ode_algorithm`: ODE solver algorithm from DifferentialEquations.jl
- `abstol::Float64`: Absolute tolerance for ODE solver
- `reltol::Float64`: Relative tolerance for ODE solver
- `horizon_tolerance::Float64`: Relative tolerance for horizon detection (|r-2m|/r < tol)
- `origin_tolerance::Float64`: Absolute tolerance for origin detection (r < tol)
- `maxiters::Int`: Maximum number of iterations
- `save_everystep::Bool`: Whether to save at every integration step

# Examples
```julia
# High precision configuration
config = TOVSolverConfig(
    ode_algorithm=Vern9(),
    abstol=1e-10,
    reltol=1e-10
)

# Fast configuration with lower precision
config_fast = TOVSolverConfig(
    ode_algorithm=Tsit5(),
    abstol=1e-6,
    reltol=1e-6
)
```
"""
struct TOVSolverConfig
    ode_algorithm
    abstol::Float64
    reltol::Float64
    horizon_tolerance::Float64
    origin_tolerance::Float64
    maxiters::Int
    save_everystep::Bool
    
    function TOVSolverConfig(;
        ode_algorithm=Vern9(),
        abstol::Real=1e-10,
        reltol::Real=1e-10,
        horizon_tolerance::Real=DEFAULT_HORIZON_TOLERANCE,
        origin_tolerance::Real=DEFAULT_ORIGIN_TOLERANCE,
        maxiters::Int=10^6,
        save_everystep::Bool=true
    )
        new(
            ode_algorithm,
            Float64(abstol),
            Float64(reltol),
            Float64(horizon_tolerance),
            Float64(origin_tolerance),
            maxiters,
            save_everystep
        )
    end
end

"""
    solve(prob::TOVProblem, config::TOVSolverConfig=TOVSolverConfig())

Solve a TOV problem with the specified configuration.

# Arguments
- `prob::TOVProblem`: Problem definition
- `config::TOVSolverConfig`: Solver configuration (optional, uses defaults if not provided)

# Returns
- For RegularOrigin: Single `TOVSolution`
- For AdmissibleCrossing with :both: Vector of two `TOVSolution` objects

# Examples
```julia
# Solve from origin with default settings
params = TOVParameters(0.1, 3.0)
ic = RegularOrigin(m=0.0, m_prime=1.0)
prob = TOVProblem(params, ic, 5.0)
sol = solve(prob)

# Solve with custom configuration
config = TOVSolverConfig(abstol=1e-12, reltol=1e-12)
sol = solve(prob, config)

# Solve admissible crossing (returns two solutions)
ic_crossing = AdmissibleCrossing(2.0, crossing=:both, params=params)
prob_crossing = TOVProblem(params, ic_crossing, 0.0)
sols = solve(prob_crossing)  # Returns vector of two solutions
```
"""
function solve(prob::TOVProblem, config::TOVSolverConfig=TOVSolverConfig())
    # Get initial states (may be multiple for AdmissibleCrossing with :both)
    if prob.initial_condition isa RegularOrigin
        u0_list = [get_initial_state(prob.initial_condition)]
    else  # AdmissibleCrossing
        u0_list = get_initial_state(prob.initial_condition, prob.params)
    end
    
    # Solve for each initial state
    solutions = TOVSolution[]
    
    for u0 in u0_list
        sol = solve_single(prob, u0, config)
        push!(solutions, sol)
    end
    
    # Return single solution or vector
    if length(solutions) == 1
        return solutions[1]
    else
        return solutions
    end
end

"""
    solve_single(prob::TOVProblem, u0, config::TOVSolverConfig)

Solve a single TOV problem with given initial state.

Internal function used by `solve`.
"""
function solve_single(prob::TOVProblem, u0, config::TOVSolverConfig)
    # Create callbacks
    callbacks = create_callbacks(prob, config)
    
    # Define ODE problem
    ode_prob = ODEProblem(tov_second_order!, u0, prob.r_span, prob.params)

    ode_sol = DifferentialEquations.solve(
        ode_prob,
        config.ode_algorithm;
        abstol=config.abstol,
        reltol=config.reltol,
        callback=callbacks,
        maxiters=config.maxiters,
        save_everystep=config.save_everystep
    )
    
    # Determine if horizon was hit
    hit_horizon = false
    reached_origin = false
    
    if length(ode_sol.t) > 0
        r_final = ode_sol.t[end]
        m_final = ode_sol[1,end]
        
        # Check if at horizon
        if is_at_horizon(r_final, m_final, rtol=100*config.horizon_tolerance) || prob.initial_condition isa AdmissibleCrossing
            hit_horizon = true
        end
        
        # Check if at origin
        if r_final < 100*config.origin_tolerance || prob.initial_condition isa RegularOrigin
            reached_origin = true
        end
    end
    return TOVSolution(
        ode_sol,
        prob.params,
        prob.initial_condition,
        hit_horizon,
        reached_origin,
    )
end

"""
    solve(params::TOVParameters, ic::AbstractInitialCondition, r_end::Real, config::TOVSolverConfig=TOVSolverConfig())

Convenience method to solve without explicitly creating a TOVProblem.

# Example
```julia
params = TOVParameters(0.1, 3.0)
ic = RegularOrigin(m=0.0, m_prime=1.0)
sol = solve(params, ic, 5.0)
```
"""
function solve(
    params::TOVParameters,
    ic::AbstractInitialCondition,
    r_end::Real,
    config::TOVSolverConfig=TOVSolverConfig()
)
    prob = TOVProblem(params, ic, r_end)
    return solve(prob, config)
end
