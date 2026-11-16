# TODO:
# - consider using type parameters instead of Float64
# - consider adding input validation, e.g.
#   - non-negative abstol and reltol
#   - forbidden numbers like NaN, Inf for tolerances
#   - non-zero maxiters
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
            ode_algorithm = Vern9(),
            abstol::Real = 1.0e-10,
            reltol::Real = 1.0e-10,
            horizon_tolerance::Real = DEFAULT_HORIZON_TOLERANCE,
            origin_tolerance::Real = DEFAULT_ORIGIN_TOLERANCE,
            maxiters::Int = Int(1.0e6),
            save_everystep::Bool = true
        )
        return new(
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

function solve(prob::TOVProblem, config::TOVSolverConfig = TOVSolverConfig())
    # Get initial states (may be multiple for AdmissibleCrossing with :both)
    ic = initial_conditions(prob)
    if ic isa RegularOrigin
        u0_list = [initial_state(ic)]
    else  # AdmissibleCrossing
        u0_list = initial_state(ic, parameters(prob))
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

function solve_single(prob::TOVProblem, u0, config::TOVSolverConfig)
    coord_sys = coord_system(prob)

    # Create TOV equation
    tov_eq! = TOVEquation(coord_sys)

    # Create callbacks
    callbacks = create_callbacks(prob, config)

    # Define ODE problem
    ode_prob = ODEProblem(tov_eq!, u0, [arg_span(prob)...], parameters(prob))

    ode_sol = DifferentialEquations.solve(
        ode_prob,
        config.ode_algorithm;
        abstol = config.abstol,
        reltol = config.reltol,
        callback = callbacks,
        maxiters = config.maxiters,
        save_everystep = config.save_everystep
    )

    # Determine if horizon was hit
    hit_horizon = false
    reached_origin = false

    if length(ode_sol.t) > 0
        arg_final = ode_sol.t[end]
        u1_final = ode_sol[1, end]
        u2_final = ode_sol[1, end]

        # Check if at horizon
        if reached_horizon(coord_sys, arg_final, u1_final, u2_final; atol = config.horizon_tolerance) #|| initial_conditions(prob) isa AdmissibleCrossing
            hit_horizon = true
        end

        # Check if at origin
        if initial_conditions(prob) isa RegularOrigin || reached_horizon(coord_sys, arg_final, u1_final, u2_final; tol = config.origin_tolerance)
            reached_origin = true
        end
    end
    return TOVSolution(
        ode_sol,
        parameters(prob),
        coord_system(prob),
        initial_conditions(prob),
        hit_horizon,
        reached_origin,
    )
end
