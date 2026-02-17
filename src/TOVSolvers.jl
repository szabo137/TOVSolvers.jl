module TOVSolvers

const DEFAULT_R_ZERO = 1e-16
const DEFAULT_HORIZON_TOLERANCE = 1e-10
const DEFAULT_ORIGIN_TOLERANCE = 1e-10

using DifferentialEquations
using StaticArrays

# Core types
export TOVParameters, TOVProblem, TOVSolution
export RegularOrigin, AdmissibleCrossing
export TOVSolverConfig

# Solving interface
export solve, find_admissible_crossings, slow_manifold

# Utilities
export compute_discriminant, compute_crossing_slopes
export classify_solution
export is_at_horizon, compute_H, is_on_slow_manifold, get_initial_state

# Include source files
include("parameters.jl")
include("initial_conditions.jl")
include("equations.jl")
include("problem.jl")
include("solver.jl")
include("admissible_crossings.jl")
include("solution.jl")
include("callbacks.jl")

end # module TOVSolvers
