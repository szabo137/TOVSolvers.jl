module TOVSolvers

const DEFAULT_R_ZERO = 1e-16
const DEFAULT_HORIZON_TOLERANCE = 1e-10
const DEFAULT_ORIGIN_TOLERANCE = 1e-10

using DifferentialEquations
using StaticArrays

# Coordinate Systems
export AbstractTOVCoordSystem
export coordinate_names, reached_horizon, reached_origin
export create_horizon_callback, create_origin_callback

export AbstractInitialCondition
export initial_state, initial_arg

# Core types
export TOVParameters, TOVProblem, TOVSolution
export RegularOrigin, AdmissibleCrossing
export TOVSolverConfig

# Solving interface
export solve, find_admissible_crossings, slow_manifold

# Utilities
export compute_discriminant, compute_crossing_slopes
export classify_solution

# Supported Coordinate Systems
export MMprimeSystem

"""
    hi = hello_world()
A simple function to return "Hello, World!"
"""
function hello_world()
    return "Hello, World!"
end

# Include source files
include("common.jl")
include("parameters.jl")
include("coordinate_system.jl")
include("initial_conditions.jl")
include("equations.jl")
include("problem.jl")
include("solver.jl")
include("admissible_crossings.jl")
include("solution.jl")

include("coordinate_systems/rmmp.jl")

end # module TOVSolvers
