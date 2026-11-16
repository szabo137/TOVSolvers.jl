module TOVSolvers

const DEFAULT_R_ZERO = 1.0e-16
const DEFAULT_HORIZON_TOLERANCE = 1.0e-10
const DEFAULT_ORIGIN_TOLERANCE = 1.0e-10

using OrdinaryDiffEq
using StaticArrays

# constants
export DEFAULT_ORIGIN_TOLERANCE, DEFAULT_HORIZON_TOLERANCE, DEFAULT_R_ZERO

# Interface types
export AbstractTOVCoordSystem
export AbstractTOVEquation
export AbstractInitialCondition
export AbstractTOVParameters
export AbstractTOVProblem
export AbstractTOVSolution

# Coordinate-system interface
export coordinate_names, reached_horizon, reached_origin
export create_horizon_callback, create_origin_callback

# Equation and initial-condition interface
export coord_system
export initial_state, initial_argument, default_direction

# Parameter, problem, and solution interface
export parameter_tuple
export parameters, initial_condition, argument_span, integration_direction
export final_argument, final_state, termination_reason

# Core concrete types
export TOVParameters, TOVProblem, TOVSolution
export RegularOrigin
export TOVSolverConfig
export TOVEquation

# Solving interface
export create_callbacks, solve, slow_manifold

# Utilities
export classify_solution

# Supported coordinate systems
export MMprimeSystem

# Include source files. `common.jl` defines the shared abstract types and
# interface-function declarations.
include("common.jl")
include("parameters.jl")
include("callbacks.jl")
include("initial_conditions.jl")
include("equations.jl")
include("problem.jl")
include("solver.jl")
include("solution.jl")

include("coordinate_systems/rmmp.jl")

end # module TOVSolvers
