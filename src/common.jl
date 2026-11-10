### Coordinate Systems

abstract type AbstractTOVCoordSystem end


"""
    coordinate_names(::AbstractTOVCoordSystems)::NTuple{N,Symbol}

Interface function: return tuple of symbols denoting the names of the coordinates for the given coordinate system. 
"""
function coordinate_names end

"""
    reached_horizon(::AbstractTOVCoordSystem, arg_final, u1_final, u2_final; tol = 1e-10)
    reached_horizon(::AbstractTOVSolution)

Interface function: return true if solution reached the horizon in a given coordinate system, or TOV solution.
"""
function reached_horizon end

"""
    reached_origin(::AbstractTOVCoordSystem, arg_final, u1_final, u2_final; tol = 1e-10)
    reached_horizon(::AbstractTOVSolution)

Interface function: return true if solution reached the origin in a given coordinate system, or TOV solution.
"""
function reached_origin end

### Callbacks

"""
    create_horizon_callback(coord_sys::AbstractTOVCoordSystem, params::TOVParameters, tol::Float64)

Interface function: return ContinousCallback if solution in the given coordinate system reaches the horizon. 
"""
function create_horizon_callback end

"""
    create_origin_callback(coord_sys::AbstractTOVCoordSystem, params::TOVParameters, tol::Float64)

Interface function: return ContinousCallback if solution in given coordinate system reaches origin
"""
function create_origin_callback end

### TOV equation

abstract type AbstractTOVEquation end

"""
    coord_system(::AbstractTOVEquation)::AbstractTOVCoordSystem
    coord_system(::AbstractInitialCondition)::AbstractTOVCoordSystem
    coord_system(::AbstractTOVProblem)::AbstractTOVCoordSystem
    coord_system(::AbstractTOVSolution)::AbstractTOVCoordSystem

Interface function: return coordinate system associated with the given TOV equation, problem, solution, or initial condition. 
"""
function coord_system end

### Initial Conditions

abstract type AbstractInitialCondition end

"""
    initial_state(::AbstractInitialCondition)::NTuple{N,T}

Interface function: return the intitial state for a given kind of initial conditions. 
"""
function initial_state end

"""
    initial_arg(::AbstractInitialCondition)::NTuple{N,T}

Interface function: return the intitial argument for a given kind of initial conditions. 
"""
function initial_arg end

"""
  default_direction(::AbstractInitialCondition)

Interface function: return default integration direction.
"""
function default_direction end

### TOV parameters

abstract type AbstractTOVParameters end

"""
  paramter_tuple(::AbstractTOVParameters)

Interface function: return parameters of a TOV system as a tuple.
"""
function paramter_tuple end

### TOV Problem

abstract type AbstractTOVProblem end

"""
  parameters(::AbstractTOVProblem)
  parameters(::AbstractTOVSolution)

Interface function: return TOV parameters for a given TOV problem or solution.
"""
function parameters end

"""
  initial_conditions(::AbstractTOVProblem)
  initial_conditions(::AbstractTOVSolution)

Interface function: return initial_conditions for a given TOV problem or solution.
"""
function initial_conditions end

"""
  arg_span(::AbstractTOVProblem)

Interface function: return argument span for a given TOV problem.
"""
function arg_span end

"""
  integ_direction(::AbstractTOVProblem)

Interface function: return integration direction for a given TOV problem.
"""
function integ_direction end

### TOV solution

"""
  AbstractTOVSolution

Interface functions (among others): 
- `Base.length(::AbstractTOVSolution)`
- `Base.getindex(::AbstractTOVSolution, i::Int)`
- `Base.iterate(::AbstractTOVSolution, state=1)`
"""
abstract type AbstractTOVSolution end

"""
  hit_horizon(::AbstractTOVSolution)

Interface function: return wether the solution reached the horizon.
"""
function hit_horizon end

