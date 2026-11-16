"""
    AbstractTOVCoordSystem

Supertype for coordinate-system representations of a TOV solution. Implement
`coordinate_names`, `reached_horizon`, and `reached_origin` for each subtype.
"""
abstract type AbstractTOVCoordSystem end

"""
    coordinate_names(coord_sys)::Tuple{Vararg{Symbol}}

Return coordinate names in the same order used by every state in `coord_sys`.
Implement this for each coordinate-system subtype.
"""
function coordinate_names end

"""
    reached_horizon(coord_sys, arg, state; atol, rtol)::Bool

Return whether `state` at independent argument `arg` is at the horizon in
`coord_sys`. Implement this geometric test for each coordinate-system subtype.
"""
function reached_horizon end

"""
    reached_origin(coord_sys, arg, state; atol, rtol)::Bool

Return whether `state` at independent argument `arg` is at the origin in
`coord_sys`. Implement this geometric test for each coordinate-system subtype.
"""
function reached_origin end

"""
    assert_regular_origin(coord_sys, arg, state; kwargs...)

Throw `ArgumentError` unless `(arg, state)` is a valid regular-origin initial
condition for `coord_sys`. Implement for each coordinate system.
"""
function assert_regular_origin end

"""
    create_horizon_callback(coord_sys, params; kwargs...)

Create the solver callback for a horizon crossing. Implement it for every
coordinate system that can be integrated by the package.
"""
function create_horizon_callback end

"""
    create_origin_callback(coord_sys, params; kwargs...)

Create the solver callback for an origin crossing. Implement it for every
coordinate system that can be integrated by the package.
"""
function create_origin_callback end

"""
    AbstractTOVEquation

Supertype for a TOV equation formulation. Implement `coord_system` for each
subtype.
"""
abstract type AbstractTOVEquation end

"""
    coord_system(x)::AbstractTOVCoordSystem

Return the coordinate system associated with an equation, initial condition,
problem, or solution. Implement it for each concrete subtype.
"""
function coord_system end

"""
    AbstractInitialCondition

Supertype for TOV initial conditions. Implement `initial_state`,
`initial_argument`, and `default_direction` for each subtype.
"""
abstract type AbstractInitialCondition end

"""
    initial_state(initial_condition)

Return the initial dependent-variable state, ordered as required by its
coordinate system. Implement it for each initial-condition subtype.
"""
function initial_state end

"""
    initial_argument(initial_condition)

Return the initial value of the independent integration argument. Implement it
for each initial-condition subtype.
"""
function initial_argument end

"""
    default_direction(initial_condition)

Return the default integration direction, conventionally `:forward` or `:backward`.
Implement it for each initial-condition subtype.
"""
function default_direction end

"""
    AbstractTOVParameters

Supertype for the physical and equation-of-state parameters of a TOV system.
Implement `parameter_tuple` for each subtype.
"""
abstract type AbstractTOVParameters end

"""
    parameter_tuple(params)::Tuple

Return the parameters in the canonical order used by the equation. Implement
it for each parameter subtype.
"""
function parameter_tuple end

"""
    AbstractTOVProblem

Supertype for a fully specified TOV integration problem. Implement the problem
accessors below for each subtype.
"""
abstract type AbstractTOVProblem end

"""
    parameters(x)::AbstractTOVParameters

Return the TOV parameters stored by a problem or solution. Implement it for
each concrete problem and solution subtype.
"""
function parameters end

"""
    initial_condition(x)::AbstractInitialCondition

Return the initial condition stored by a problem or solution. Implement it for
each concrete problem and solution subtype.
"""
function initial_condition end

"""
    argument_span(problem)::Tuple

Return the integration interval as `(arg_start, arg_stop)`. Implement it for
each problem subtype.
"""
function argument_span end

"""
    integration_direction(problem)

Return the requested integration direction, conventionally `:forward` or `:backward`.
Implement it for each problem subtype.
"""
function integration_direction end

"""
    AbstractTOVSolution

Supertype for TOV integration results. Concrete subtypes must also implement
`Base.length`, `Base.getindex`, and `Base.iterate` using one consistent sample
representation.
"""
abstract type AbstractTOVSolution end

"""
    final_argument(solution)

Return the independent argument at the final saved solution state. Implement
it for each solution subtype.
"""
function final_argument end

"""
    final_state(solution)

Return the final dependent-variable state in coordinate-system order. Implement
it for each solution subtype.
"""
function final_state end

"""
    termination_reason(solution)::Symbol

Return the reason integration stopped, such as `:horizon`, `:origin`,
`:completed`, or `:failure`. Implement it for each solution subtype.
"""
function termination_reason end

"""
    reached_horizon(solution; kwargs...)::Bool

Return whether the final solution state is at the horizon. This convenience
method requires `coord_system`, `final_argument`, and `final_state`.
"""
function reached_horizon(solution::AbstractTOVSolution; kwargs...)
    reached_horizon(coord_system(solution), final_argument(solution), final_state(solution); kwargs...)
end

"""
    reached_origin(solution; kwargs...)::Bool

Return whether the final solution state is at the origin. This convenience
method requires `coord_system`, `final_argument`, and `final_state`.
"""
function reached_origin(solution::AbstractTOVSolution; kwargs...)
    reached_origin(coord_system(solution), final_argument(solution), final_state(solution); kwargs...)
end
