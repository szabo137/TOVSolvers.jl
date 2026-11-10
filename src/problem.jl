struct TOVProblem{C<:AbstractTOVCoordSystem,IC <: AbstractInitialCondition} <: AbstractTOVProblem
    params::TOVParameters
    coord_sys::C
    ic::IC
    arg_span::Tuple{Float64, Float64}
    direction::Symbol
    
    function TOVProblem(
        params::TOVParameters,
        coord_sys::C,
        ic::IC,
        arg_span::Tuple{Real, Real};
        direction::Symbol=:auto
    ) where {C<:AbstractTOVCoordSystem,IC <: AbstractInitialCondition}

        coord_system(ic) == coord_sys || throw(
            ArgumentError(
                "coordinate system must match with the one from initial conditions"
            )
        )
        
        direction in [:forward, :backward, :auto] || throw(ArgumentError("direction must be :forward, :backward, or :auto"))

        if direction == :auto
            direction = default_direction(ic)
        end
        
        # TODO: avoid type casting
        arg_start, arg_end = arg_span
        
        # Validate direction matches span
        if direction == :forward && arg_end < arg_start
            @warn "Forward integration requested but arg_end < arg_start. Setting direction to :backward"
            direction = :backward
        elseif direction == :backward && arg_end > arg_start
            @warn "Backward integration requested but arg_end > arg_start. Setting direction to :forward"
            direction = :forward
        end

        
        new{C,IC}(params,coord_sys, ic, arg_span, direction)
    end
end

parameters(prob::TOVProblem) = prob.params
initial_conditions(prob::TOVProblem) = prob.ic
coord_system(prob::TOVProblem) = prob.coord_sys
arg_span(prob::TOVProblem) = prob.arg_span
integ_direction(prob::TOVProblem) = prob.direction

function TOVProblem(
    params::TOVParameters,
        coord_sys::AbstractTOVCoordSystem,
    ic::AbstractInitialCondition,
    arg_limit::Real; # could be r_end, or r0, depending on the ic
    direction::Symbol=:auto
)
    arg_init = initial_arg(ic)

    arg_init != arg_limit || throw(
        ArgumentError(
            "given argument limit must not match initial argument given by the initial conditions"
            )
    )

    arg_span = arg_limit < arg_init ? (arg_limit,arg_init) : (arg_init, arg_limit)
    
    if direction == :auto
        direction = default_direction(ic)
    end
    
    return TOVProblem(params, coord_sys, ic, arg_span, direction=direction)
end

function TOVProblem(
    params::TOVParameters,
    ic::AbstractInitialCondition,
    arg_limit::Real; # could be r_end, or r0, depending on the ic
    direction::Symbol=:auto
)
    return TOVProblem(params, coord_system(ic),ic, arg_limit;direction)
end
