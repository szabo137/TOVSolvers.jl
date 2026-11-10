struct TOVSolution{S,C<:AbstractTOVCoordSystem, IC<:AbstractInitialCondition} <: AbstractTOVSolution
    ode_sol::S # Store the raw ODE solution for interpolation if needed
    params::TOVParameters
    coord_sys::C
    ic::IC
    hit_horizon::Bool
    reached_origin::Bool
    
    function TOVSolution(
        ode_sol::S,
        params::TOVParameters,
        coord_sys::C,
        initial_condition::IC,
        hit_horizon::Bool,
        reached_origin::Bool,
    ) where {S, C<:AbstractTOVCoordSystem, IC <: AbstractInitialCondition}
        coord_system(ic) == coord_sys || throw(
            ArgumentError(
                "coordinate system must match with the one from initial conditions"
            )
        )
        
        new{S,C,IC}(
            ode_sol,
            params,
            coord_sys,
            initial_condition,
            hit_horizon,
            reached_origin,
        )
    end
end

parameters(sol::TOVSolution) = getfield(sol,:params)
initial_conditions(sol::TOVSolution) = getfield(sol,:ic)
coord_system(sol::TOVSolution) = getfield(sol,:coord_sys)
hit_horizon(sol::TOVSolution) = getfield(sol,:hit_horizon)
reached_origin(sol::TOVSolution) = getfield(sol,:reached_origin)

# Convenience methods
Base.length(sol::TOVSolution) = length(sol.r)
Base.getindex(sol::TOVSolution, i::Int) = (sol.r[i], sol.m[i], sol.m_prime[i])
Base.iterate(sol::TOVSolution, state=1) = state > length(sol) ? nothing : (sol[state], state+1)


function Base.getproperty(sol::TOVSolution, s::Symbol)
    idx = findfirst(x -> x===s, coordinate_names(coord_system(sol)))

    if isnothing(idx)
        return getfield(sol,s)
    end

    if idx == 1
        return sol.ode_sol.t
    else
        return sol.ode_sol[idx-1,:]
    end

    @error "This should never be reached"
end

# Solution classifier
classify_solution(sol::TOVSolution) = _classify_solution(initial_conditions(sol),sol)

function _classify_solution(ic::RegularOrigin, sol::TOVSolution)
        sol.hit_horizon ? :regular_origin_to_horizon : :regular_origin_incomplete
end

function Base.show(io::IO, sol::TOVSolution)
    coord_sys = coord_system(sol)
    coord_names = coordinate_names(coord_sys)
    classification = classify_solution(sol)
    n_points = length(sol)
    args = sol.ode_sol.t
    arg_range = (args[1], args[end])

    u1s = sol.ode_sol[1,:]
    u1_range = (minimum(u1s), maximum(u1s))

    u2s = sol.ode_sol[2,:]
    u2_range = (minimum(u2s), maximum(u2s))
    
    println(io, "TOVSolution with $n_points points, in $coord_names system")
    println(io, "  Classification: $classification")
    println(io, "  Parameters: omega=$(sol.params.omega), x=$(sol.params.x)")
    
    println(io, "  $(string(coord_names[1])) ∈ [$(arg_range[1]), $(arg_range[2])]")
    println(io, "  $(string(coord_names[2])) ∈ [$(u1_range[1]), $(u1_range[2])]")
    println(io, "  $(string(coord_names[3])) ∈ [$(u2_range[1]), $(u2_range[2])]")

    println(io, "  Hit horizon: $(sol.hit_horizon)")
    println(io, "  Reached origin: $(sol.reached_origin)")
    #println(io, "  Return code: $(sol.retcode)")
end
