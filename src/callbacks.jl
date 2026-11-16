function create_callbacks(prob::AbstractTOVProblem, config)
    callbacks = []

    # Always add negative mass callback
    #push!(callbacks, create_negative_mass_callback())

    integ_dir = integration_direction(prob)

    if integ_dir == :forward
        # For forward integration from origin, add horizon callback
        push!(callbacks, create_horizon_callback(coord_system(prob), parameters(prob), config.horizon_tolerance))
    elseif integ_dir == :backward
        # For backward integration from horizon, add origin callback
        push!(callbacks, create_origin_callback(coord_system(prob), parameters(prob), config.origin_tolerance))

        # Also add horizon callback in case we reach it again
        push!(callbacks, create_horizon_callback(coord_system(prob), parameters(prob), config.horizon_tolerance))
    else
        throw(ArgumentError("only :forward and :backward are valid integration directions. $(integ_dir) given."))
    end

    return CallbackSet(callbacks...)
end
