function create_callbacks(prob::AbstractTOVProblem, config)
    callbacks = []
    
    # Always add negative mass callback
    #push!(callbacks, create_negative_mass_callback())
    
    if integ_direction(prob)== :forward
        # For forward integration from origin, add horizon callback
        push!(callbacks, create_horizon_callback(coord_system(prob), parameters(prob), config.horizon_tolerance))
    elseif prob.direction == :backward
        # For backward integration from horizon, add origin callback
        push!(callbacks, create_origin_callback(coord_system(prob),parameters(prob), config.origin_tolerance))
        
        # Also add horizon callback in case we reach it again
        push!(callbacks, create_horizon_callback(coord_system(prob), parameters(prob), config.horizon_tolerance))
    end
    
    return CallbackSet(callbacks...)
end
