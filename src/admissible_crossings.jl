"""
    compute_discriminant(r_h, m_h, params::TOVParameters)

Compute the discriminant Δ = B² - 4AC for the admissible crossing condition.

For a crossing at the horizon r = 2m to be admissible (regular), the 
discriminant must be non-negative.

# Arguments
- `r_h`: Horizon radius
- `m_h`: Horizon mass (should equal r_h/2)
- `params`: TOV parameters

# Returns
- Discriminant value Δ
"""
function compute_discriminant(r_h::Real, m_h::Real, params::TOVParameters)
    omega = params.omega
    x = params.x
    
    A = -(x+1) * 2 * m_h
    B = x * (4*m_h - (x+5)*m_h) + 8*omega*(x+2)*m_h^3
    C = omega * m_h^3 * (x - 32*omega*m_h^2)
    
    return B^2 - 4*A*C
end

"""
    compute_crossing_slopes(r_h, m_h, params::TOVParameters)

Compute the admissible crossing slopes m'± at the horizon.

The slopes are given by:
m'± = (-B ± √Δ) / (2A)

where Δ is the discriminant and A, B, C are coefficients from the 
quadratic equation for regularity at the horizon.

# Arguments
- `r_h`: Horizon radius
- `m_h`: Horizon mass (should equal r_h/2)
- `params`: TOV parameters

# Returns
- `(m_plus, m_minus)`: Tuple of the two crossing slopes, or `(NaN, NaN)` if no real solutions exist

# Example
```julia
params = TOVParameters(0.1, 3.0)
m_plus, m_minus = compute_crossing_slopes(1.0, 0.5, params)
```
"""
function compute_crossing_slopes(r_h::Real, m_h::Real, params::TOVParameters)
    omega = params.omega
    x = params.x
    
    # Coefficients
    A = -(x+1) * 2 * m_h
    B = x * (4*m_h - (x+5)*m_h) + 8*omega*(x+2)*m_h^3
    C = omega * m_h^3 * (x - 32*omega*m_h^2)
    
    # Discriminant
    Δ = B^2 - 4*A*C
    
    if Δ < 0
        @warn "No real admissible crossing exists at r_h=$r_h (discriminant=$Δ < 0)"
        return (NaN, NaN)
    end
    
    sqrt_Δ = sqrt(Δ)
    m_plus = (-B + sqrt_Δ) / (2*A)
    m_minus = (-B - sqrt_Δ) / (2*A)
    
    return (m_plus, m_minus)
end

"""
    find_admissible_crossings(params::TOVParameters, r_range; n_points=100)

Find all admissible horizon crossings in a given radius range.

Scans the radius range and identifies where the discriminant is non-negative,
indicating the existence of admissible crossings.

# Arguments
- `params`: TOV parameters
- `r_range`: Range of radii to scan (e.g., 0.1:0.1:10.0)
- `n_points`: Number of points to sample in the range

# Returns
- Vector of `AdmissibleCrossing` objects for each admissible horizon

# Example
```julia
params = TOVParameters(0.1, 3.0)
crossings = find_admissible_crossings(params, 0.1:0.1:5.0)
```
"""
function find_admissible_crossings(params::TOVParameters, r_range; n_points::Int=100)
    r_min, r_max = extrema(r_range)
    r_values = range(r_min, r_max, length=n_points)
    
    admissible = AdmissibleCrossing[]
    
    for r_h in r_values
        m_h = r_h / 2.0
        Δ = compute_discriminant(r_h, m_h, params)
        
        if Δ >= 0
            # Store the crossing with :both to indicate both slopes are available
            push!(admissible, AdmissibleCrossing(r_h, crossing=:both, params=params))
        end
    end
    
    return admissible
end

"""
    get_initial_state(ic::AdmissibleCrossing, params::TOVParameters)

Get the initial state vectors [m, m'] for an admissible crossing.

Returns one or two state vectors depending on the crossing type.

# Arguments
- `ic`: AdmissibleCrossing initial condition
- `params`: TOV parameters

# Returns
- If crossing is :both: Vector of two state vectors [(m, m'₊), (m, m'₋)]
- If crossing is :plus or :minus: Single state vector [m, m']
"""
function get_initial_state(ic::AdmissibleCrossing, params::TOVParameters)
    m_plus, m_minus = compute_crossing_slopes(ic.r_h, ic.m_h, params)
    
    if isnan(m_plus) || isnan(m_minus)
        error("Cannot compute initial state: no admissible crossing at r_h=$(ic.r_h)")
    end
    
    if ic.crossing == :plus
        return [[ic.m_h, m_plus]]
    elseif ic.crossing == :minus
        return [[ic.m_h, m_minus]]
    else  # :both
        return [[ic.m_h, m_plus], [ic.m_h, m_minus]]
    end
end
