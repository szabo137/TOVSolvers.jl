"""
    compute_H(r, m, m_prime, params::TOVParameters)

Compute the first-order part H(r, m, m') from the TOV equation:

H(r, m, m') = m'x(2r - (x+5)m) - (x+1)r(m')²
              + omegar²(xm + (x+2)rm') - omega²r⁵

# Arguments
- `r`: Radius
- `m`: Mass
- `m_prime`: Mass derivative dm/dr
- `params`: TOV parameters (omega, x)

# Returns
- Value of H(r, m, m')
"""
function compute_H(r::Real, m::Real, m_prime::Real, params::TOVParameters)
    omega = params.omega
    x = params.x
    
    term1 = m_prime * x * (2*r - (x+5)*m)
    term2 = -(x+1) * r * m_prime^2
    term3 = omega * r^2 * (x*m + (x+2)*r*m_prime)
    term4 = -omega^2 * r^5
    
    return term1 + term2 + term3 + term4
end

"""
    tov_second_order!(du, u, params, r)

Second-order ODE formulation of the TOV equation.
State vector: u = [m, m']

The equation is:
m''(r) = H(r, m, m') / (xr(r - 2m))

where H is given by compute_H.
"""
function tov_second_order!(du, u, p, r)
    x = p.x
    w = p.omega 
    m  = u[1]
    mp = u[2]

    # Avoid division by zero near r = 0
    denom = x * (r - 2*m)*r

    if denom <=1e-8
        du[1] = 1.0
        du[2] = 0.0
    end
    # Second derivative from the given equation
    mpp = (
        - mp * x * (-2r + (x+5)*m)
        - (x+1) * r * mp^2
        + w * r^2 * (x*m + (x+2)*r*mp)
        - w^2 * r^5
    ) / denom

    du[1] = mp
    du[2] = mpp
end

function m_ode!(du, y, p, r)
    x, w = p # WARN: check order!
    m  = y[1]
    mp = y[2]

    # Avoid division by zero near r = 0
    denom = x * (r - 2*m)*r

    if denom <=1e-8
        du[1] = 1.0
        du[2] = 0.0
    end
    # Second derivative from the given equation
    mpp = (
        - mp * x * (-2r + (x+5)*m)
        - (x+1) * r * mp^2
        + w * r^2 * (x*m + (x+2)*r*mp)
        - w^2 * r^5
    ) / denom

    du[1] = mp
    du[2] = mpp
end

"""
    is_at_horizon(r, m; rtol=1e-6)

Check if the solution is at or near the horizon r = 2m.

# Arguments
- `r`: Current radius
- `m`: Current mass
- `rtol`: Relative tolerance for horizon detection

# Returns
- `true` if |r - 2m| / r < rtol
"""
function is_at_horizon(r::Real, m::Real; rtol::Real=1e-6)
    r <= 0 && return false
    return abs(r - 2*m) / r < rtol
end

"""
    is_on_slow_manifold(r, m, params::TOVParameters; rtol=1e-6)

Check if the solution is on or near the slow manifold m = (omega/x)r³.

# Arguments
- `r`: Current radius
- `m`: Current mass  
- `params`: TOV parameters
- `rtol`: Relative tolerance

# Returns
- `true` if the solution is on the slow manifold within tolerance
"""
function is_on_slow_manifold(r::Real, m::Real, params::TOVParameters; rtol::Real=1e-6)
    r <= 0 && return false
    m_slow = slow_manifold(r, params)
    return abs(m - m_slow) / (abs(m_slow) + 1e-14) < rtol
end
