struct MMprimeSystem <: AbstractTOVCoordSystem end

coordinate_names(::MMprimeSystem) = (:r, :m, :m_prime)

function _compute_H(r::Real, m::Real, m_prime::Real, params::TOVParameters)
    omega = params.omega
    x = params.x

    term1 = m_prime * x * (2 * r - (x + 5) * m)
    term2 = -(x + 1) * r * m_prime^2
    term3 = omega * r^2 * (x * m + (x + 2) * r * m_prime)
    term4 = -omega^2 * r^5

    return term1 + term2 + term3 + term4
end

function _get_tov_equation(::MMprimeSystem)
    # u  = (m, m')
    # du = (m', m'')

    function _eq(du, u, params, r)
        x = params.x
        m = u[1]
        mp = u[2]

        denom = x * (r - 2 * m) * r
        mpp = _compute_H(r, m, mp, params) / denom

        du[1] = mp
        du[2] = mpp

        return nothing # important!
    end
    return _eq
end

function assert_regular_origin(::MMprimeSystem, r, m, mp; tol = 1.0e-10)

    if m >= tol && mp >= tol
        @warn "Both m and m_prime are non-zero. This will create a singularity at r=0."
    end

    return nothing
end

# TODO: implement _assert_admissible_crossing

function create_horizon_callback(coord_sys::MMprimeSystem, params::TOVParameters, atol::Float64)
    condition(u, r, integrator) = begin
        m = u[1]
        # Return negative when approaching horizon (triggers when crossing zero)
        #return isapprox(r,2*m; atol)
        return r - 2 * m - atol * r
    end

    affect!(integrator) = begin
        @info "Horizon reached at r=$(integrator.t), m=$(integrator.u[1]). Terminating integration."
        terminate!(integrator)
    end

    return ContinuousCallback(condition, affect!)
end

# TODO: implement callbacks for reaching origin

function reached_horizon(::MMprimeSystem, r, m, mp; atol = 1.0e-10)
    return isapprox(r, 2 * m; atol = 100 * atol)
end
