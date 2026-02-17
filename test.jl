using DifferentialEquations
# Parameters (adjust as needed)
const x = 3.0
const w = 1e-3

# Rewrite the second-order ODE as a first-order system
# y[1] = m, y[2] = m'
function m_ode!(du, y, p, r)
    x, w = p
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
# Event: detect r = 2m(r)
# We stop integration exactly at the crossing to avoid the singular denominator
function crossing_condition(u, r, integrator)
    m = u[1]
    return r - 2m
end

function crossing_affect!(integrator)
    terminate!(integrator)
end

crossing_cb = ContinuousCallback(crossing_condition, crossing_affect!)

params = (x, w)

# Integration domain
rspan = (1e-10, 2.0)

y0 = [0.0,1.0]

prob = ODEProblem(m_ode!, y0, rspan, params)
sol = solve(
    prob,
    Vern9();
    callback = crossing_cb,
    abstol=1e-9,
    reltol=1e-9)

println("..... DONE ......")
