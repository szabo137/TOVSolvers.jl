
using TOVSolvers
using CairoMakie
using LaTeXStrings
using CSV
using Tables

include("statsbox.jl")

DATADIR = "BK_plots"
PLOTDIR = "plots"

params = TOVParameters(
    1e-3, # omega
    3.0,  # x
)

@info params

ic = RegularOrigin(
    m=1e-3,
    m_prime=0.01
)

@info ic

prob = TOVProblem(
    params,
    ic,
    (
        1e-2, # r0
        100.0, # r_max
        )
)

@info prob

config = TOVSolverConfig(abstol=1e-9,reltol=1e-9, horizon_tolerance=1e-10)

@info "Solving ..."
sol = solve(prob, config)

@info sol

@info "Plotting ... "
fig = Figure(size = (700, 400))

ax  = Axis(
   fig[1, 1],
    xlabel = "m/r",
    ylabel = "m'",
    limits = (
        (0.0,0.5 ),
        (0.0,1.25)
    )
)

### my plots

w = params.omega
rvals = sol.r
mvals = sol.m
mpvals = sol.m_prime

scaled_mvals = mvals ./ rvals
pvals = @. mpvals/rvals^2 - w

### pos pressure branch
pos_mask = @. pvals >0
m_pos = scaled_mvals[pos_mask]
mp_pos = mpvals[pos_mask]

### neg pressure branch
neg_mask = @. pvals <0
m_neg = scaled_mvals[neg_mask]
mp_neg = mpvals[neg_mask]

lines!(ax, m_pos, mp_pos,linewidth = 3, color = :blue, label = "positive pressure")
lines!(ax, m_neg, mp_neg,linewidth = 3, color = :red, label = "negative pressure")

axislegend(ax,position = :lt)

statbox!(fig,sol)

filename = joinpath(PLOTDIR, "Collin_spiral_p_colored.pdf")
save(filename, fig)
display(fig)

