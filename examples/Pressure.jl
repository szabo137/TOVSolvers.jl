using TOVSolvers
using CairoMakie
using LaTeXStrings
using CSV
using Tables

PLOTDIR = "plots"

params = TOVParameters(
    1e-3, # omega
    3.0,  # x
)

@info params

ic = RegularOrigin(
    m=1e-10,
    m_prime=0.01
)

@info ic

prob = TOVProblem(
    params,
    ic,
    (
        1e-3, # r0
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

ax_phase_scaled = Axis(
    fig[1, 1],
    xlabel = "r",
    ylabel = "p",
    #xscale = log10,
    #yscale = Makie.Symlog10(1e-4),
    yscale = log10,
    limits = (
        (0.0,0.5),
        (1e-4,100)
    )
)

w = params.omega
rvals = sol.r
mvals = sol.m
mpvals = sol.m_prime

scaled_mvals = mvals ./ rvals

pvals = @. mpvals/rvals^2 - w

pos_mask = @. pvals >0
m_pos = scaled_mvals[pos_mask]
neg_mask = @. pvals <0
m_neg = scaled_mvals[neg_mask]

pvals_pos = filter(x -> x>0, pvals)
pvals_neg = abs.(filter(x -> x<0, pvals))

lines!(ax_phase_scaled, m_pos, pvals_pos,linewidth = 3, color = :blue, label="positive pressure")
lines!(ax_phase_scaled, m_neg, pvals_neg,linewidth = 3, color = :red, label = "negative pressure")

axislegend(ax_phase_scaled,position = :rt)

statbox!(fig,sol)

filename = joinpath(PLOTDIR, "Pressure.pdf")

save(filename, fig)
display(fig)
