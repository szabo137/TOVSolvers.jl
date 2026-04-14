
using TOVSolvers
using CairoMakie
using LaTeXStrings
using CSV
using Tables

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
fig = Figure(size = (1000, 800))

ax  = Axis(
   fig[1, 1],
    xlabel = "m/r",
    ylabel = "m'",
    #xscale = log10,
    #yscale = Makie.Symlog10(-1e-10,1e2),
    #yscale = log10,
    limits = (
        (0.0,0.5 ),
        (0.0,1.25)
    )
)

#### Burkhards data
file_name = "BK_07-02-26.csv"

data = CSV.read(
    joinpath(
        DATADIR,
        file_name
    ),
    Tables.matrix;
header = false
)

scatter!(ax,data, label = "BK", marker=:circle, color = :orange)


### my plots

w = params.omega
rvals = sol.r
mvals = sol.m
mpvals = sol.m_prime

lines!(ax, mvals ./ rvals, mpvals, label = "$(ic_label(sol.initial_condition))", linewidth=3)

axislegend(ax,position = :lt)

filename = joinpath(PLOTDIR, "Collin_spiral_BKcompare.pdf")

save(filename, fig)
display(fig)

