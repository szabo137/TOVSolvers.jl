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
fig = Figure(size = (1000, 800))

ax_rm  = Axis(
    fig[1, 1], 
    xlabel = "r",
    ylabel = "m'(r)", 
    xscale = log10,
    yscale = log10,
    limits =(
        (1e-3,1e2), 
        (1e-5, 1e2)
    )
)

#### Burkhards data
file_name = "BK_09-04-26.csv"

data = CSV.read(
    joinpath(
        DATADIR,
        file_name
    ),
    Tables.matrix;
header = false

)


scatter!(ax_rm,data, label = "BK", marker=:circle, color = :green)


### my plots

w = params.omega
rvals = sol.r
mvals = sol.m
mpvals = sol.m_prime

lines!(ax_rm, rvals, mpvals,  label = "$(ic_label(sol.initial_condition))", linewidth = 4, color = :black)

axislegend(ax_rm,position = :lt)

filename = joinpath(PLOTDIR, "Pressure_BKcompare.pdf")

save(filename, fig)
display(fig)

