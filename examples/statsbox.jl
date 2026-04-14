# TODO: 
# - put this into a Makie extension



function statbox!(fig::Makie.FigureAxisPlot, sol; position=(1, 2))
    f, _, _ = fig
    statbox!(f, sol; position)
    fig
end
function statbox!(fig::Makie.Figure, h::TOVSolution; position=(1, 2))
    classification = classify_solution(sol)
    n_points = length(sol)
    r_range = (sol.r[1], sol.r[end])
    m_range = (minimum(sol.m), maximum(sol.m))
    mp_range = (minimum(sol.m_prime), maximum(sol.m_prime))
    
    labels = [
        latexstring("\$\\omega=$(sol.params.omega),\\quad x=$(sol.params.x)\$"),
        latexstring("\$r \\in [$(round(r_range[1]; digits = 2)),\\ $(round(r_range[2]; digits = 2))]\$"),
        latexstring("\$m \\in [$(round(m_range[1]; digits = 2)),\\ $(round(m_range[2]; digits = 2))]\$"),
        latexstring("\$m' \\in [$(round(mp_range[1]; digits = 2)),\\ $(round(mp_range[2]; digits = 2))]\$"),
        "Classification: $classification",
        "Hit horizon: $(sol.hit_horizon)",
        "Reached origin: $(sol.reached_origin)",
    ]

    elements = fill(PolyElement(polycolor=:transparent), length(labels))
    Legend(getindex(fig, position...), elements, labels)
    fig
end
