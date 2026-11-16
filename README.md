# TOVSolvers

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://szabo137.github.io/TOVSolvers.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://szabo137.github.io/TOVSolvers.jl/dev)
[![Test workflow status](https://github.com/szabo137/TOVSolvers.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/szabo137/TOVSolvers.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/szabo137/TOVSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/szabo137/TOVSolvers.jl)
[![Docs workflow Status](https://github.com/szabo137/TOVSolvers.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/szabo137/TOVSolvers.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

`TOVSolvers.jl` is a Julia package for solving the
Tolman–Oppenheimer–Volkoff (TOV) equations with linear affine equations of
state. It provides a common interface for multiple coordinate systems,
initial conditions, event callbacks, and numerical solver configurations.

## Installation

Install from the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/szabo137/TOVSolvers.jl")
```

## Quickstart

```julia
using TOVSolvers

coord_sys = MMprimeSystem()
params = TOVParameters(
    1.0,                # omega
    3.0                 # x
)

initial_condition = RegularOrigin(
    coord_sys,
    1e-16,              # r_initial
    0.0,                # m_initial
    1.0                 # m_prime_initial
)

problem = TOVProblem(
    params,
    initial_condition,
    10.0;               # r_final
    direction=:forward,
)

config = TOVSolverConfig(
    abstol=1e-10,
    reltol=1e-10,
)

solution = solve(problem, config)
```

Inspect the result:

```julia
reached_horizon(solution)
reached_origin(solution)
classify_solution(solution)
```

## Credits and Funding

This work has been partially supported by the **Center for Advanced Systems
Understanding (CASUS)**, funded by the German Federal Ministry of Education and
Research (BMBF) and the Saxon Ministry for Science, Culture and Tourism (SMWK)
with tax funds based on the budget approved by the Saxon State Parliament.

### Acknowledgements

Special thanks to **Tobias Dornheim** for funding and support, and **Burkhard Kämpfer** for value input.

## License

[MIT](LICENSE) © Uwe Hernandez Acosta
