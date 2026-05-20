# PiCLES.jl

**Particle-In-CelL for Efficient Swell** — a fast surface wave model for Earth
System Models, written in [Julia](https://julialang.org).

PiCLES advances wave growth and propagation along Lagrangian particle
trajectories, then uses Particle-in-Cell (PIC) remeshing to deposit a compact
state vector (energy `ε`, energy-weighted group-velocity components `m_x, m_y`)
back onto a regular ocean grid. The reduced state vector (~5 variables per
particle vs. ~600 for a third-generation spectral model) is the key to its
speed: depending on resolution, PiCLES is 1–4 orders of magnitude cheaper than
WW3-class models while still resolving the dominant wind sea relevant to
air–sea coupling.

## Where to go next

- The [Model](@ref) page summarises the equations PiCLES integrates and how
  the PIC remeshing works.
- The [Quick start](@ref) walks through a minimal 2D run on a Cartesian grid.
- The [API reference](@ref) lists the public modules and types.

## Citing

If PiCLES is useful to you, please cite the JAMES paper:

> Hell, M. C., Fox-Kemper, B., & Chapron, B. (2025). *A Particle-in-Cell wave
> model for efficient sea-state estimates in Earth System Models — PiCLES.*
> Journal of Advances in Modeling Earth Systems, 17(8).
> [doi:10.1029/2025MS005221](https://doi.org/10.1029/2025MS005221)

The software is archived on Zenodo at
[doi:10.5281/zenodo.13799205](https://doi.org/10.5281/zenodo.13799205).

## License

PiCLES.jl is distributed under the Apache 2.0 license. See
[`LICENSE`](https://github.com/mochell/PiCLES.jl/blob/main/LICENSE) for the
full text.
