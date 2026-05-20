# Quick start

This walks through the smallest possible PiCLES run: a Cartesian box with
uniform 10 m s⁻¹ wind, advanced for a few timesteps.

## Install

PiCLES targets Julia 1.10 or newer. From a fresh clone:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Minimal 2D run

```julia
using PiCLES
using PiCLES.FetchRelations
using PiCLES.ParticleSystems: particle_waves_v5 as PW
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh
using PiCLES.Models.WaveGrowthModels2D: WaveGrowth2D
using PiCLES.Simulations: Simulation, run!

using Oceananigans.Units

# Background wind field
U10, V10 = 10.0, 10.0
u(x, y, t) = U10
v(x, y, t) = V10
winds = (u = u, v = v)

# 100 km × 100 km grid, 51 × 51 cells
grid = TwoDCartesianGridMesh(100e3, 51, 100e3, 51)

# Particle equations and ODE settings
DT = 10minutes
ODEpars, Const_ID, _ = PW.ODEParameters(r_g = 0.85)
particle_system = PW.particle_equations(u, v; γ = Const_ID.γ, q = Const_ID.q)

WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT)
ODE_settings = PW.ODESettings(
    Parameters         = ODEpars,
    log_energy_minimum = WindSeamin["lne"],
    saving_step        = DT,
    timestep           = DT,
    total_time         = 6days,
    dt                 = 1e-3,
    dtmin              = 1e-4,
    force_dtmin        = true,
)

wave_model = WaveGrowth2D(;
    grid              = grid,
    winds             = winds,
    ODEsys            = particle_system,
    ODEsets           = ODE_settings,
    periodic_boundary = false,
    minimal_particle  = FetchRelations.MinimalParticle(U10, V10, DT),
)

sim = Simulation(wave_model; Δt = DT, stop_time = 2hours)
run!(sim, cash_store = true)
```

After `run!`, `sim.store.store` holds the saved state slices. Each slice is
an `Nx × Ny × 3` array of `(ε, m_x, m_y)` at the corresponding save time.

## Running the smoke tests

```julia
using Pkg
Pkg.activate(".")
Pkg.test()
```

The smoke suite builds an 11 × 11 model and advances a single timestep — it
verifies the package loads, the fetch relations are finite, and that the
PIC timestep does not blow up the state. The richer scripts under
`test/T0*.jl` and `examples/` are not run by the smoke suite; they document
the validation cases from the paper.
