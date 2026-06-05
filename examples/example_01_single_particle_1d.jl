"""
Example: 1D wave growth with the PIC remeshing algorithm.

The 1D capability is an alias over the 2D engine: `WaveGrowth1D` builds a `WaveGrowth2D` on a
thin, periodic-in-y `CartesianGridMesh1D` with zero meridional wind, and the output is squeezed
back to 1D. Note that no particle-system version is named anywhere — the canonical
`particle_equations` / `ODESettings` / `ODEParameters` come straight from `PiCLES.ParticleSystems`.

This is a long-form exploratory run intended for manual inspection and plotting.
"""

using PiCLES
using PiCLES.ParticleSystems                       # canonical (version-agnostic) particle system
import PiCLES: FetchRelations
using PiCLES.Grids.CartesianGrid: CartesianGridMesh1D
using PiCLES.Models: WaveGrowth1D
using PiCLES.Simulations
using PiCLES.Plotting

using Oceananigans.Units
import Plots

plot_path_base = "plots/tests/example_01_1D/"
mkpath(plot_path_base)

# %% parameters -----------------------------------------------------------------
DT = Float64(10minutes)
r_g0 = 0.85

ODEpars, Const_ID, Const_Scg = ODEParameters(r_g=r_g0)

u10 = 10.0
# a 1D wind is a function of (x, t); the alias supplies v ≡ 0 internally
u(x, t) = x * 0 + u10

particle_system = particle_equations(γ=Const_ID.γ, q=Const_ID.q)

WindSeamin = FetchRelations.MinimalWindsea(u10, 0.0, DT)
ODE_settings = ODESettings(
    Parameters=ODEpars,
    log_energy_minimum=log(WindSeamin["E"]),
    log_energy_maximum=log(17),     # ~ Hs 16 m
    saving_step=DT,
    timestep=DT,
    total_time=2 * 24 * 3600.0,
    dt=1e-3,
    dtmin=1e-4,
    dtmax=DT,
)

# A small constructor helper so each experiment gets a fresh model. Boundary topology
# (periodic vs. not) is fixed at construction, so we build a new model per case rather than
# mutating an existing one.
function build_1d_model(; periodic_boundary)
    grid1d = CartesianGridMesh1D(30e3, 50; Ny=3, periodic_boundary=periodic_boundary)
    return WaveGrowth1D(; grid=grid1d,
        winds=u,
        ODEsys=particle_system,
        ODEsets=ODE_settings,
        ODEinit_type="wind_sea",
        minimal_particle=FetchRelations.MinimalParticle(2, 0, DT),
        periodic_boundary=periodic_boundary,
        boundary_type="same")
end

# %% experiment 1: positive winds, periodic -------------------------------------
@info "experiment 1: positive winds, periodic \n"
wave_model = build_1d_model(periodic_boundary=true)
wave_simulation = Simulation(wave_model, Δt=DT, stop_time=3hours)
initialize_simulation!(wave_simulation)
run!(wave_simulation, store=false, cash_store=true, debug=false)

Plotting.plot_results(wave_simulation, title="$u10 m/s, periodic=true")
Plots.savefig(joinpath(plot_path_base, "u$(u10)_per_true.png"))

# %% experiment 2: positive winds, non-periodic ---------------------------------
@info "experiment 2: positive winds, non-periodic \n"
wave_model = build_1d_model(periodic_boundary=false)
wave_simulation = Simulation(wave_model, Δt=DT, stop_time=3hours)
initialize_simulation!(wave_simulation)
run!(wave_simulation, store=false, cash_store=true, debug=false)
Plotting.plot_results(wave_simulation, title="$u10 m/s, periodic=false")
Plots.savefig(joinpath(plot_path_base, "u$(u10)_per_false.png"))

# %% experiment 3: negative winds, periodic -------------------------------------
@info "experiment 3: negative winds, periodic \n"
u10 = -10.0
u(x, t) = x * 0 + u10
wave_model = build_1d_model(periodic_boundary=true)
wave_simulation = Simulation(wave_model, Δt=DT, stop_time=3hours
)
initialize_simulation!(wave_simulation)
run!(wave_simulation, store=false, cash_store=true, debug=false)
Plotting.plot_results(wave_simulation, title="$u10 m/s, periodic=true")
Plots.savefig(joinpath(plot_path_base, "u$(u10)_per_true.png"))

# %% experiment 4: negative winds, non-periodic ---------------------------------
@info "experiment 4: negative winds, non-periodic \n"
u10 = -10.0
u(x, t) = x * 0 + u10
wave_model = build_1d_model(periodic_boundary=false)
wave_simulation = Simulation(wave_model, Δt=DT, stop_time=3hours
)
initialize_simulation!(wave_simulation)
run!(wave_simulation, store=false, cash_store=true, debug=false)
Plotting.plot_results(wave_simulation, title="$u10 m/s, periodic=false")
Plots.savefig(joinpath(plot_path_base, "u$(u10)_per_false.png"))  

@info "... finished\n"
