using Pkg
# this will be replace by the module load in the future
Pkg.activate("PiCLES/")  # Activate the PiCLES package 

using PiCLES
using PiCLES.Operators.core_2D: ParticleDefaults
using PiCLES.Models.WaveGrowthModels2D: WaveGrowth2D
using PiCLES.Simulations
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, ProjetionKernel, TwoDCartesianGridStatistics

using PiCLES.ParticleSystems: particle_waves_v6 as PW
using Oceananigans.Units

# just for simple plotting
import Plots as plt

# Parameters
U10, V10 = 10.0, 10.0
DT = 10minutes
r_g0 = 0.85 # ratio of c / c_g (phase velocity/ group velocity).

# Define wind functions
u(x, y, t) = U10
v(x, y, t) = V10
winds = (u=u, v=v)

# Define grid
grid = TwoDCartesianGridMesh(100e3, 51, 100e3, 51)


# Define ODE parameters
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=r_g0)

# Define particle equations
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q);

# Calculate minimal windsea based on characteristic winds
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT)

# Define default particle
default_particle = ParticleDefaults(WindSeamin["lne"], WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0)

# Define ODE settings
ODE_settings = PW.ODESettings(
    Parameters=ODEpars,
    log_energy_minimum=WindSeamin["lne"],  # define mininum energy threshold
    saving_step=DT,
    timestep=DT,
    total_time=T = 6days,
    dt=1e-3, #60*10, 
    dtmin=1e-4, #60*5, 
    force_dtmin=true)

# Build wave model
wave_model = WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,
    periodic_boundary=false,
    minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

# Build simulation
wave_simulation = Simulation(wave_model, Δt=DT, stop_time=2hour)#1hours)

# Run simulation
run!(wave_simulation, cash_store=true)

# Plot initial state
istate = wave_simulation.store.store[end];
p1 = plt.heatmap(grid.data.x[:,1] / 1e3, grid.data.y[1,:] / 1e3, istate[:, :, 1])
