#ENV["JULIA_INCREMENTAL_COMPILE"] = true

using Pkg
#Pkg.activate("../PiCLES/")

# using Pkg
Pkg.activate(".")

import Plots as plt
using Setfield, IfElse

using PiCLES.ParticleSystems: particle_waves_v5 as PW

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleInstance, GetGroupVelocity
using PiCLES.Operators: TimeSteppers
using PiCLES.Simulations
using PiCLES.Operators.TimeSteppers: time_step!, movie_time_step!

using PiCLES.ParticleMesh: TwoDGrid, TwoDGridNotes, TwoDGridMesh
using PiCLES.Models.WaveGrowthModels2D

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units
import Oceananigans.Utils: prettytime

using PiCLES.Architectures
using GLMakie

using PiCLES.Operators.core_2D: GetGroupVelocity, speed
using PiCLES.Plotting.movie: init_movie_2D_box_plot

using Revise

Revise.retry()
using PiCLES: ParticleInstance, ParticleInstance2D, ParticleInstance2DLayer




# debugging:
#using ProfileView

# %%
save_path = "plots/tests/T06_layers/"
mkpath(save_path)

# % Parameters
U10, V10 = 10.0, 10.0
dt_ODE_save = 30minutes
DT = 30minutes
# version 3

ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)


u_func(x, y, t) = U10 #* sin(t / (6 * 60 * 60 * 2π)) * sin(x / 50e3) * sin(y / 50e3)
v_func(x, y, t) = V10 #* cos(t / (6 * 60 * 60 * 2π)) * sin(x / 50e3) * sin(y / 50e3)

u(x, y, t) = u_func(x, y, t)
v(x, y, t) = v_func(x, y, t)
winds = (u=u, v=v)


grid = TwoDGrid(100e3, 51, 100e3, 51)
mesh = TwoDGridMesh(grid, skip=1);
gn = TwoDGridNotes(grid);

Revise.retry()


#ProfileView.@profview 
#ProfileView.@profview 
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q, input=true, dissipation=true);

# define setting and standard initial conditions
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT);
#WindSeamin = FetchRelations.MinimalWindsea(u(0, 0, 0), v(0, 0, 0), DT / 2)
#WindSeamin = FetchRelations.get_initial_windsea(u(0, 0, 0), v(0, 0, 0), DT/5)
lne_local = log(WindSeamin["E"])
cg_u_local = WindSeamin["cg_bar_x"]
cg_v_local = WindSeamin["cg_bar_y"]

ODE_settings = PW.ODESettings(
    Parameters=ODEpars,
    # define mininum energy threshold
    log_energy_minimum=lne_local,#log(FetchRelations.Eⱼ(0.1, DT)),
    #maximum energy threshold
    log_energy_maximum=log(27),#log(17),  # correcsponds to Hs about 16 m
    saving_step=dt_ODE_save,
    timestep=DT,
    total_time=T = 6days,
    adaptive=true,
    dt=1e-3, #60*10, 
    dtmin=1e-4, #60*5, 
    force_dtmin=true,
    callbacks=nothing,
    save_everystep=false)


# plt.heatmap(gn.x / 1e3, gn.y / 1e3, transpose(u.(mesh.x, mesh.y, 0)))
# plt.heatmap(gn.x / 1e3, gn.y / 1e3, transpose(v.(mesh.x, mesh.y, 0)))
# %% build model


Revise.retry()

wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    layers=10,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",  # default_ODE_parameters
    periodic_boundary=false,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)


size(wave_model.State)
typeof(wave_model.State) 



#particle_system
defaults = ParticleDefaults(0.1, 0.1, 0.1, 0.1, 0.1)

P1 =InitParticleInstance(particle_system, defaults, ODE_settings, (1, 2), true, true)


# 1D
ParticleInstance(1, 2.1, P1.ODEIntegrator, true, true)
# 3d
ParticleInstance2D((1, 2), (2.1, 3.1), P1.ODEIntegrator, true, true)
ParticleInstance((1, 2), (2.1, 3.1), P1.ODEIntegrator, true, true)
# 2d
ParticleInstance2DLayer((1, 2, 1), (2.1, 3.1), P1.ODEIntegrator, true, true)
ParticleInstance((1, 2, 3), (2.1, 3.1), P1.ODEIntegrator, true, true)


### build Simulation
#wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=4hours)#1hours)


wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=6hour)#1hours)
initialize_simulation!(wave_simulation)


#init_state_store!(wave_simulation, save_path)
#wave_simulation.model.MovieState = wave_simulation.model.State

@time run!(wave_simulation, cash_store=true, debug=true)
#reset_simulation!(wave_simulation)
# run simulation
#ProfileView.@profview run!(wave_simulation, cash_store=true, debug=true)


istate = wave_simulation.store.store[end];
p1 = plt.heatmap(gn.x / 1e3, gn.y / 1e3, istate[:, :, 1])

