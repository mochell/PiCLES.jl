# %%
# ENV["JULIA_INCREMENTAL_COMPILE"] = true
using Pkg
Pkg.activate("PiCLES/")


import Plots as plt
using Setfield, IfElse

using PiCLES.ParticleSystems: particle_waves_v5 as PW

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleInstance, GetGroupVelocity
using PiCLES.Operators: TimeSteppers
using PiCLES.Simulations
using PiCLES.Operators.TimeSteppers: time_step!, movie_time_step!

using PiCLES.ParticleMesh: TwoDGrid, TwoDGridNotes, TwoDGridMesh
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, TwoDCartesianGridStatistics

using PiCLES.Models.WaveGrowthModels2D

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units
import Oceananigans.Utils: prettytime

using PiCLES.Architectures
using PiCLES.Architectures: AbstractGridStatistics, CartesianGridStatistics

using PiCLES.Operators.core_2D: ParticleDefaults as ParticleDefaults2D

#using GLMakie
using Plots

using PiCLES.Operators.core_2D: GetGroupVelocity, speed
using PiCLES.Plotting.movie: init_movie_2D_box_plot

using Revise

using StaticArrays
using StructArrays
using BenchmarkTools

using PiCLES.Grids

using PiCLES.Operators.TimeSteppers: time_step!
using DifferentialEquations
using PiCLES.Operators: mapping_2D



# %%

Revise.retry()
save_path = "plots/tests/S02_box_2D_mesh_grid/"
mkpath(save_path)
pwd()
# % Parameters
U10, V10 = 0.00, 00.0
DT = 20minutes
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)

u(x, y, t) = IfElse.ifelse.(x .< 250e3, U10, 0.00) + y * 0.0 + t * 0.0
v(x, y, t) = IfElse.ifelse.(x .< 250e3, V10, 0.00) + y * 0.0 + t * 0.0 .* cos(t * 5 / (1 * 60 * 60 * 2π))

# u(x, y, t) = U10 + x * 0.0 + y * 0.0 + t * 0.0
# v(x, y, t) = V10 + x * 0.0 + y * 0.0 + t * 0.0
winds = (u=u, v=v)

grid = TwoDCartesianGridMesh(400e3, 41, 200e3, 21; periodic_boundary=(false, true))
heatmap(grid.data.x[:, 1], grid.data.y[1, :], transpose(grid.data.mask))

# %%
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q, 
    propagation=true,
    input=false, 
    dissipation=false,
    peak_shift=false,
    direction=false,
    );

default_ODE_parameters = (r_g=ODEpars.r_g, C_α=Const_Scg.C_alpha,
        C_φ=Const_ID.c_β, C_e=Const_ID.C_e, g=9.81);#, M=M);

# define setting and standard initial conditions
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT);
lne_local  = round( log(WindSeamin["E"]) , digits=4)
cg_u_local = round( WindSeamin["cg_bar_x"] , digits=4)
cg_v_local = round(WindSeamin["cg_bar_y"], digits=4)

ParticleMin = FetchRelations.MinimalParticle(2, 0, DT)
# get_initial_windsea(2, 2, DT,particle_state=true )

ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=ParticleMin[1], #log(FetchRelations.Eⱼ(0.1, DT)),
    #maximum energy threshold
    log_energy_maximum=log(27),#log(17),  # correcsponds to Hs about 16 m
    saving_step=DT,
    timestep=DT,
    total_time=T = 6days,
    adaptive=true,
    dt=1e-3, #60*10, 
    dtmin=1e-4, #60*5, 
    force_dtmin=true,
    callbacks=nothing,
    save_everystep=false)


# %%
function plot_particle_collection(wave_model)
    particles = wave_model.ParticleCollection
    p = plot(layout=(3, 2), size=(1200, 1100))
    heatmap!(p, transpose(particles.on), subplot=1, title="on | iter=" * string(wave_model.clock.iteration))
    heatmap!(p, transpose(particles.boundary), subplot=2, title="boundary")
    heatmap!(p, transpose(wave_model.State[:, :, 1]), subplot=3, title="State: Energy", clims=(0, NaN))
    heatmap!(p, transpose(wave_model.State[:, :, 2]), subplot=4, title="State: x momentum ", clims=(0, NaN))
    heatmap!(p, transpose(wave_model.State[:, :, 3]), subplot=6, title="State: y momentum ")
    # title = plot!(title="Plot title", grid=false, showaxis=false, bottom_margin=-50Plots.px)
    plot!(p, aspect_ratio=:equal)
    display(p)
end


# %%

Revise.retry()

default_windsea = FetchRelations.get_initial_windsea(2, 2, DT,particle_state=true )
# default_particle = ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0)
wave_model = WaveGrowthModels2D.WaveGrowth2D(; 
    grid         =   grid,
    winds        =   winds,
    ODEsys       =   particle_system,
    ODEsets      =   ODE_settings,  # ODE_settings
    #ODEinit_type=ParticleDefaults2D(default_windsea[1]*0.1, default_windsea[2]*0.1, default_windsea[3]*0.1, 0.0, 0.0),
    ODEinit_type=ParticleDefaults(ParticleMin),
    #ParticleDefaults2D(log(2), 0.0, 0.0, 0.0, 0.0), #"wind_sea",  # default_ODE_valuves
    periodic_boundary=  true,
    boundary_type=   "same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie        =   true)

#wave_model.ODEsettings.wind_min_squared = 0.0
#wave_model.minimal_state = 2 * wave_model.minimal_state

using PiCLES.Operators.mapping_2D: reset_PI_u!, ParticleToNode!

# ### build Simulation
wave_simulation = Simulation(wave_model, Δt=20minutes, stop_time=4hours)#1hours)
initialize_simulation!(wave_simulation)
# plot_particle_collection(wave_model)


for PI in wave_simulation.model.ParticleCollection[5:15, 5:15]
    #reset_PI_u!(PI, ui= [ log( (5/4)^2) , 5.0 , 10.0, 0.0, 0.0])
    reset_PI_u!(PI, ui= FetchRelations.get_initial_windsea(10.0, 12.0, 4hour, particle_state=true) )
    ParticleToNode!(PI, wave_simulation.model.State, wave_simulation.model.grid, wave_simulation.model.periodic_boundary)
end

plot_particle_collection(wave_model)


# %% Advance only
wave_simulation.model.ParticleCollection[8, 8].ODEIntegrator.u
wave_simulation.model.ParticleCollection[5, 2].ODEIntegrator.u

FailedCollection = Vector{AbstractMarkedParticleInstance}([])
TimeSteppers.time_step!_advance(wave_simulation.model, wave_simulation.Δt, FailedCollection)

wave_simulation.model.ParticleCollection[20, 12].ODEIntegrator.u
wave_simulation.model.ParticleCollection[8, 8].ODEIntegrator.u
wave_simulation.model.ParticleCollection[5, 2].ODEIntegrator.u


TimeSteppers.time_step!_remesh(wave_simulation.model, wave_simulation.Δt)

wave_simulation.model.ParticleCollection[20, 12].ODEIntegrator.u
wave_simulation.model.ParticleCollection[8, 8].ODEIntegrator.u
wave_simulation.model.ParticleCollection[5, 2].ODEIntegrator.u

wave_simulation.model.State[5, 2, :]
wave_simulation.model.State[:,:,1]

#TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt)


# %% timstepper test
Revise.retry()
plot_particle_collection(wave_model)

for i in 1:1:180
    TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt)

    if i%8 == 0 
        plot_particle_collection(wave_simulation.model)
        sleep(0.02)
    end
    wave_simulation.model.State[:, :, :] .= 0.0

end


# %%
wave_simulation.model.ParticleCollection[20, 10]
wave_simulation.model.ParticleCollection[20, 10].ODEIntegrator.u

wave_simulation.model.grid.stats