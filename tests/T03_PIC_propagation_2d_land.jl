ENV["JULIA_INCREMENTAL_COMPILE"] = true
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
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, ProjectionKernel, TwoDCartesianGridStatistics

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
U10, V10 = 15.0, -10.0
DT = 20minutes

ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)


u(x, y, t) = IfElse.ifelse.(x .< 250e3, U10, 0.00) + y * 0.0 + t * 0.0
v(x, y, t) = IfElse.ifelse.(x .< 250e3, V10, 0.00) + y * 0.0 + t * 0.0 .* cos(t * 5 / (1 * 60 * 60 * 2π))

# u(x, y, t) = U10 + x * 0.0 + y * 0.0 + t * 0.0
# v(x, y, t) = V10 + x * 0.0 + y * 0.0 + t * 0.0
winds = (u=u, v=v)

grid = TwoDCartesianGridMesh(400e3, 41, 200e3, 21, periodic_boundary=(false, true))
heatmap(grid.data.x[:, 1], grid.data.y[1, :], transpose(grid.data.mask))

# % Make fake mask
mask = ones(Bool, size(grid.data.x)) # 1 is ocean, 0 is land (?)
mask[10:20, 5:10] .= 0
#mask  = .!mask # to make one active block
gridstats_mask = TwoDCartesianGridMesh(grid.stats; mask=mask)
grid = TwoDCartesianGridMesh(gridstats_mask, grid.stats, ProjectionKernel)


heatmap(grid.data.x[:,1], grid.data.y[1,:], transpose(grid.data.mask))
# heatmap(transpose(v.(grid.data.x, grid.data.y, 0)))

# %%
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q, 
    propagation=true,
    input=false, 
    dissipation=false,
    peak_shift=false,
    direction=true,
    );

default_ODE_parameters = (r_g=ODEpars.r_g, C_α=Const_Scg.C_alpha,
        C_φ=Const_ID.c_β, C_e=Const_ID.C_e, g=9.81);#, M=M);

# define setting and standard initial conditions
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT);
lne_local = log(WindSeamin["E"])
cg_u_local = WindSeamin["cg_bar_x"]
cg_v_local = WindSeamin["cg_bar_y"]

ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=lne_local,#log(FetchRelations.Eⱼ(0.1, DT)),
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
    p = plot(layout=(3, 2), size=(1200, 1000))
    heatmap!(p, transpose(particles.on), subplot=1, title="on | iter=" * string(wave_model.clock.iteration))
    heatmap!(p, transpose(particles.boundary), subplot=2, title="boundary")
    heatmap!(p, transpose(wave_model.State[:, :, 1]), subplot=3, title="State: Energy", clims=(0, NaN))
    heatmap!(p, transpose(wave_model.State[:, :, 2]), subplot=4, title="State: x momentum ", clims=(0, NaN))
    heatmap!(p, transpose(wave_model.State[:, :, 3]), subplot=6, title="State: y momentum ")
    # title = plot!(title="Plot title", grid=false, showaxis=false, bottom_margin=-50Plots.px)
    display(p)
end

# %%

Revise.retry()

default_windsea = FetchRelations.get_initial_windsea(U10, V10, DT,particle_state=true )
default_particle = ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0)
wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type=ParticleDefaults2D(default_windsea[1], default_windsea[2], default_windsea[3], 0.0, 0.0),
    #ParticleDefaults2D(log(2), 0.0, 0.0, 0.0, 0.0), #"wind_sea",  # default_ODE_valuves
    periodic_boundary=true,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

#wave_model.ODEsettings.wind_min_squared = 0.0
#wave_model.minimal_state = 2 * wave_model.minimal_state

# ### build Simulation
wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=4hours)#1hours)
initialize_simulation!(wave_simulation)
plot_particle_collection(wave_model)

# %% Single Adance step
PI = wave_simulation.model.ParticleCollection[2,1]
step!(PI.ODEIntegrator, DT, true)

model = wave_simulation.model
mapping_2D.advance!(PI, model.State, FailedCollection,
    model.grid, model.winds, wave_simulation.Δt,
    model.ODEsettings.log_energy_maximum,
    model.ODEsettings.wind_min_squared,
    model.periodic_boundary,
    model.ODEdefaults)


# %% Adance only test
model = wave_simulation.model
FailedCollection = Vector{AbstractMarkedParticleInstance}([])
for a_particle in model.ParticleCollection[findall(model.grid.data.mask .== 1)]
    @info a_particle.position_ij
    mapping_2D.advance!(a_particle, model.State, FailedCollection,
        model.grid, model.winds, wave_simulation.Δt,
        model.ODEsettings.log_energy_maximum,
        model.ODEsettings.wind_min_squared,
        model.periodic_boundary,
        model.ODEdefaults)

        # ParticleToNode!(a_particle, model.State, model.grid, model.periodic_boundary)

        plot_particle_collection(model)

        sleep(0.001)
end



# %% timstepper test

Revise.retry()
plot_particle_collection(wave_model)

for i in 1:1:180
    TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt)

    if i%8 == 0
        plot_particle_collection(wave_simulation.model)
        sleep(0.4)
    end
    wave_simulation.model.State[:, :, :] .= 0.0

end

