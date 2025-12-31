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
#using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, ProjetionKernel, TwoDCartesianGridStatistics
using PiCLES.Grids.TripolarGridMOM6: TripolarGridMOM6

using PiCLES.Models.WaveGrowthModels2D

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units
import Oceananigans.Utils: prettytime

using PiCLES.Architectures
using PiCLES.Architectures: AbstractGridStatistics, CartesianGridStatistics

using PiCLES.Operators.core_2D: ParticleDefaults as ParticleDefaults2D

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


using PiCLES.Plotting: PlotState_DoubleGlobe, PlotState_SingleGlobe, PlotState_DoubleGlobeSeam, OrthographicTwoMaps, OrthographicTwoMapsSeam

using Plots

using Makie

using PiCLES.Operators.mapping_2D: reset_PI_u!, ParticleToNode!
# %%

Revise.retry()
plot_path_base = "plots/tests/T03_PIC_tripolar_land/"
mkpath(plot_path_base)
pwd()
# % Parameters
U10, V10 = 15.0, -10.0
DT = 20minutes

ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)
# u(x, y, t) = IfElse.ifelse.(x .< 250e3, U10, 0.00) + y * 0.0 + t * 0.0
# v(x, y, t) = IfElse.ifelse.(x .< 250e3, V10, 0.00) + y * 0.0 + t * 0.0 .* cos(t * 5 / (1 * 60 * 60 * 2π))

u(x, y, t) = U10 + x * 0.0 + y * 0.0 + t * 0.0
v(x, y, t) = (V10 + x * 0.0 + y * 0.0) .* cos(t * 5 / (1 * 60 * 60 * 2π))
winds = (u=u, v=v)

Revise.retry()


load_path = "PiCLES/src/Grids/files/";
gridd = TripolarGridMOM6.MOM6GridMesh(load_path * "ocean_hgrid_221123.nc", 6; mask_radius=5);

fig = Figure(size=(900, 500), fontsize=22)
OrthographicTwoMapsSeam(fig, gridd.data.x, gridd.data.y, gridd.data.angle_dx)
# heatmap(gridd.data.angle_dx)
fig

# %%
Plots.heatmap(gridd.data.x[:, 1], gridd.data.y[1, :], transpose(gridd.data.mask))

function plot_particle_collection(wave_model; lims = [Nothing, Nothing, Nothing])
    particles = wave_model.ParticleCollection
    p = Plots.plot(layout=(3, 2), size=(1200, 1000))
    Plots.heatmap!(p, transpose(particles.on), subplot=1, title="on | iter=" * string(wave_model.clock.iteration))
    Plots.heatmap!(p, transpose(particles.boundary), subplot=2, title="boundary")

    sE = wave_model.State[:, :, 1]
    sE[wave_model.grid.data.mask.==0] .= NaN
    sE[wave_model.grid.data.mask.==2] .= NaN
    cmax_lim = lims[1] == Nothing ? maximum(sE) : lims[1]
    Plots.heatmap!(p, transpose(sE), subplot=3, title="State: Energy", clims=(0, cmax_lim))

    sm1 = wave_model.State[:, :, 2]
    sm1[wave_model.grid.data.mask.==0] .= NaN
    sm1[wave_model.grid.data.mask.==2] .= NaN
    cmax_lim = lims[2] == Nothing ? maximum(sE) : lims[2]
    Plots.heatmap!(p, transpose(sm1), subplot=4, title="State: x momentum ", clims=(0, cmax_lim))

    sm2 = wave_model.State[:, :, 3]
    sm2[wave_model.grid.data.mask.==0] .= NaN
    sm2[wave_model.grid.data.mask.==2] .= NaN
    cmax_lim = lims[3] == Nothing ? maximum(sE) : lims[3]
    Plots.heatmap!(p, transpose(sm2), subplot=6, title="State: y momentum ", clims = (0, cmax_lim))
    # title = plot!(title="Plot title", grid=false, showaxis=false, bottom_margin=-50Plots.px)
    display(p)
end

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



Revise.retry()

default_windsea = FetchRelations.get_initial_windsea(2.0, -2.0, DT, particle_state=true)
# default_particle = ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0)
wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=gridd,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type=ParticleDefaults2D(default_windsea[1], default_windsea[2], default_windsea[3], 0.0, 0.0),
    #ParticleDefaults2D(log(2), 0.0, 0.0, 0.0, 0.0), #"wind_sea",  # default_ODE_valuves
    periodic_boundary=false,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)


# ### build Simulation
TT = 20hours
wave_simulation = Simulation(wave_model, Δt=60minutes, stop_time=TT)#1hours)
initialize_simulation!(wave_simulation)

plot_particle_collection(wave_model; lims=[0.1, 0.015, 0.015])


# for PI in wave_simulation.model.ParticleCollection[25:40, 130:end]
#     #reset_PI_u!(PI, ui= [ log( (5/4)^2) , 5.0 , 10.0, 0.0, 0.0])
#     reset_PI_u!(PI, ui=FetchRelations.get_initial_windsea(0.0, -12.0, 4hour, particle_state=true))
#     ParticleToNode!(PI, wave_simulation.model.State, wave_simulation.model.grid, wave_simulation.model.periodic_boundary)
# end

for PI in wave_simulation.model.ParticleCollection[170:end-4, 60:115]
    #reset_PI_u!(PI, ui= [ log( (5/4)^2) , 5.0 , 10.0, 0.0, 0.0])
    reset_PI_u!(PI, ui=FetchRelations.get_initial_windsea(15.0, 00.0, 4hour, particle_state=true))
    ParticleToNode!(PI, wave_simulation.model.State, wave_simulation.model.grid, wave_simulation.model.periodic_boundary)
end

# for PI in wave_simulation.model.ParticleCollection[170:end-4, 105:115]
#     #reset_PI_u!(PI, ui= [ log( (5/4)^2) , 5.0 , 10.0, 0.0, 0.0])
#     reset_PI_u!(PI, ui=FetchRelations.get_initial_windsea(15.0, 00.0, 4hour, particle_state=true))
#     ParticleToNode!(PI, wave_simulation.model.State, wave_simulation.model.grid, wave_simulation.model.periodic_boundary)
# end


for PI in wave_simulation.model.ParticleCollection[4:20, 25:35]
    #reset_PI_u!(PI, ui= [ log( (5/4)^2) , 5.0 , 10.0, 0.0, 0.0])
    reset_PI_u!(PI, ui=FetchRelations.get_initial_windsea(-15.0, 00.0, 4hour, particle_state=true))
    ParticleToNode!(PI, wave_simulation.model.State, wave_simulation.model.grid, wave_simulation.model.periodic_boundary)
end


# for PI in wave_simulation.model.ParticleCollection[25:35, 140:end-4]
#     #reset_PI_u!(PI, ui= [ log( (5/4)^2) , 5.0 , 10.0, 0.0, 0.0])
#     reset_PI_u!(PI, ui=FetchRelations.get_initial_windsea(15.0, 0.0, 4hour, particle_state=true))
#     ParticleToNode!(PI, wave_simulation.model.State, wave_simulation.model.grid, wave_simulation.model.periodic_boundary)
# end

# for PI in wave_simulation.model.ParticleCollection[105:120, 140:end-4]
#     #reset_PI_u!(PI, ui= [ log( (5/4)^2) , 5.0 , 10.0, 0.0, 0.0])
#     reset_PI_u!(PI, ui=FetchRelations.get_initial_windsea(0, 15.0, 4hour, particle_state=true))
#     ParticleToNode!(PI, wave_simulation.model.State, wave_simulation.model.grid, wave_simulation.model.periodic_boundary)
# end


for PI in wave_simulation.model.ParticleCollection[135:150, 140-4:end-2-4]
    #reset_PI_u!(PI, ui= [ log( (5/4)^2) , 5.0 , 10.0, 0.0, 0.0])
    reset_PI_u!(PI, ui=FetchRelations.get_initial_windsea(-15.0, 10.0, 6hour, particle_state=true))
    ParticleToNode!(PI, wave_simulation.model.State, wave_simulation.model.grid, wave_simulation.model.periodic_boundary)
end

#plot_particle_collection(wave_simulation.model; lims=[0.2, 0.1, 0.015])





# for PI in wave_simulation.model.ParticleCollection[end-30:end-15, 10:end]
#     #reset_PI_u!(PI, ui= [ log( (5/4)^2) , 5.0 , 10.0, 0.0, 0.0])
#     reset_PI_u!(PI, ui=FetchRelations.get_initial_windsea(15.0, 0.0, 4hour, particle_state=true))
#     ParticleToNode!(PI, wave_simulation.model.State, wave_simulation.model.grid, wave_simulation.model.periodic_boundary)
# end

plot_particle_collection(wave_simulation.model; lims=[0.2, 0.1, 0.015])

fig = PlotState_DoubleGlobeSeam(wave_simulation.model, scaled=true)


# %%
# # fig = PlotState_DoubleGlobeSeam(wave_simulation.model, scaled=true)

# for i in 1:1:20
#     TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt)
#     if i % 10 == 0
#         # plot_particle_collection(wave_simulation.model; lims=[0.2, 0.1, 0.015])
#         fig =PlotState_DoubleGlobeSeam(wave_simulation.model, scaled=true)
#         fig
#         sleep(2)
#     end
#     wave_simulation.model.State[:, :, :] .= 0.0
# end

Gc = x -> 4.0
Gc(2.0)

fig = PlotState_DoubleGlobeSeam(wave_simulation.model, scaled=true)
# plot_particle_collection(wave_simulation.model; lims=[0.2, 0.1, 0.015])

for i in 1:1:12
    run!(wave_simulation, cash_store=true, debug=false)
    fig = PlotState_DoubleGlobeSeam(wave_simulation.model, scaled=true)
    display(fig)

    # plot_particle_collection(wave_simulation.model)#; lims=[0.2, 0.1, 0.015])
    wave_simulation.stop_time = wave_simulation.stop_time + TT
    sleep(0.5)
end

