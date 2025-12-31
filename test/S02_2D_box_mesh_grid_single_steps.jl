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
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, ProjetionKernel, TwoDCartesianGridStatistics

using PiCLES.Models.WaveGrowthModels2D

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units
import Oceananigans.Utils: prettytime

using PiCLES.Architectures
using PiCLES.Architectures: AbstractGridStatistics, CartesianGridStatistics

using PiCLES.custom_structures: N_NonPeriodic, N_Periodic, N_TripolarNorth

using PiCLES.Operators.core_2D: ParticleDefaults as ParticleDefaults2D

#using GLMakie

using Plots

using PiCLES.Operators.core_2D: GetGroupVelocity, speed
using PiCLES.Plotting.movie: init_movie_2D_box_plot

using Revise

using StructArrays
using BenchmarkTools



# %%
save_path = "plots/tests/S02_box_2D_mesh_grid/"
mkpath(save_path)
pwd()
# % Parameters
U10, V10 = 15.0, 10.0
DT = 20minutes

ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)

# u(x, y, t) = 0.01 + U10 * sin(t / (6 * 60 * 60 * 2π)) * sin(x / 50e3) * sin(y / 50e3)
# v(x, y, t) = 0.01 - V10 * cos(t / (6*60*60 * 2π) ) * sin(x / 50e3) * sin(y / 50e3)
u_std = 200e3 * 1
v_std = 200e3 * 1
# u_func(x, y, t) = U10 #* exp(-(x - 5e3)^2 / u_std^2) * exp(-(y - 5e3)^2 / v_std^2) * sin(t * 2 / (1 * 60 * 60 * 2π))
# v_func(x, y, t) = V10 #* exp(-(x - 5e3)^2 / u_std^2) * exp(-(y - 5e3)^2 / v_std^2) * cos(t * 2 / (1 * 60 * 60 * 2π))
# u_func(x, y, t) = U10 * exp(-(x - 250e3)^2 / u_std^2) * exp(-(y - 250e3)^2 / v_std^2) #* sin(t * 2 / (1 * 60 * 60 * 2π))
# v_func(x, y, t) = V10 * exp(-(x - 250e3)^2 / u_std^2) * exp(-(y - 250e3)^2 / v_std^2) #* cos(t * 2 / (1 * 60 * 60 * 2π))

u_func(x, y, t) = IfElse.ifelse.(x .< 250e3, U10, 0.00) + y * 0.0 + t * 0.0
v_func(x, y, t) = (IfElse.ifelse.(x .< 250e3, V10, 0.00) + y * 0.0) + t*0.0 # .* cos(t * 5 / (1 * 60 * 60 * 2π))

u(x, y, t) = u_func(x, y, t)
v(x, y, t) = v_func(x, y, t)
winds = (u=u, v=v)

Revise.retry()
grid = TwoDCartesianGridMesh(500e3, 51, 400e3, 41; periodic_boundary=(true, false))

grid.stats.Nx
grid.stats.Ny


# % Make fake mask
mask = ones(Bool, size(grid.data.x)); # 1 is ocean, 0 is land (?)
mask[20:35, 20:35] .= 0
# reset mesh with amsk 
gridstats_mask = TwoDCartesianGridMesh(grid.stats; mask=mask)
grid = TwoDCartesianGridMesh(gridstats_mask, grid.stats, ProjetionKernel)

# heatmap(grid.data.x[:,1], grid.data.y[1,:], transpose(grid.data.mask))
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

using PiCLES.Operators.TimeSteppers: time_step!

function plot_particle_collection(wave_model)
    particles = wave_model.ParticleCollection
    p = plot(layout=(2, 2), size=(1200, 800))
    heatmap!(p, transpose(particles.on), subplot=1, title="on | iter=" * string(wave_model.clock.iteration))
    heatmap!(p, transpose(particles.boundary), subplot=2, title="boundary")
    heatmap!(p, transpose(wave_model.State[:, :, 1]), subplot=3, title="State: Energy")
    heatmap!(p, transpose(wave_model.State[:, :, 2]), subplot=4, title="State: x momentum ")
    # title = plot!(title="Plot title", grid=false, showaxis=false, bottom_margin=-50Plots.px)
    display(p)
end

Revise.retry()

default_particle = ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0)

wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type=ParticleDefaults2D(log(5), 5.0, 5.0, 0.0,0.0), #"wind_sea",  # default_ODE_valuves
    periodic_boundary=false,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)


# ### build Simulation
wave_simulation = Simulation(wave_model, Δt=20minutes, stop_time=4hours)#1hours)
# wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=30minutes)#1hours)
initialize_simulation!(wave_simulation)

plot_particle_collection(wave_model)

typeof(wave_model.ODEdefaults)
# @time run!(wave_simulation, cash_store=true, debug=false)
# #reset_simulation!(wave_simulation)
# # run simulation
# #ProfileView.@profview run!(wave_simulation, cash_store=true, debug=true)
plot(wave_model.State[:, 10, :])

for i in 1:1:5
    TimeSteppers.time_step!(wave_model, wave_simulation.Δt)
end

plot_particle_collection(wave_model) 
plot(wave_model.State[:, 10, :])


# %%  test node to particle
using PiCLES.Operators.mapping_2D: NodeToParticle!


# size(wave_simulation.store.store)

# for ii in 1:1:length(wave_simulation.store.store)
#     istate = wave_simulation.store.store[ii];
#     p1 = plt.heatmap(grid.data.x[:,1], grid.data.y[1,:], transpose(istate[:, :, 1]))
#     #plt.savefig(save_path * "T02_2D_box_mesh_grid_$(ii).png")
#     display(p1)
#     sleep(0.5)
# end
# # istate = wave_simulation.store.store[100];
# # p1 = plt.heatmap(grid.data.x[:,1] / 1e3, grid.data.y[1, :] / 1e3, transpose(istate[:, :, 1]))

# %%
Revise.retry()

# %%








# %% ----------- up to here ----------------
# show all Failed particles
using PiCLES.Utils: ParticleTools

Revise.retry()
DD = ParticleTools.ParticleToDataframe(wave_simulation.model.FailedCollection)

DD_stats = ParticleTools.ParticleStatsToDataframe(wave_simulation.model.FailedCollection)

#sum(DD_stats.boundary)

function plot_state_and_error_points(wave_simulation, FailedTable)
    plt.plot()

    p1 = plt.heatmap(gn.x / 1e3, gn.y / 1e3, transpose(wave_simulation.model.State[:, :, 1]))

    for xy in FailedTable.position_xy
        x = xy[1] / 1e3
        y = xy[2] / 1e3
        plt.scatter!((x, y), color="red", label="wind")
    end

    plt.plot!(legend=:none, title="c_g", ylabel="cg m/s", xlabel="position index x") |> display
end

plot_state_and_error_points(wave_simulation, DD_stats)
# %%

## testing single ODES
PF = wave_simulation.model.FailedCollection[end]
PF.Particle
#PF.Particle.ODEIntegrator.u = [-15.441984291167334, 0.006707651986269529, 0.003889144130029894, 4000.0, 2333.3333333333335]
PF.Particle.ODEIntegrator.u
PF.Particle.ODEIntegrator.t
PF.Particle.ODEIntegrator.sol.t
PF.Particle.ODEIntegrator.sol.u
PF.Particle.ODEIntegrator.sol.prob.u0

using PiCLES.Operators.core_2D: Get_u_FromShared, GetVariablesAtVertex
S_state = Get_u_FromShared(PF.Particle, wave_simulation.model.MovieState)
ui = GetVariablesAtVertex(S_state, PF.Particle.position_xy[1], PF.Particle.position_xy[2])

usel = PF.Particle.ODEIntegrator.sol.prob.u0
t_sel = PF.Particle.ODEIntegrator.sol.u
index, weight = PIC.compute_weights_and_index(grid, usel[4], usel[5])

using PiCLES: ParticleInCell as PIC
index, weight = PIC.compute_weights_and_index(grid, PF.Particle.ODEIntegrator.u[4], PF.Particle.ODEIntegrator.u[5])


# try manual solution with the same initial conditions
using DifferentialEquations

#PF.Particle.ODEIntegrator.u = ui
ui2 = copy(PF.Particle.ODEIntegrator.sol.prob.u0)
reinit!(PF.Particle.ODEIntegrator, ui2, erase_sol=false, reset_dt=false, reinit_cache=true)
u_modified!(PF.Particle.ODEIntegrator, true)
#auto_dt_reset!(PF.Particle.ODEIntegrator)
PF.Particle.ODEIntegrator.t
# set_t!(PF.Particle.ODEIntegrator, PF.Particle.ODEIntegrator.sol.t[end])

wind_tuple = winds.u(PF.Particle.position_xy[1], PF.Particle.position_xy[2], PF.Particle.ODEIntegrator.t), winds.v(PF.Particle.position_xy[1], PF.Particle.position_xy[2], PF.Particle.ODEIntegrator.t)
step!(PF.Particle.ODEIntegrator, DT, true)
PF.Particle.ODEIntegrator.u


function set_u_and_t!(integrator, u_new, t_new)
    integrator.u = u_new
    integrator.t = t_new
end

#5min
DT = 5min#wave_simulation.model.ODEsettings.timestep
ui2
set_u_and_t!(PF.Particle.ODEIntegrator, ui2, 0.0)
auto_dt_reset!(PF.Particle.ODEIntegrator)
step!(PF.Particle.ODEIntegrator, DT, true)
PF.Particle.ODEIntegrator.u

#index, weight = PIC.compute_weights_and_index(grid, PF.Particle.ODEIntegrator.u[4], PF.Particle.ODEIntegrator.u[5])

#PF.Particle.ODEIntegrator.sol
#wave_simulation.model.winds.u()

# %%
#PF.Particle.boundary
Revise.retry()

istate = wave_simulation.store.store[end];
#istate = wave_simulation.store.store[5];
#wave_simulation.model.ParticleCollection[105]

cg = GetGroupVelocity(istate)
p1 = plt.heatmap(gn.x / 1e3, gn.y / 1e3, istate[:, :, 1])

# %%
p1 = plt.heatmap(gn.x / 1e3, gn.y / 1e3, cg.c_x)

# %%
#pcolormesh in Julia 
# gn = TwoDGridNotes(grid);

# gridmesh = [(i,j) for i in gn.x, j in gn.y];
# mesh.x = [i for i in gn.x, j in gn.y];
# mesh.y = [j for i in gn.x, j in gn.y];
# mesh.x = mesh.x[1:3:end, 1:3:end];
# mesh.y = mesh.y[1:3:end, 1:3:end];

# %%
# %%

p1 = plt.heatmap(gn.x / 1e3, gn.y / 1e3, wave_simulation.model.State[:, :, 2])

# plt.quiver!(p1, mesh.x / 1e3, mesh.y / 1e3, quiver=(wave_simulation.model.State[1:3:end, 1:3:end, 2] * 20, wave_simulation.model.State[1:3:end, 1:3:end, 3] * 20), color=:red, scale_unit=:data, label="wind")
# # scale_unit options: :width, :height, :x, :y, :xy, :data

plt.contour(p1, gn.x / 1e3, gn.y / 1e3, u.(mesh.x, mesh.y, wave_simulation.model.clock.time), color="white")

# %%
using GLMakie
Revise.retry()
wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",  # default_ODE_parameters
    periodic_boundary=false,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

wave_simulation = Simulation(wave_model, Δt=5minutes, stop_time=4hours)#1hours)
initialize_simulation!(wave_simulation)

#reset_simulation!(wave_simulation, particle_initials=copy(wave_model.ODEdefaults))

fig, n = init_movie_2D_box_plot(wave_simulation, name_string="Rotating Stationary Winds masked")

#wave_simulation.stop_time += 1hour
N = 250
record(fig, save_path * "T02_2D_totating_winds_nonper.gif", 1:N, framerate=10) do i
    @info "Plotting frame $i of $N..."
    #@time for _ = 1:10
    #run!(wave_simulation, store=false)
    @info wave_simulation.model.clock
    movie_time_step!(wave_simulation.model, wave_simulation.Δt)
    #coupled_time_step!(ocean_simulation, ice_simulation)
    #end
    n[] = 1
end
