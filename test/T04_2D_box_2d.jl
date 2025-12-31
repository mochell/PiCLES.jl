ENV["JULIA_INCREMENTAL_COMPILE"]=true
using Pkg
Pkg.activate(".")

#using Plots
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
# debugging:
#using ProfileView

# %%
save_path = "plots/tests/T04_box_2d/"
mkpath(save_path)

# % Parameters
U10,V10           = 10.0, 10.0
dt_ODE_save       = 30minutes
DT                = 30minutes
# version 3


# function to define constants 


ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)


# u(x, y, t) = 0.01 + U10 * sin(t / (6 * 60 * 60 * 2π)) * sin(x / 50e3) * sin(y / 50e3)
# v(x, y, t) = 0.01 - V10 * cos(t / (6*60*60 * 2π) ) * sin(x / 50e3) * sin(y / 50e3)
u_std= 2e3 *1
v_std= 2e3 *1
u_func(x, y, t) = U10  * exp( - (x - 5e3)^2/ u_std^2) * exp( - (y - 5e3)^2/ v_std^2) *  sin(t*2 / (1 * 60 * 60 * 2π))
v_func(x, y, t) = V10  * exp( - (x - 5e3)^2/ u_std^2) * exp( - (y - 5e3)^2/ v_std^2) *  cos(t*2 / (1 * 60 * 60 * 2π) )   

# u_func(x, y, t) = 0.1 + IfElse.ifelse.( sin(t * 6 / (1 * 60 * 60 * 2π)) > 0 , 
#                 sin(t * 6 / (1 * 60 * 60 * 2π)) *U10 * exp(-(x - 5e3)^2 / u_std^2) * exp(-(y - 5e3)^2 / v_std^2),
#                                         0.1) 
# v_func(x, y, t) = 0.1 + IfElse.ifelse.(sin(t * 3 / (1 * 60 * 60 * 2π)) > 0,
#                             0.0,
#                             -0.0)

# u_func(x, y, t) = IfElse.ifelse.(x .< 5e3, U10, 0.2) + y * 0 + t * 0
# v_func(x, y, t) = (IfElse.ifelse.(x .< 5e3, V10, 0.2) + y * 0) .* cos(t * 3 / (1 * 60 * 60 * 2π))

# this shuold hopefully work
# u(x, y, t) = x * 0 + y * 0 + t * 0/ DT + 5.0
# v(x, y, t) = x * 0 + y * 0 + t * 0/ DT + 10.0

u(x, y, t) = u_func(x, y, t)
v(x, y, t) = v_func(x, y, t)
winds = (u=u, v=v)

typeof(winds.u)
typeof(winds.u(1e3, 1e3, 11))
#typeof(u_func(1e3, 1e3, 11))
#typeof(winds.u(x,y,t))

# u2 = winds.u(x, y, t)
# typeof(u2)
# typeof(winds.u(x, y, t))
# %%

grid = TwoDGrid(10e3, 31, 10e3, 31)
mesh = TwoDGridMesh(grid, skip=1);
gn = TwoDGridNotes(grid);

#heatmap( v.(mesh.x, mesh.y, 0) )


Revise.retry()

# define variables based on particle equation

#ProfileView.@profview 
#ProfileView.@profview 
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q, input=true, dissipation=true);
#particle_equations = PW3.particle_equations_vec5(u, v, u, v, γ=Const_ID.γ, q=Const_ID.q);

# define V4 parameters absed on Const NamedTuple:
default_ODE_parameters = (r_g = r_g0, C_α = Const_Scg.C_alpha, 
                                    C_φ = Const_ID.c_β, C_e = Const_ID.C_e, g= 9.81 );

# define setting and standard initial conditions
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT );
#WindSeamin = FetchRelations.MinimalWindsea(u(0, 0, 0), v(0, 0, 0), DT / 2)
#WindSeamin = FetchRelations.get_initial_windsea(u(0, 0, 0), v(0, 0, 0), DT/5)
lne_local = log(WindSeamin["E"])
cg_u_local = WindSeamin["cg_bar_x"]
cg_v_local = WindSeamin["cg_bar_y"]

ODE_settings    = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=lne_local,#log(FetchRelations.Eⱼ(0.1, DT)),
    #maximum energy threshold
    log_energy_maximum=log(27),#log(17),  # correcsponds to Hs about 16 m
    saving_step=dt_ODE_save,
    timestep=DT,
    total_time=T=6days,
    adaptive=true,
    dt=1e-3, #60*10, 
    dtmin=1e-4, #60*5, 
    force_dtmin=true,
    callbacks=nothing,
    save_everystep=false)


default_particle = ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0)

# Define grid
#grid = TwoDGrid(150e3, 50, 150e3, 50)

# gridmesh = [(i, j) for i in gn.x, j in gn.y];
# mesh.x = [i for i in gn.x, j in gn.y];
# mesh.y = [j for i in gn.x, j in gn.y];
# # mesh.x = mesh.x[1:3:end, 1:3:end];
# # mesh.y = mesh.y[1:3:end, 1:3:end];


# plt.heatmap(gn.x / 1e3, gn.y / 1e3, transpose(u.(mesh.x, mesh.y, 0)))

# plt.heatmap(gn.x / 1e3, gn.y / 1e3, transpose(v.(mesh.x, mesh.y, 0)))
# %% build model
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


### build Simulation
#wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=4hours)#1hours)
wave_simulation = Simulation(wave_model, Δt=20minutes, stop_time=1hour)#1hours)
initialize_simulation!(wave_simulation)


#init_state_store!(wave_simulation, save_path)
#wave_simulation.model.MovieState = wave_simulation.model.State

@time run!(wave_simulation, cash_store=true, debug=true)
#reset_simulation!(wave_simulation)
# run simulation
#ProfileView.@profview run!(wave_simulation, cash_store=true, debug=true)


istate = wave_simulation.store.store[5];
p1 = plt.heatmap(gn.x / 1e3, gn.y / 1e3, istate[:, :, 1])


# %%
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
index, weight  = PIC.compute_weights_and_index(grid, PF.Particle.ODEIntegrator.u[4], PF.Particle.ODEIntegrator.u[5])


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

cg= GetGroupVelocity(istate)
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

# # heatmap(wave_simulation.model.State[:, :, 2])
# # heatmap(wave_simulation.model.State[:, :, 3])


#sqrt( wave_simulation.model.winds.v.(mesh.x, mesh.y, 0.0)^2 + wave_simulation.model.winds.u.(mesh.x, mesh.y, 0.0)^2)

# %%
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

fig, n = init_movie_2D_box_plot(wave_simulation, name_string= "Rotating Stationary Winds")

#wave_simulation.stop_time += 1hour
N = 250
record(fig, save_path*"T02_2D_totating_winds_nonper.gif", 1:N, framerate=10) do i
    @info "Plotting frame $i of $N..."
    #@time for _ = 1:10
    #run!(wave_simulation, store=false)
    @info wave_simulation.model.clock
    movie_time_step!(wave_simulation.model, wave_simulation.Δt)
    #coupled_time_step!(ocean_simulation, ice_simulation)
    #end
    n[] = 1
end
