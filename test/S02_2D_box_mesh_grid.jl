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


using Plots

using PiCLES.Operators.core_2D: GetGroupVelocity, speed
using PiCLES.Plotting.movie: init_movie_2D_box_plot

using Revise

using StructArrays
using BenchmarkTools

using GLMakie


# %%
save_path = "plots/tests/S02_box_2D_mesh_grid/"
mkpath(save_path)
pwd()
# % Parameters
U10, V10 = -15.0, 15.0#15.0
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

# u(x, y, t) = IfElse.ifelse.(x .< 250e3, U10, -3.00) + y * 0 + t * 0
# v(x, y, t) = (IfElse.ifelse.(x .< 250e3, V10, 0.00) + y * 0) + t *0 #.* cos(t * 5 / (1 * 60 * 60 * 2π))

# u(x, y, t) = IfElse.ifelse.(x .< 250e3, U10, 1.00) + y * 0 + t * 0
# v(x, y, t) = (IfElse.ifelse.(x .< 250e3, V10, 0.00) + y * 0) + t * 0 #.* cos(t * 5 / (1 * 60 * 60 * 2π))

u(x, y, t) = U10  + y * 0 + t * 0
v(x, y, t) = V10 + y * 0 + t * 0 #(V10 + y * 0) .* cos(t * 5 / (1 * 60 * 60 * 2π))
winds = (u=u, v=v)


Revise.retry()

grid = TwoDCartesianGridMesh(500e3, 51, 400e3, 41; angle=0.0, periodic_boundary=(false, true))

Plots.heatmap(grid.data.x[:, 1], grid.data.y[1, :], transpose(grid.data.mask))

# % Make fake mask
mask = ones(Bool, size(grid.data.x)) # 1 is ocean, 0 is land (?)
mask[20:35, 20:35] .= 0
# reset mesh with amsk 
gridstats_mask = TwoDCartesianGridMesh(grid.stats; mask=mask);
grid = TwoDCartesianGridMesh(gridstats_mask, grid.stats, ProjetionKernel)


Revise.retry()
grid.stats.Nx
grid.stats.Ny

Plots.heatmap(grid.data.x[:,1], grid.data.y[1,:], transpose(grid.data.mask))

# x_prime = ProjetionKernel(grid)[1, 1] .* grid.data.x + ProjetionKernel(grid)[1, 2] .* grid.data.y
# y_prime = ProjetionKernel(grid)[2, 1] .* grid.data.x + ProjetionKernel(grid)[2, 2] .* grid.data.y



# %%
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q, input=true, dissipation=true);
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


default_particle = ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0)

wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",  # default_ODE_parameters
    periodic_boundary=false,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)


Revise.retry()

# ### build Simulation
wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=4hours)#1hours)
# wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=30minutes)#1hours)
initialize_simulation!(wave_simulation);

Plots.heatmap(transpose(wave_model.ParticleCollection.on))
Plots.heatmap(transpose(wave_model.ParticleCollection.boundary))


# run simulation
# run!(wave_simulation, cash_store=true, debug=true)
# or, alternatively, make movie

fig, n = init_movie_2D_box_plot(wave_simulation)

#wave_simulation.stop_time += 1hour
N = 26 *5
# plot_name = "2D_box_masked_test_half_domain_diagonal_angle30"
plot_name = "2D_box_masked_test_half_domain_zonal"
record(fig, save_path * plot_name * ".gif", 1:N, framerate=10) do i
    @info "Plotting frame $i of $N..."
    @info wave_simulation.model.clock
    movie_time_step!(wave_simulation.model, wave_simulation.Δt)

    n[] = 1
end

# %%
