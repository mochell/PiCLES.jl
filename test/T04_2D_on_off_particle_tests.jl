
#using Plots
using Plots
using Setfield, IfElse

using PiCLES.ParticleSystems: particle_waves_v5 as PW

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleInstance, GetGroupVelocity
using PiCLES.Operators: TimeSteppers
using PiCLES.Simulations
using PiCLES.Operators.TimeSteppers: time_step!, movie_time_step!

using PiCLES
using PiCLES.Models.WaveGrowthModels2D

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units
import Oceananigans.Utils: prettytime

using PiCLES.Architectures


using PiCLES.Operators.core_2D: GetGroupVelocity, speed
using PiCLES.Plotting.movie: init_movie_2D_box_plot

# %%


save_path = "plots/tests/T04_2D_particle_on_off/"
mkpath(save_path)

##### basic parameters
# timestep
DT = 20minutes
# Characterstic wind velocities and std
U10, V10 = 20.0, 20.0

# Define basic ODE parameters
r_g0 = 0.85
Const_ID = PW.get_I_D_constant()

Const_Scg = PW.get_Scg_constants(C_alpha=-1.41, C_varphi=1.81e-5)


# define grid
grid = PiCLES.Grids.CartesianGrid.TwoDCartesianGridMesh(200e3, 41,  50e3, 11; periodic_boundary=(false, false))
grid.stats.Nx, grid.stats.Ny

# example user function
u_func(x, y, t) = U10 + x * 0 + y * 0 + t * 0
v_func(x, y, t) = V10 + x * 0 + y * 0 + t * 0

# provide function handles for ODE and Simulation in the right format
u(x, y, t) = u_func(x, y, t)
v(x, y, t) = v_func(x, y, t)
winds = (u=u, v=v)


# define ODE system and parameters
Revise.retry()
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q);
 

default_ODE_parameters = (r_g=r_g0, C_α=Const_Scg.C_alpha,
    C_φ=Const_ID.c_β, C_e=Const_ID.C_e, g=9.81)


Revise.retry()
# Default initial conditions based on timestep and chaeracteristic wind velocity
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT)
default_particle = ParticleDefaults(WindSeamin["lne"], WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0)

# ... and ODESettings
ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=WindSeamin["lne"],
    #maximum energy threshold
    log_energy_maximum=log(27),#log(17),  # correcsponds to Hs about 16 m
    saving_step=60,
    timestep=DT,
    total_time=T = 12days,
    dt=1e-3,
    dtmin=1e-4,
    dtmax=20minutes)


# %% half domain tests
Revise.retry()
#gridmesh = [(i, j) for i in [-10,10], j in  [0]]
#gridmesh = [(i, j) for i in [10], j in [0]]

U10 = -12
V10 = 0.0

@show U10, V10

x0 =50e3
Lx = (grid.stats.Nx.N - 1) * grid.stats.dx
# u_func(x, y, t) = IfElse.ifelse.(x .< x0, U10, U10 * (1 -x/Lx) ) + y * 0 + t * 0
# v_func(x, y, t) = IfElse.ifelse.(x .< x0, V10, V10 * (1 -x/Lx) ) + y * 0 + t * 0

# u_func(x, y, t) = IfElse.ifelse.(x .< x0, x*0+ 0, U10 * (x - x0) / (Lx-x0)) + y * 0 + t * 0
# v_func(x, y, t) = IfElse.ifelse.(x .< x0, x*0+ 0, V10 * (x - x0) / (Lx-x0)) + y * 0 + t * 0

u_func(x, y, t) = IfElse.ifelse.(x .< x0, x *0+  0.0 , U10 ) + y * 0 + t * 0
v_func(x, y, t) = IfElse.ifelse.(x .< x0, x *0 + 0.0 , V10 ) + y * 0 + t * 0
u(x, y, t) = u_func(x, y, t)
v(x, y, t) = v_func(x, y, t)
winds = (u=u, v=v)

#winds, u, v  =convert_wind_field_functions(u_func, v_func, x, y, t)
Revise.retry()
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)

# Define wave model
wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",  # default_ODE_parameters
    periodic_boundary=false,
    boundary_type="same",
    minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT), #
    minimal_state=FetchRelations.MinimalState(2, 2, DT) * 1,
    movie=true)


# %%
### build Simulation
Revise.retry()
wave_simulation = Simulation(wave_model, Δt=DT/2, stop_time=4hours)
initialize_simulation!(wave_simulation)

init_state_store!(wave_simulation, save_path)

#run!(wave_simulation, cash_store=true, debug=true)
run!(wave_simulation, store=true, cash_store=false, debug=false)

close_store!(wave_simulation)

# %% make movie 

# using GLMakie
# wave_simulation.model.MovieState

# fig = Figure()
# ax = Axis(fig[1, 1])
# hm = heatmap!(ax, wave_simulation.model.ParticleCollection.on)
# Colorbar(fig[1, 2], hm)
# display(fig)

# # %%
# Revise.retry()
# # or, alternatively, make movie
# plot_name="T02_2D_growing_U" * string(U10) * "_V" * string(V10)
# N=80
# axline=x0/1e3
# fig, n = init_movie_2D_box_plot(wave_simulation; resolution=(1300, 800), name_string=plot_name, aspect=3, axline=axline)

# #wave_simulation.stop_time += 1hour
# #N = 36
# #plot_name = "dummy"

# record(fig, save_path * plot_name * ".gif", 1:N, framerate=10) do i
#     @info "Plotting frame $i of $N..."
#     @info wave_simulation.model.clock
#     movie_time_step!(wave_simulation.model, wave_simulation.Δt)
#     n[] = 1
# end

# %% make step my step analysis. 
using Plots

function plot_particle_collection(wave_model)
    particles = wave_model.ParticleCollection
    p = plot(layout=(3, 2), size=(1200, 1100))
    heatmap!(p, transpose(particles.on), subplot=1, title="on | iter=" * string(wave_model.clock.iteration) * " | total energy = " * string(round(sum(wave_model.State[:, :, 1]), digits=4)))
    heatmap!(p, transpose(particles.boundary), subplot=2, title="boundary")
    heatmap!(p, transpose(wave_model.State[:, :, 1]), subplot=3, title="State: Energy", clims=(0, NaN))
    heatmap!(p, transpose(wave_model.State[:, :, 2]), subplot=4, title="State: x momentum ", clims=(0, NaN))
    heatmap!(p, transpose(wave_model.State[:, :, 3]), subplot=6, title="State: y momentum ")
    # title = plot!(title="Plot title", grid=false, showaxis=false, bottom_margin=-50Plots.px)
    plot!(p, aspect_ratio=:equal)
    display(p)
end

plot_particle_collection(wave_simulation.model)
# %%

Revise.retry()
wave_simulation = Simulation(wave_model, Δt=DT / 2, stop_time=4hours)
initialize_simulation!(wave_simulation)

# %%

for i in 1:1:200
    TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt)

    if i % 8 == 0
        plot_particle_collection(wave_simulation.model)
        sleep(0.02)
    end
    wave_simulation.model.State[:, :, :] .= 0.0

end

# %%
TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt, debug=false)
plot_particle_collection(wave_simulation.model)
wave_simulation.model.State[:, :, :] .= 0.0

# %%
@show wave_model.ParticleCollection[:, 6].on

@show wave_model.ParticleCollection[31, 2].position_ij[1]

# %%

@info wave_simulation.model.ocean_points

a_particle = wave_simulation.model.ParticleCollection[31, 2]

@show a_particle.on
a_particle.on = ~a_particle.on
@show a_particle.on
@show wave_simulation.model.ParticleCollection[31, 2].on # <- wrong 
wave_simulation.model.ParticleCollection[a_particle.position_ij[1], a_particle.position_ij[2]]= a_particle
@show wave_simulation.model.ParticleCollection[31, 2].on
