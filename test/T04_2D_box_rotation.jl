ENV["JULIA_INCREMENTAL_COMPILE"]=true

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
using PiCLES.Plotting.movie: init_movie_2D_box_plot, init_movie_2D_box_plot_small

# debugging:
#using ProfileView

# %%
save_path = "plots/tests/T04_box_2d/"
mkpath(save_path)

# % Parameters
U10,V10           = 10.0, 10.0
dt_ODE_save       = 30minutes
DT                = 20minutes
# version 3


# function to define constants 
# ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)
# Const_ID

ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)

# u(x, y, t) = 0.01 + U10 * sin(t / (6 * 60 * 60 * 2π)) * sin(x / 50e3) * sin(y / 50e3)
# v(x, y, t) = 0.01 - V10 * cos(t / (6*60*60 * 2π) ) * sin(x / 50e3) * sin(y / 50e3)
u_std= 20e3 *1
v_std= 20e3 *1
center = 25e3
u_func(x, y, t) = U10 * exp(-(x - center)^2 / u_std^2) * exp(-(y - center)^2 / v_std^2) * cos(t * 1.2 / (1 * 60 * 60 * 2π))
v_func(x, y, t) = V10 * exp(-(x - center)^2 / u_std^2) * exp(-(y - center)^2 / v_std^2) * sin(t * 1.2 / (1 * 60 * 60 * 2π))

# u_func(x, y, t) = 0.1 + IfElse.ifelse.( sin(t * 3 / (1 * 60 * 60 * 2π)) > 0 , 
#                     sin(t * 3 / (1 * 60 * 60 * 2π)) *U10 * exp(-(x - 5e3)^2 / u_std^2) * exp(-(y - 5e3)^2 / v_std^2),
#                                         0.0) 
# v_func(x, y, t) = 0.1 + IfElse.ifelse.(sin(t * 3 / (1 * 60 * 60 * 2π)) > 0,
#                     sin(t * 3 / (1 * 60 * 60 * 2π)) * U10 * exp(-(x - 5e3)^2 / u_std^2) * exp(-(y - 5e3)^2 / v_std^2),
#                             0.0)

#t, x, y, c̄_x, c̄_y, lne, Δn, Δφ_p, r_g, C_α, C_φ, g, C_e = vars = PW.init_vars();

# u_func(x, y, t) = IfElse.ifelse.(x .< 5e3, U10, 0.2) + y * 0 + t * 0
# v_func(x, y, t) = (IfElse.ifelse.(x .< 5e3, V10, 0.2) + y * 0) .* cos(t * 3 / (1 * 60 * 60 * 2π))

# this shuold hopefully work
# u(x, y, t) = x * 0 + y * 0 + t * 0/ DT + 5.0
# v(x, y, t) = x * 0 + y * 0 + t * 0/ DT + 10.0

u(x, y, t) = u_func(x, y, t)
v(x, y, t) = v_func(x, y, t)
winds = (u=u, v=v)


grid = TwoDGrid(100e3, 81, 100e3, 81)
mesh = TwoDGridMesh(grid, skip=1);
gn = TwoDGridNotes(grid);

Revise.retry()
# define variables based on particle equation

#ProfileView.@profview 
#ProfileView.@profview 
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q, input=true, dissipation=true);
#particle_equations = PW3.particle_equations_vec5(u, v, u, v, γ=Const_ID.γ, q=Const_ID.q);


# define setting and standard initial conditions
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT )
#WindSeamin = FetchRelations.MinimalWindsea(u(0, 0, 0), v(0, 0, 0), DT / 2)
#WindSeamin = FetchRelations.get_initial_windsea(u(0, 0, 0), v(0, 0, 0), DT/5)
lne_local = log(WindSeamin["E"])
default_particle = ParticleDefaults(lne_local, WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0)

ODE_settings    = PW.ODESettings(
    Parameters=ODEpars,
    # define mininum energy threshold
    log_energy_minimum=lne_local,#log(FetchRelations.Eⱼ(0.1, DT)),
    #maximum energy threshold
    log_energy_maximum=log(27),#log(17),  # correcsponds to Hs about 16 m
    saving_step=dt_ODE_save,
    timestep=DT,
    total_time=T=12days,
    adaptive=true,
    dt=1e-3, #60*10, 
    dtmin=1e-4, #60*5, 
    force_dtmin=true,
    callbacks=nothing,
    save_everystep=false)



# %%
Revise.retry()
wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",  # default_ODE_parameters
    periodic_boundary=false,
    boundary_type="same",
    # minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT), #
    # minimal_state=FetchRelations.MinimalState(2, 2, DT) * 1,
    movie=true)


wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=24hours)#1hours)
initialize_simulation!(wave_simulation)

#reset_simulation!(wave_simulation, particle_initials=copy(wave_model.ODEdefaults))

Revise.retry()
#using PiCLES.Plotting.movie: init_movie_2D_box_plot
fig, n = init_movie_2D_box_plot_small(wave_simulation, name_string="Rotating Stationary Winds")

#wave_simulation.stop_time += 1hour
N =  60 *8
record(fig, save_path*"T02_2D_totating_winds_nonper_small.gif", 1:N, framerate=10) do i
    @info "Plotting frame $i of $N..."
    #@time for _ = 1:10
    #run!(wave_simulation, store=false)
    @info wave_simulation.model.clock
    movie_time_step!(wave_simulation.model, wave_simulation.Δt)
    #end
    n[] = 1
end
