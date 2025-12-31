
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

# %%


save_path = "plots/tests/T04_2D_particle_on_off/"
mkpath(save_path)

##### basic parameters
# timestep
DT = 20minutes
# Characterstic wind velocities and std
U10, V10 = 10.0, 10.0

# Define basic ODE parameters
r_g0 = 0.85
Const_ID = PW4.get_I_D_constant()

Const_Scg = PW4.get_Scg_constants(C_alpha=-1.41, C_varphi=1.81e-5)


# define grid
grid = TwoDGrid(100e3, 21, 50e3, 11)
mesh = TwoDGridMesh(grid, skip=1);
gn = TwoDGridNotes(grid);

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
ODE_settings =  PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=WindSeamin["lne"],
    #maximum energy threshold
    log_energy_maximum=log(27),#log(17),  # correcsponds to Hs about 16 m
    # define minimum wind squared threshold
    wind_min_squared=2.0,
    saving_step=DT,
    timestep=DT,
    total_time=T = 6days,
    adaptive=true,
    dt=1e-3, #60*10, 
    dtmin=1e-4, #60*5, 
    force_dtmin=true,
    callbacks=nothing,
    save_everystep=false)


# %% half domain tests
Revise.retry()
#gridmesh = [(i, j) for i in [-10,10], j in  [0]]
#gridmesh = [(i, j) for i in [10], j in [0]]

U10 = -8
V10 = 0.0

@show U10, V10

x0 =50e3
Lx = (gn.Nx - 1) * gn.dx
# u_func(x, y, t) = IfElse.ifelse.(x .< x0, U10, U10 * (1 -x/Lx) ) + y * 0 + t * 0
# v_func(x, y, t) = IfElse.ifelse.(x .< x0, V10, V10 * (1 -x/Lx) ) + y * 0 + t * 0

u_func(x, y, t) = IfElse.ifelse.(x .< x0, x*0+ 0, U10 * (x - x0) / (Lx-x0)) + y * 0 + t * 0
v_func(x, y, t) = IfElse.ifelse.(x .< x0, x*0+ 0, V10 * (x - x0) / (Lx-x0)) + y * 0 + t * 0

# u_func(x, y, t) = IfElse.ifelse.(x .< x0, x *0+  0.0 , U10 ) + y * 0 + t * 0
# v_func(x, y, t) = IfElse.ifelse.(x .< x0, x *0 + 0.0 , V10 ) + y * 0 + t * 0
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

typeof(wave_model)


# %%
### build Simulation
Revise.retry()
wave_simulation = Simulation(wave_model, Δt=DT/2, stop_time=4hours)
initialize_simulation!(wave_simulation)

init_state_store!(wave_simulation, save_path)

length(wave_simulation.store.shape) 

#run!(wave_simulation, cash_store=true, debug=true)
run!(wave_simulation, store=true, cash_store=false, debug=false)

close_store!(wave_simulation)

#run!(wave_simulation, cash_store=true, debug=true)
# %

# plot(wave_simulation.model.State[:,:,2])
# wave_simulation.model.State[:, :, 2]
# # wave_simulation.model.State[:, :, 1]
# time_step!(wave_simulation.model, wave_simulation.Δt)

# wave_simulation.model.State[:, :, 2]

# (wave_simulation.model.State[:, :, 1] .>= wave_model.minimal_state[1])

# (wave_simulation.model.State[:, :, 2] .>= wave_model.minimal_state[2])

# wave_simulation.model.State[:, :, 2]
# wave_simulation.model.State[:, :, 3]


# %
# or, alternatively, make movie
plot_name="T02_2D_growing_U" * string(U10) * "_V" * string(V10)
N=80
axline=x0/1e3
fig, n = init_movie_2D_box_plot(wave_simulation; resolution=(1300, 800), name_string=plot_name, aspect=3, axline=axline)

#wave_simulation.stop_time += 1hour
#N = 36
#plot_name = "dummy"

record(fig, save_path * plot_name * ".gif", 1:N, framerate=10) do i
    @info "Plotting frame $i of $N..."
    @info wave_simulation.model.clock
    movie_time_step!(wave_simulation.model, wave_simulation.Δt)
    n[] = 1
end

# %%
#wave_simulation.model.State[:, :, 2]
#test particle equations
#using BenchmarkTools
using Profile

Revise.retry()
tt = [(i, j) for i in [-1,0, 1], j in [-1, 0, 1]]
for (i,j) in tt
    #@show PW4.speed(i,j)
    #@time PW4.α_func(i,j)
    @show i,j, PW4.sin2_a_min_b( 1, 1 , i,j)
end

PW4.sin2_a_min_b( 1, 1 , 0.01,0.01 )
PW4.sin2_a_min_b( 0,0.1  , 0,0.1 )
PW4.sin2_a_min_b( 0,0  , 1,1 )
PW4.sin2_a_min_b( 1,0  , 0,1 )
PW4.sin2_a_min_b( 0,1  , 0,1 )

PW4.sin2_a_min_b( 0,1  , 0,0 )


PW4.αₚ(0,0,0,0)
