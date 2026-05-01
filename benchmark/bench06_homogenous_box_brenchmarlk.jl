using Pkg

Pkg.activate("PiCLES/")

# using Base.Threads
# @info "Num. of threads", nthreads()

# import Plots as plt
using Setfield
using PiCLES.ParticleSystems: particle_waves_v5 as PW

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleInstance, GetGroupVelocity
using PiCLES.Simulations
using PiCLES.Operators.TimeSteppers: time_step!, movie_time_step!

using PiCLES.ParticleMesh: TwoDGrid, TwoDGridNotes, TwoDGridMesh
using PiCLES.Models.WaveGrowthModels2D

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units
import Oceananigans.Utils: prettytime

using PiCLES.Architectures
#using GLMakie
#using PiCLES.Plotting.movie: init_movie_2D_box_plot

#using ProfileView
using BenchmarkTools
#using Revise

using Profile
# debugging:
using ProfileView

using DataFrames

#using OrdinaryDiffEq

@info "precompiled!"
# %%
save_path = "plots/examples/homogenous_box/"
mkpath(save_path)

# % Parameters
U10, V10 = 10.0, 10.0

DT = 30minutes
# version 3
r_g0 = 0.85

# function to define constants 
Const_ID = PW.get_I_D_constant()
@set Const_ID.γ = 0.88
Const_Scg = PW.get_Scg_constants(C_alpha=-1.41, C_varphi=1.81e-5)


u(x::Number, y::Number, t::Number) = 10.0 #* sin(t / (6 * 60 * 60 * 2π)) * sin(x / 50e3) * sin(y / 50e3)
v(x::Number, y::Number, t::Number) = 10.0 #* cos(t / (6 * 60 * 60 * 2π)) * sin(x / 50e3) * sin(y / 50e3)

winds = (u=u, v=v)

grid = TwoDGrid(100e3, 51, 100e3, 51)
#grid = TwoDGrid(20e3, 21, 20e3, 21)
mesh = TwoDGridMesh(grid, skip=1)
gn = TwoDGridNotes(grid)
#Revise.retry()


# define variables based on particle equation
particle_system = PW.particle_equations(u, v, γ=0.88, q=Const_ID.q, input=true, dissipation=true)
#particle_equations = PW3.particle_equations_vec5(u, v, u, v, γ=0.88, q=Const_ID.q);

# define V4 parameters absed on Const NamedTuple:
default_ODE_parameters = (r_g=r_g0, C_α=Const_Scg.C_alpha,
    C_φ=Const_ID.c_β, C_e=Const_ID.C_e, g=9.81)


# define setting and standard initial conditions
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT;)
#WindSeamin = FetchRelations.MinimalWindsea(u(0, 0, 0), v(0, 0, 0), DT / 2)
#WindSeamin = FetchRelations.get_initial_windsea(u(0, 0, 0), v(0, 0, 0), DT/5)
lne_local = log(WindSeamin["E"])
cg_u_local = WindSeamin["cg_bar_x"]
cg_v_local = WindSeamin["cg_bar_y"]

ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),#log(FetchRelations.Eⱼ(0.1, DT)),
    #maximum energy threshold
    solver=DP5(),
    log_energy_maximum=log(27),#log(17),  # correcsponds to Hs about 16 m
    saving_step=6days,
    timestep=DT,
    total_time=T = 6days,
    adaptive=true,
    dt=10, #60*10, 
    dtmin=1, #60*5, 
    force_dtmin=true,
    callbacks=nothing,
    save_everystep=false)


default_particle = ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0)

Revise.retry()
wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",  # default_ODE_parameters
    periodic_boundary=false,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true);

#wave_model.State

Revise.retry()

#global wave_model.ParticleCollection
### build Simulation
#wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=4hours)#1hours)
wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=1hour);#1hours)
initialize_simulation!(wave_simulation)

# # %% single time steps
# import PiCLES.Operators.TimeSteppers: time_step!
# for i in 1:3
#     wave_simulation.model.State .= 0.0
#     @time time_step!(wave_simulation.model, wave_simulation.Δt, debug=false)
#     #time_step!(wave_simulation.model, wave_simulation.Δt, debug=false)
#     istate = wave_simulation.model.State
#     p1 = plt.heatmap(gn.x / 1e3, gn.y / 1e3, istate[:, :, 1])
#     display(p1)
#     sleep(0.5)
# end
# # %%

Revise.retry()
run!(wave_simulation, cash_store=false, debug=false);

# %%
reset_simulation!(wave_simulation)
wave_simulation.stop_time = 20minutes
#ProfileView.@profview run!(wave_simulation, cash_store=false, debug=false);

@time @allocated run!(wave_simulation, cash_store=false, debug=false); 
# org: 01/13/2024: 5.926792 seconds (54.35 M allocations: 1.434 GiB, 12.16% gc time)
# 1st: 01/15/2024: 0.178086 seconds (573.16 k allocations: 37.163 MiB)
# 2nd: 01/18/2024:  0.154943 seconds (413.15 k allocations: 23.212 MiB)
# 3rd: 01/21/2024:0.829573 seconds (311.96 k allocations: 14.554 MiB)

wave_simulation.stop_time = 40minutes
@time @allocated run!(wave_simulation, cash_store=false, debug=false); 
# org: 01/13/2024: 2.322934 seconds (18.59 M allocations: 566.626 MiB, 10.23% gc time)
# 1st: 0.163212 seconds (368.23 k allocations: 23.530 MiB, 40.60% gc time)
# 2nd: 0.106983 seconds (275.43 k allocations: 15.474 MiB)
# 3rd: 0.561313 seconds (207.97 k allocations: 9.703 MiB)

wave_simulation.stop_time = 60minutes
@time @allocated run!(wave_simulation, cash_store=false, debug=false); 
# org: 01/132024: 2.208400 seconds (18.73 M allocations: 570.800 MiB, 10.53% gc time)
# 1st:   0.088965 seconds (369.62 k allocations: 23.595 MiB)
# 2nd:  0.100938 seconds (275.44 k allocations: 15.475 MiB)
# 3rd:  0.520794 seconds (207.98 k allocations: 9.704 MiB)

wave_simulation.stop_time = 80minutes
@time @allocated run!(wave_simulation, cash_store=false, debug=false); 
# 2.703650 seconds (18.26 M allocations: 558.912 MiB, 11.22% gc time)
#   0.098254 seconds (375.63 k allocations: 25.917 MiB)

# %% ----- speed profile
@time @allocated wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=1hour);#1hours)
@time @allocated initialize_simulation!(wave_simulation)
# org: 0.391615 seconds (858.79 k allocations: 46.285 MiB)
# 1st:   0.064669 seconds (705.31 k allocations: 42.911 MiB)
# 2nd  0.052143 seconds (432.21 k allocations: 27.949 MiB)


wave_simulation.stop_time = 20minutes
@time @allocated run!(wave_simulation, cash_store=false, debug=false); # 5.772268 seconds (54.35 M allocations: 1.434 GiB, 10.11% gc time)


wave_simulation.stop_time = 40minutes
ProfileView.@profview run!(wave_simulation, cash_store=false, debug=false);

# run!(wave_simulation, cash_store=false, debug=false);
# %%
wave_simulation.stop_time = 60minutes
Profile.clear()
@profile run!(wave_simulation, cash_store=false, debug=false);
ProfileView.view(expand_tasks= true, expand_threads=true)
Profile.print(mincount=10, groupby=:thread)
# a few particle_system equations are slow because they require a lot of memory allocation? and they arre called when SciMLBase takes derivates! there are step funcions.. 



# %% ---- allocation profile
@time @allocated wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=1hour);#1hours)
@profview_allocs @time @allocated initialize_simulation!(wave_simulation)

#@profview_allocs
wave_simulation.stop_time = 40minutes
@time @allocated run!(wave_simulation, cash_store=false, debug=false);

# %%
wave_simulation.stop_time = 60minutes
@profview_allocs @time @allocated run!(wave_simulation, cash_store=false, debug=false)