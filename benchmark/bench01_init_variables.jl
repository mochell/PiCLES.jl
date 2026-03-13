ENV["JULIA_INCREMENTAL_COMPILE"] = false
using Pkg
Pkg.activate("PiCLES/")

# #using OrdinaryDiffEq
# #using OrdinaryDiffEq: OrdinaryDiffEqCore.ODEIntegrator


using Plots
using Setfield

using PiCLES.ParticleSystems: particle_waves_v5 as PW
using PiCLES.Utils: Init_Standard

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: InitParticleInstance
using Oceananigans.Units

using BenchmarkTools
#using Revise
using Profile

plot_path_base = "plots/benchmark/"
mkpath(plot_path_base)

"""
This script is used to benchmark the functions called for (re)-initializing a particle instance.

"""

# %%
u(x::Number, y::Number, t::Number) = 10.0
v(x::Number, y::Number, t::Number) = 10.0

DT = 4hours
ParticleState0, default_ODE_parameters, WindSeamin, Const_ID = Init_Standard(u(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), DT)
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)

ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),
    #maximum energy threshold
    #solver=DP5(), #AutoTsit5(Rosenbrock23()),
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=2minutes,
    timestep=DT,
    total_time=T = 6days,
    save_everystep=false,
    maxiters=1e4,
    adaptive=true,
    dt=10,#60*10, 
    dtmin=1,#60*5, 
    force_dtmin=true,)


# %% 1.) InitParticleInstance
#  -- there is not much to do. Most of the allocation is in initializing the ODEIntegrator 
ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),
    #maximum energy threshold
    solver=AutoTsit5(Rosenbrock23()),
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=2minutes, timestep=DT, total_time=T = 6days, save_everystep=false, maxiters=1e4,
    adaptive=true, dt=10, dtmin=1, force_dtmin=true,)

#ODE_settings.solver = AutoTsit5(Rosenbrock23())
@time @allocated PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (0, 0), false, true)

@profview_allocs for i in 1:1000 # Base.Iterators.take(PI.ODEIntegrator, 30)
    PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (0, 0), false, true)
end

# %% DP5()
ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),
    #maximum energy threshold
    solver=DP5(),
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=2minutes, timestep=DT, total_time=T = 6days, save_everystep=false, maxiters=1e4,
    adaptive=true, dt=10, dtmin=1, force_dtmin=true,)

#ODE_settings.solver = AutoTsit5(Rosenbrock23())
@time @allocated PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (0, 0), false, true)
# 1st:   0.000216 seconds (159 allocations: 232.000 KiB) 

@profview_allocs for i in 1:10000 # Base.Iterators.take(PI.ODEIntegrator, 30)
    PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (0, 0), false, true)
end



# %% ------- 2.) windsea 

Revise.retry()

@time @allocated WindSeamin0 = FetchRelations.get_initial_windsea(u(0.0, 0.0, 0), v(0.0, 0.0, 0), DT / 2)
#org:   0.000339 seconds (60 allocations: 3.094 KiB)
#1st:   0.000102 seconds (8 allocations: 1.938 KiB)

@time @allocated ui = FetchRelations.get_initial_windsea(u(0.0, 0.0, 0), v(0.0, 0.0, 0), DT / 2, particle_state=true)
# 2nd: 0.000014 seconds (2 allocations: 112 bytes)


@profview_allocs for i in 1:10000 # Base.Iterators.take(PI.ODEIntegrator, 30)
    #WindSeamin0 = FetchRelations.get_initial_windsea(u(0.0, 0.0, 0), v(0.0, 0.0, 0), DT / 2)
    WindSeamin0 = FetchRelations.get_initial_windsea(u(0.0, 0.0, 0), v(0.0, 0.0, 0), DT / 2, particle_state=true)
end

@benchmark WindSeamin0 = FetchRelations.get_initial_windsea(u(0.0, 0.0, 0), v(0.0, 0.0, 0), DT / 2)
# org: Memory estimate: 3.09 KiB, allocs estimate: 60. 2.843 μs 
# 1st: Memory estimate: 1.94 KiB, allocs estimate: 8. 488 ns

@benchmark WindSeamin0 = FetchRelations.get_initial_windsea(u(0.0, 0.0, 0), v(0.0, 0.0, 0), DT / 2, particle_state=true)
# 2nd: Memory estimate: 112 bytes, allocs estimate: 2. 200 ns

# %% -------  3.) set_u_and_t!(PI.ODEIntegrator, ui, last_t)  & u_modified!(PI.ODEIntegrator, true)

function set_u_and_t!(integrator, u_new, t_new)
    integrator.u = u_new
    integrator.t = t_new
end

last_t = 600.0

@benchmark for i in 1:2 # Base.Iterators.take(PI.ODEIntegrator, 30)
    set_u_and_t!(PI.ODEIntegrator, ui, last_t)
    u_modified!(PI.ODEIntegrator, true)
end

# Memory estimate: 0 bytes, allocs estimate: 0.
