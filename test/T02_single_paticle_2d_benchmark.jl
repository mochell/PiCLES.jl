
# %%
using DifferentialEquations
using Plots
using Setfield


using PiCLES.ParticleSystems: particle_waves_v5 as PW

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleValues, InitParticleInstance
using PiCLES.ParticleMesh: TwoDGrid, TwoDGridNotes
using Oceananigans.Units

using BenchmarkTools
#using Revise
using Profile
# debugging:

# %%

plot_path_base = "plots/tests/T04_2D_single_particle/"
mkpath(plot_path_base)

function set_u_and_t!(integrator, u_new, t_new)
    integrator.u = u_new
    integrator.t = t_new
end


# %% Parameters
U10, V10 = +10.0, +10.0

# version 3
r_g0 = 0.85
# function to define constants for grouwth and dissipation

#
Const_Scg = PW.get_Scg_constants()

#u(x, y, t) = 0.01 - U10 * sin(t / (6 * 60 * 60 * 2π))
#v(x, y, t) = 0.01 - V10 * cos(t / (6 * 60 * 60 * 2π))

#u(x, y, t) = - U10 + 0.01 + x * 0 + y * 0 + t *0
#v(x, y, t) = + V10 + 0.01 + x * 0 + y * 0 + t *0

u(x, y, t) = (U10 * cos(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
v(x, y, t) = -(V10 * sin(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0

winds = (u=u, v=v)

Revise.retry()
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)
typeof(particle_system)

particle_system
# define V4 parameters absed on Const NamedTuple:
default_ODE_parameters = (
    r_g=r_g0,
    C_α=Const_Scg.C_alpha,
    C_φ=Const_ID.c_β,
    C_e=Const_ID.C_e,
    g=9.81,
)

# define simple callback
condition(u, t, integrator) = 0.9 * u[1] > log(17)
affect!(integrator) = terminate!(integrator)
cb = ContinuousCallback(condition, affect!)

# define standard initial conditions
DT = 4hours
WindSeamin = FetchRelations.get_initial_windsea(u(0, 0, 0), v(0, 0, 0), DT / 2)
#WindSeamin = FetchRelations.MinimalWindsea(u(0, 0, 0), v(0, 0, 0), 20minutes)

ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),
    #maximum energy threshold
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=300hours,
    timestep=DT,
    total_time=T = 6days,
    callbacks=cb,
    save_everystep=false,
    maxiters=1e4,
    adaptive=true,
    dt=10,#60*10, 
    dtmin=1,#60*5, 
    force_dtmin=true,
)

grid = TwoDGrid(3, 3, 3, 3)
ParticleState = ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0)
ParticleState2 = ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 1.0)


# initialize particle given the wind conditions:
#ParticleState = InitParticleValues(copy(particle_defaults), TwoDGridNotes(grid), winds, DT)



function time_step_local!(PI, DT)
    "take 1 step over DT"

    @info "proposed dt", get_proposed_dt(PI.ODEIntegrator) / 60

    step!(PI.ODEIntegrator, DT, true)

    #@info "u:", PI.ODEIntegrator.u
    #clock_time += DT
    last_t = PI.ODEIntegrator.t

    ## define here the particle state at time of resetting
    #ui = [log(exp(PI.ODEIntegrator.u[1]) * 0.5), PI.ODEIntegrator.u[2] / 2, PI.ODEIntegrator.u[3] / 2, 0.0, 0.0]
    #ui = [lne_local, cg_u_local, cg_v_local, 0.0, 0.0]
    ui = PI.ODEIntegrator.u

    #ui = [PI.ODEIntegrator.u[1], PI.ODEIntegrator.u[2], PI.ODEIntegrator.u[3], 0.0, 0.0]
    WindSeamin = FetchRelations.get_initial_windsea(u(0, 0, last_t), v(0, 0, last_t), DT / 2)
    ui = [log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0]

    set_u_and_t!(PI.ODEIntegrator, ui, last_t)
    # #set_u!(PI.ODEIntegrator, ui)
    #reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)
    #reinit!(PI3.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)

    # #set_t!(PI.ODEIntegrator, last_t )
    u_modified!(PI.ODEIntegrator, true)

    add_saveat!(PI.ODEIntegrator, PI.ODEIntegrator.t)
    savevalues!(PI.ODEIntegrator)

    return PI
end

# %%
@time @allocated PI = InitParticleInstance(particle_system, ParticleState, ODE_settings, (0, 0), false, true)
@time @allocated PI = InitParticleInstance(particle_system, ParticleState2, ODE_settings, (0, 0), false, true)
@time @allocated time_step_local!(PI, DT)

#@time @allocated time_step_local_replace!(PI, DT)

# %%
#using ProfileCanvas
@time @allocated PI = InitParticleInstance(particle_system, ParticleState, ODE_settings, (0, 0), false, true)

DT = 10minutes
DT = 3hours
@profview_allocs @time @allocated time_step_local!(PI, DT)

@profview_allocs for i in Base.Iterators.take(PI.ODEIntegrator, 30)
    #@info "t:", PI.ODEIntegrator.t
    #@info "u:", PI.ODEIntegrator.u
    #@info "i:", i
    time_step_local!(PI, DT)

end


# %% Profiling single time step:
using ProfileView
Revise.retry()
@time @allocated PI = InitParticleInstance(particle_system, ParticleState, ODE_settings, (0, 0), false, true)
@time @allocated time_step_local!(PI, DT)
Revise.retry()
DT = 10minutes
ProfileView.@profview for i in Base.Iterators.take(PI.ODEIntegrator, 10)
    #@info "t:", PI.ODEIntegrator.t
    #@info "u:", PI.ODEIntegrator.u
    #@info "i:", i
    @time @allocated time_step_local!(PI, DT)

end




# %

# ODEProblem(particle_system, z_initials, (0.0, ODE_settings.total_time), ODE_parameters)

# %%
