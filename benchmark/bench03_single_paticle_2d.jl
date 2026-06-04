#using OrdinaryDiffEq
using Plots
using Setfield

using PiCLES.ParticleSystems: particle_waves_v6 as PW
using PiCLES.Utils: Init_Standard

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: InitParticleInstance
using Oceananigans.Units

using BenchmarkTools
#using Revise
using Profile

plot_path_base = "plots/tests/T04_2D_single_particle/"
mkpath(plot_path_base)

# %%
u(x::Number, y::Number, t::Number) = (10.0 * cos(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
v(x::Number, y::Number, t::Number) = -(10.0 * sin(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0

DT = 4hours
ParticleState0, default_ODE_parameters, WindSeamin, Const_ID = Init_Standard(u(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), DT)
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)

ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),
    #maximum energy threshold
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


PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (0, 0), false, true)

WindSeamin0 = FetchRelations.get_initial_windsea(u(0.0, 0.0, 0), v(0.0, 0.0, 0), DT / 2)

ui = FetchRelations.get_initial_windsea(u(0.0, 0.0, 0), v(0.0, 0.0, 0), DT / 2, particle_state=true)


function set_u_and_t!(integrator, u_new, t_new)
    integrator.u = u_new
    integrator.t = t_new
end

function time_step_local!(PI, DT)
    "take 1 step over DT"

    #@info "proposed dt", get_proposed_dt(PI.ODEIntegrator) / 60
    step!(PI.ODEIntegrator, DT, true)

    #@info "u:", PI.ODEIntegrator.u
    #clock_time += DT
    last_t = PI.ODEIntegrator.t

    ## define here the particle state at time of resetting
    #ui = [log(exp(PI.ODEIntegrator.u[1]) * 0.5), PI.ODEIntegrator.u[2] / 2, PI.ODEIntegrator.u[3] / 2, 0.0, 0.0]
    #ui = [lne_local, cg_u_local, cg_v_local, 0.0, 0.0]
    ui = PI.ODEIntegrator.u

    #ui = [PI.ODEIntegrator.u[1], PI.ODEIntegrator.u[2], PI.ODEIntegrator.u[3], 0.0, 0.0]
    #WindSeamin = FetchRelations.get_initial_windsea(u(0.0, 0.0, last_t), v(0.0, 0.0, last_t), DT / 2)
    #WindSeamin = copy(WindSeamin0)
    #ui = [log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0]

    #ui = copy(ui)#FetchRelations.get_initial_windsea(u(0.0, 0.0, 0), v(0.0, 0.0, 0), DT / 2, particle_state=true)
    ui = FetchRelations.get_initial_windsea(u(0.0, 0.0, 0), v(0.0, 0.0, 0), DT / 2, particle_state=true)

    ui[4:5] = [0.0, 0.0]
    set_u_and_t!(PI.ODEIntegrator, ui, last_t)
    # #set_u!(PI.ODEIntegrator, ui)
    #reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)
    #reinit!(PI3.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)

    # #set_t!(PI.ODEIntegrator, last_t )
    u_modified!(PI.ODEIntegrator, true)

    # add_saveat!(PI.ODEIntegrator, PI.ODEIntegrator.t)
    # savevalues!(PI.ODEIntegrator)

    return PI
end
# %%
for i in Base.Iterators.take(PI.ODEIntegrator, 20)
    time_step_local!(PI, DT)
end


# %%

Revise.retry()
@benchmark PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (0, 0), false, true)

@time @allocated PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (0, 0), false, true)
# 1st :  0.007556 seconds (212 allocations: 236.797 KiB)
# 2nd :   0.000363 seconds (206 allocations: 13.562 KiB)
# 3rd :   0.000240 seconds (153 allocations: 8.766 KiB)

@time @allocated time_step_local!(PI, DT)
# 1st  0.001300 seconds (861 allocations: 82.266 KiB)
# 2nd  0.000277 seconds (90 allocations: 8.266 KiB)
# 3rd   0.000348 seconds (4 allocations: 208 bytes)

#@time @allocated time_step_local_replace!(PI, DT)

# %%
#using ProfileCanvas
PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (0, 0), false, true)

DT = 10minutes
DT = 10hours
@profview_allocs @time @allocated time_step_local!(PI, DT)
# (4 allocations: 208 bytes) # regardless of integration time! i.e. 4 acclocation per integration timestep

@profview_allocs for i in Base.Iterators.take(PI.ODEIntegrator, 300)
    time_step_local!(PI, DT)
end

# %% Profiling single time step:
using ProfileView
Revise.retry()
@time @allocated PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (0, 0), false, true)
@time @allocated time_step_local!(PI, DT)
Revise.retry()
DT = 10minutes
ProfileView.@profview for i in Base.Iterators.take(PI.ODEIntegrator, 20)
    #@info "t:", PI.ODEIntegrator.t
    #@info "u:", PI.ODEIntegrator.u
    #@info "i:", i
    @time @allocated time_step_local!(PI, DT)

end


# %%
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100#10000.0
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

function benchmark_advance_cycle2(ODE_settings, DT, NT)

    #PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (0, 0), false, true)
    # warmup
    time_step_local!(PI, 1minutes)

    trial = @benchmark begin
        #@info $DT, $NT, ODE_settings.solver
        PI = InitParticleInstance(particle_system, copy(ParticleState0) , $ODE_settings, (0, 0), false, true)
        # for i in Base.Iterators.take($PI.ODEIntegrator, $NT)
        #     time_step_local!($PI, $DT)
        # end
        for i in 1:$NT
            time_step_local!(PI, $DT)
        end
    end
    #
#    end samples = 3

   return trial
end


ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),
    #maximum energy threshold
    solver=DP5(), #AutoTsit5(Rosenbrock23()),
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=6days,
    timestep=DT,
    total_time=T = 6days,
    save_everystep=false,#false,
    maxiters=1e4,
    adaptive=true,
    dt=10,#60*10, 
    dtmin=1,#60*5, 
    force_dtmin=true,);

#PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (0, 0), false, true)
A2 = benchmark_advance_cycle2(ODE_settings, 10minutes, 10)
# 2nd: Memory estimate: 18.75 KiB, allocs estimate: 264. 528.188 μs 
# 3rd:  Memory estimate: 10.94 KiB, allocs estimate: 194. 8.938 ms 
# 4th:   Memory estimate: 10.94 KiB, allocs estimate: 194. 734.417 μs 

A1 = benchmark_advance_cycle2(ODE_settings, 10minutes, 20)

A1 = benchmark_advance_cycle2(ODE_settings, 20minutes, 10)

A2 = benchmark_advance_cycle2(ODE_settings, 10minutes, 10)

PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (0, 0), false, true)
A1 = benchmark_advance_cycle2(ODE_settings, 5minutes, 10)

A1 = benchmark_advance_cycle2(ODE_settings, 10minutes, 6)
#  Memory estimate: 10.12 KiB, allocs estimate: 178. 4.212 ms
# %%

ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),
    #maximum energy threshold
    solver=AutoTsit5(Rosenbrock23()),
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=6days,
    timestep=DT,
    total_time=T = 6days,
    save_everystep=false,#false,
    maxiters=1e4,
    adaptive=true,
    dt=10,#60*10, 
    dtmin=1,#60*5, 
    force_dtmin=true,);

A1 = benchmark_advance_cycle2(ODE_settings, 10minutes, 6)
# Memory estimate: 48.86 KiB, allocs estimate: 585.  1.680 ms 
# Memory estimate: 57.67 KiB, allocs estimate: 687. 2.190 ms

# %%
# %%