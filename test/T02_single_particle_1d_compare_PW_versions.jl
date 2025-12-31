import Plots


using PiCLES.ParticleSystems: particle_waves_v3beta as PW3
using PiCLES.ParticleSystems: particle_waves_v4 as PW4
using PiCLES.ParticleSystems: particle_waves_v5 as PW5

import PiCLES: FetchRelations

using PiCLES.Operators.core_1D: ParticleDefaults, InitParticleValues, InitParticleInstance
using PiCLES.ParticleMesh: OneDGrid, OneDGridNotes

using ModelingToolkit, DifferentialEquations

using PiCLES.Utils.ParticleTools 
using Plots

using Oceananigans.Units

# % Parameters
@register_symbolic u(x, t)
@register_symbolic u_x(x, t)


## only popstiive valules work in paricle waves v3!
U10 = 3.5
r_g0 = 0.85
c_β = 4e-2
C_e0 = (2.35 / r_g0) * 2e-3 * c_β
γ = 0.88

dt_ODE_save = Float64(60 * 2) # 3 min
DT = Float64(60 * 30) * 2 #* 24 # seconds
T = 24 * 2 * 60 * 60 # seconds


## u must be always a function of x and t !!!
u(x, t) = x .* 0 + t * 0 + U10
#u_x(x, t) = x .* 0 + t * 0

Revise.retry()
particle_equations = PW3.particle_equations(u, u, γ=γ)
@named particle_system = ODESystem(particle_equations)

particle_equations4 = PW4.particle_equations(u, γ=γ)
@named particle_system4 = ODESystem(particle_equations4)

particle_system5 = PW5.particle_equations(u, γ=γ)


# define variables based on particle equation
t, x, c̄_x, lne, r_g, C_α, g, C_e = PW3.init_vars_1D()

# %% define storing stucture and populate inital conditions

default_ODE_parameters = Dict(
    r_g => 0.85,
    C_α => -1.41,
    g => 9.81,
    C_e => C_e0,
)


condition(u, t, integrator) = 0.9 * u[1] > log(17)
affect!(integrator) = terminate!(integrator)
cb = ContinuousCallback(condition, affect!)


WindSeamin = FetchRelations.get_initial_windsea(u(0, 0), 10minutes)
#WindSeamin = FetchRelations.MinimalWindsea(u(0, 0, 0), 20minutes)


ODE_settings =  PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),#log(FetchRelations.Eⱼ(0.1, DT)),
    #maximum energy threshold
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=dt_ODE_save,
    timestep=DT,
    total_time=T,
    callbacks=cb,
    save_everystep=false, 
    dt=1e-3, #60*10, 
    dtmin=1e-9, #60*5, 
)


grid1d = OneDGrid(1, 3, 3)
particle_defaults = ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar"], 0.0)
#particle_defaults = ParticleDefaults(ODE_settings.log_energy_minimum, cg_local, 1.51)

xx = OneDGridNotes(grid1d).x[2]
# initialize particle given the wind conditions:
ParticleState, particle_on = InitParticleValues(copy(particle_defaults), xx, u(xx, 0), DT)


Revise.retry()

typeof(particle_system)
typeof(particle_system4)
typeof(particle_system5)

PI3 = InitParticleInstance(particle_system, ParticleState, ODE_settings, 0, false, true)
PI4 = InitParticleInstance(particle_system4, ParticleState, ODE_settings, 0, false, true)

# %%
Revise.retry()
particle_system5 = PW5.particle_equations(u, γ=γ)

# ordered_ParticleState = [ParticleState[lne], ParticleState[c̄_x], ParticleState[x]]
# # manual testing 
# named_default_ODE_parameters = NamedTuple{Tuple(Symbol.(keys(default_ODE_parameters)))}(values(default_ODE_parameters)) 
# ps1  = particle_system5([0,0,0], ordered_ParticleState, named_default_ODE_parameters, 0)

# problem = ODEProblem(particle_system5, ordered_ParticleState, (0.0, ODE_settings.total_time), named_default_ODE_parameters)
PI5 = InitParticleInstance(particle_system5, ParticleState, ODE_settings, 0, false, true)

# %%

function set_u_and_t!(integrator, u_new, t_new)
    integrator.u = u_new
    integrator.t = t_new
end


clock_time = 0
NDT = 6
for i in Base.Iterators.take(PI3.ODEIntegrator, NDT)
    @info "x:", PI3.ODEIntegrator.u[3], "t:", PI3.ODEIntegrator.t, clock_time
    step!(PI3.ODEIntegrator, DT, true)

    last_t = PI3.ODEIntegrator.t
    clock_time = PI3.ODEIntegrator.t
    ui = [log(exp(PI3.ODEIntegrator.u[1]) * 1), PI3.ODEIntegrator.u[2], PI3.ODEIntegrator.u[3] * 0]
    #ui = [log(WindSeamin["E"]), WindSeamin["cg_bar"], 0.0]
    #ui = [lne_local, cg_local, PI.ODEIntegrator.u[3]]

    #set_u!(PI.ODEIntegrator, ui)
    #reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)
    set_u_and_t!(PI3.ODEIntegrator, ui, clock_time)

    #set_t!(PI.ODEIntegrator, last_t )
    u_modified!(PI3.ODEIntegrator, true)

end

#### this is the one that works #######
clock_time = 0
for i in Base.Iterators.take(PI4.ODEIntegrator, NDT)
    @info "x:", PI4.ODEIntegrator.u[3], "t:", PI4.ODEIntegrator.t, clock_time

    step!(PI4.ODEIntegrator, DT, true)
    clock_time += DT
    clock_time = PI4.ODEIntegrator.t
    ui = [log(exp(PI4.ODEIntegrator.u[1]) * 1), PI4.ODEIntegrator.u[2], PI4.ODEIntegrator.u[3]*0]
    #ui = [log(WindSeamin["E"]), WindSeamin["cg_bar"], 0.0]
    #ui = [lne_local, cg_local, PI4.ODEIntegrator.u[3]]

    #set_u!(PI4.ODEIntegrator, ui)
    #reinit!(PI4.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)

    set_u_and_t!(PI4.ODEIntegrator, ui, clock_time)

    #set_t!(PI4.ODEIntegrator, clock_time)
    u_modified!(PI4.ODEIntegrator, true)

end

# PI5
clock_time = 0
for i in Base.Iterators.take(PI5.ODEIntegrator, NDT)
    @info "x:", PI5.ODEIntegrator.u[3], "t:", PI5.ODEIntegrator.t, clock_time

    step!(PI5.ODEIntegrator, DT, true)
    clock_time += DT
    clock_time = PI5.ODEIntegrator.t
    ui = [log(exp(PI5.ODEIntegrator.u[1]) * 1), PI5.ODEIntegrator.u[2], PI5.ODEIntegrator.u[3] * 0]
    #ui = [log(WindSeamin["E"]), WindSeamin["cg_bar"], 0.0]
    #ui = [lne_local, cg_local, PI4.ODEIntegrator.u[3]]

    #set_u!(PI4.ODEIntegrator, ui)
    #reinit!(PI4.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)

    set_u_and_t!(PI5.ODEIntegrator, ui, clock_time)

    #set_t!(PI4.ODEIntegrator, clock_time)
    u_modified!(PI5.ODEIntegrator, true)

end


# PI.ODEIntegrator.u
# affect!(PI.ODEIntegrator)
# %


#fig = figure(resolution=(800, 900))
PID = ParticleTools.ParticleToDataframe(PI3)
PID4 = ParticleTools.ParticleToDataframe(PI4)
PID5 = ParticleTools.ParticleToDataframe(PI5)

gr(display_type=:inline)
# plit each row in PID and a figure
p1 = plot(PID[1:3:end, 4] / 1e3, exp.(PID[1:3:end, 2]), marker=1, title="e", xlabel="x (km)", ylabel="e", label="PW3")#, ylim=(0.0025, maximum(exp.(PID[:, 2]))))
plot!(p1, PID4[1:3:end, 4] / 1e3, exp.(PID4[1:3:end, 2]), color=:red, marker=2, label="PW4") #|> display
plot!(p1, PID5[1:3:end, 4] / 1e3, exp.(PID5[1:3:end, 2]), color=:green, marker=2, label="PW5") #|> display

p2 = plot(PID[1:3:end, 4] / 1e3, PID[1:3:end, 3], marker=1, title="cg", ylabel="cg", xlabel="x (km)", label="PW3")#, ylim=(1.5, maximum(PID[1:3:end, 3]))) #|> display
plot!(p2, PID4[1:3:end, 4] / 1e3, PID4[1:3:end, 3], color=:red, marker=2, label="PW4") #|> display
plot!(p2, PID5[1:3:end, 4] / 1e3, PID5[1:3:end, 3], color=:green, marker=2, label="PW5") #|> display

p3 = plot(PID[1:3:end, 1] / 60 / 60, PID[1:3:end, 4] / 1e3, marker=1, title="Hofmoeller", xlabel="time (hours)", ylabel="x (km)", label="PW3", bottom_margin=5*Plots.mm )
plot!(p3, PID4[1:3:end, 1] / 60 / 60, PID4[1:3:end, 4] / 1e3, color=:red, marker=3, label="PW4") #|> display
plot!(p3, PID5[1:3:end, 1] / 60 / 60, PID5[1:3:end, 4] / 1e3, color=:green, marker=3, label="PW5") #|> display

plot(p1, p2, p3, layout=(3, 1), legend=true, size=(600, 1200), left_margin=10*Plots.mm )

# save figure
plot_path_base = "plots/PW4/single_particle/"
savefig("PW3_vs_PW4_vs_PW5.png")
