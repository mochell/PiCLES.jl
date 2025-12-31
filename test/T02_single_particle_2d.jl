using Pkg
Pkg.activate("PiCLES/")

using DifferentialEquations
using Plots
using Setfield
using IfElse

using PiCLES.ParticleSystems: particle_waves_v5 as PW
using PiCLES.Utils: Init_Standard

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: InitParticleInstance
using Oceananigans.Units

plot_path_base = "plots/tests/T04_2D_single_particle/"
mkpath(plot_path_base)

# %%
Revise.retry()
u(x::Number, y::Number, t::Number) = (10.0 * cos(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
v(x::Number, y::Number, t::Number) = -(10.0 * sin(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0

DT = 4hours
ParticleState, default_ODE_parameters, WindSeamin, Const_ID = Init_Standard(u(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), DT)
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)

# define simple callback
condition(u, t, integrator) = 0.9 * u[1] > log(17)
affect!(integrator) = terminate!(integrator)
cb = ContinuousCallback(condition, affect!)

ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),
    #maximum energy threshold
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=2minutes,
    timestep=DT,
    total_time=T = 6days,
    callbacks=cb,
    save_everystep=false,
    maxiters=1e4,
    adaptive=true,
    dt=10,#60*10, 
    dtmin=1,#60*5, 
    force_dtmin=true,)

PI = InitParticleInstance(particle_system, ParticleState, ODE_settings, (0, 0), (1, 2), false, true)
PI.ODEIntegrator
PI.ODEIntegrator.p

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
    WindSeamin = FetchRelations.get_initial_windsea(u(0.0, 0.0, last_t), v(0.0, 0.0, last_t), DT / 2)
    ui = [log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0]

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


# 
PID = ParticleTools.ParticleToDataframe(PI)

gr(display_type=:inline)
# plit each row in PID and a figure

tsub = range(start=1, stop=length(PID[:, 1]), step=5)

subtitle = "reset to windsea every $DT seconds\n"
p1 = plot(PID[tsub, 1] / (60 * 60), exp.(PID[tsub, 2]), marker=3, title=subtitle * "energy", xlabel="time (hours)", ylabel="e", label="V4") #|> display
p2 = plot(PID[tsub, 3], PID[tsub, 4], marker=3, markershape=:square, title="cg vector", xlabel="x", ylabel="y", label="V4") #|> display


# plot!(p1, PID[tsub, 1] / (60 * 60), FetchRelations.Eⱼ.(0.3 * abs(U10), PID[tsub, 1]) , marker=2, title="e", xlabel="time", ylabel="e", label="Fetch relations") #|> display

# plot!(p1, PID[tsub, 1] / (60 * 60), u.(0, 0, PID[tsub, 1]), marker=2, title="e", xlabel="time", ylabel="e", label="Fetch relations") #|> display
#plot!(p2, PID3[tsub3, 3], PID3[tsub3, 4], marker=2, title="cg vector", xlabel="x", ylabel="y", label="V3") #|> display

axlim = 10
plot!(p2, xlims=(-axlim, axlim), ylims=(-axlim, axlim))
plot!(p2, [0, 0], [-axlim, axlim], color=:black, linewidth=1, label=nothing)
plot!(p2, [-axlim, axlim], [0, 0], color=:black, linewidth=1, label=nothing)

tsubx = range(start=1, stop=length(PID[:, 1]), step=200)
time_sub = PID[tsubx, 1]
#plot quivers every qstep2
quiver!(p2, PID[tsubx, 3], PID[tsubx, 4], quiver=(u.(0, 0, time_sub) / 2, v.(0, 0, time_sub) / 2), color=:red, linewidth=2)#, label="wind")


#quiver!(p2, [0], [0], quiver=( [u.(0, 0, 0)], [v.(0, 0, 0)]), color=:red, linewidth=2, scale_units=:data, label="wind")

p3 = plot(PID[tsub, 5] / 1e3, PID[tsub, 6] / 1e3, marker=3, title="position", ylabel="postition", label="v4") #|> display

axlim = 200#1300
plot!(p3, xlims=(-axlim, axlim), ylims=(-axlim, axlim))
plot!(p3, [0, 0], [-axlim, axlim], color=:black, linewidth=1, label=nothing)
plot!(p3, [-axlim, axlim], [0, 0], color=:black, linewidth=1, label=nothing)

p4 = plot(PID[tsub, 5] / 1e3, exp.(PID[tsub, 2]), marker=3, title="e (x)", xlabel="x (km)", ylabel="e", label="V4") #|> display

plot(p1, p2, p3, p4, layout=(4, 1), legend=true, size=(600, 1600))


subtitle = "u$(U10)_v$(V10)_reset_to_windsea_dt$(DT)"
savefig(joinpath(plot_path_base, subtitle * "_continous_foreward.png"))


# %% test wind forcing coordinates 
Revise.retry()

# u(x::Number, y::Number, t::Number) = IfElse.ifelse.(x .> 10.0, 4.0, -4.0) * IfElse.ifelse.(y .> 30.0, 4.0, -4.0)  + t * 0.0
# v(x::Number, y::Number, t::Number) = IfElse.ifelse.(x .> 10.0, 4.0, -4.0) * IfElse.ifelse.(y .> 30.0, 4.0, -4.0)  + t * 0.0

u(x::Number, y::Number, t::Number) = IfElse.ifelse.(x .> 10.0, 4.0, -4.0) + y * 0.0 + t * 0.0
v(x::Number, y::Number, t::Number) = IfElse.ifelse.(y .> 15.0, 5.0, -5.0) + y * 0.0 + t * 0.0

ParticleState, default_ODE_parameters, WindSeamin, Const_ID = Init_Standard(u(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), DT)
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)

# define simple callback
condition(u, t, integrator) = 0.9 * u[1] > log(17)
affect!(integrator) = terminate!(integrator)
cb = ContinuousCallback(condition, affect!)

#used_ODE_params = (default_ODE_parameters..., M=[1 0; 0 1], x = 20.0, y= 30.0);
ODE_settings = PW.ODESettings(
    Parameters=used_ODE_params,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),
    #maximum energy threshold
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=2minutes,
    timestep=DT,
    total_time=T = 6days,
    callbacks=cb,
    save_everystep=false,
    maxiters=1e4,
    adaptive=true,
    dt=10,#60*10, 
    dtmin=1,#60*5, 
    force_dtmin=true,)

PI = InitParticleInstance(particle_system, ParticleState, ODE_settings, (0, 0), (20, 20), false, true)

PI.ODEIntegrator.p

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
    WindSeamin = FetchRelations.get_initial_windsea(u(PI.position_xy[1], PI.position_xy[2], last_t), v(PI.position_xy[1], PI.position_xy[2], last_t), DT / 2)
    ui = [log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0]

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

# %
for i in Base.Iterators.take(PI.ODEIntegrator, 3)

    time_step_local!(PI, DT)

end


# %%
# negaive values mean below values
# positive values mean above values

PID = ParticleTools.ParticleToDataframe(PI)

gr(display_type=:inline)
# plit each row in PID and a figure
tsub = range(start=1, stop=length(PID[:, 1]), step=5)
p2 = plot(PID[tsub, 3], PID[tsub, 4], marker=3, markershape=:square, title="cg vector", xlabel="x", ylabel="y", label="V4") #|> display


axlim = 10
plot!(p2, xlims=(-axlim, axlim), ylims=(-axlim, axlim))
plot!(p2, [0, 0], [-axlim, axlim], color=:black, linewidth=1, label=nothing)
plot!(p2, [-axlim, axlim], [0, 0], color=:black, linewidth=1, label=nothing)

tsubx = range(start=1, stop=length(PID[:, 1]), step=200)
time_sub = PID[tsubx, 1]
#plot quivers every qstep2
quiver!(p2, PID[tsubx, 3], PID[tsubx, 4], quiver=(u.(PI.position_xy[1], PI.position_xy[2], time_sub) / 2, v.(PI.position_xy[1], PI.position_xy[2], time_sub) / 2), color=:red, linewidth=2)#, label="wind")

