using Pkg
Pkg.activate("PiCLES/")

"""
Example: single-particle 2D with local time-varying winds.

Long-form exploratory run that validates forcing updates along particle trajectories.
"""

##using OrdinaryDiffEq
using Plots
using Setfield
using IfElse

# include("../src/ParticleSystems/particle_waves_v5.jl")

using PiCLES
using PiCLES.ParticleSystems: particle_waves_v6 as PW
using PiCLES.Utils: Init_Standard

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: InitParticleInstance
using Oceananigans.Units

plot_path_base = "plots/tests/T04_2D_single_particle/"
mkpath(plot_path_base)

# %%

u(x::Number, y::Number, t::Number) = (5.0 * cos(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
v(x::Number, y::Number, t::Number) = -(5.0 * sin(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0

# u(x::Number, y::Number, t::Number) = 6.0 + x * 0 + y * 0 + t * 0
# v(x::Number, y::Number, t::Number) = 6.0 + x * 0 + y * 0 + t * 0


DT = 2hours
ParticleState, default_ODE_parameters, WindSeamin, Const_ID = Init_Standard(u(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), DT)
particle_system = PW.particle_equations(γ=Const_ID.γ, q=Const_ID.q)

# define simple callback
# condition(u, t, integrator) = 0.9 * u[1] > log(17)
# affect!(integrator) = terminate!(integrator)
# cb = ContinuousCallback(condition, affect!)

# -------- old ODE setting 
# ODE_settings = PW.ODESettings(
#     Parameters=default_ODE_parameters,
#     # define mininum energy threshold
#     log_energy_minimum=log(WindSeamin["E"]),
#     #maximum energy threshold
#     log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
#     saving_step=2minutes,
#     timestep=DT,
#     total_time=T = 6days,
#     # callbacks=cb,
#     solver=nothing,
#     save_everystep=false,
#     maxiters=1e4,
#     adaptive=true,
#     dt=10,#60*10, 
#     dtmin=1,#60*5, 
#     force_dtmin=true,)

# -------- new ODE setting 
ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),
    #maximum energy threshold
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=2minutes,
    timestep=DT * 3,
    total_time=T = 6days,
    # callbacks=cb,
    # nothing uses standard: RK3.5
    solver  = nothing,
    reltol  = 1e-3,
    abstol  = 1e-4,
    dt      = 6minutes, # seconds
    dtmin   = 0.1,#60*5, 
    dtmax=10minutes)


using PiCLES.Solvers.RK35Integrator: ODEIntegrator, step!, solve!
using PiCLES.Operators.core_2D: initParticleDefaults

z_initials = initParticleDefaults(ParticleState)
t0 = 0.0

Forcing = PW.ForcingData(u_wind=u(0.0, 0.0, 0.0), v_wind=v(0.0, 0.0, 0.0))
@info Forcing

integ1 = ODEIntegrator(particle_system, z_initials, 0.0, ODE_settings.Parameters;
    forcing=Forcing,
    dt=ODE_settings.dt,
    reltol=ODE_settings.reltol,
    abstol=ODE_settings.abstol,
    dtmin=ODE_settings.dtmin, 
    dtmax=ODE_settings.dtmax)


Fcollection = PiCLES.custom_structures.ForcingCollection(u_wind=u, v_wind=v)

@info Fcollection

ts1, zs1 = solve!(integ1, ODE_settings.timestep*3; forcing=Fcollection, saveat=ODE_settings.saving_step)



# %%
PID = hcat(zs1...)'

gr(display_type=:inline)
# plit each row in PID and a figure

tsub = range(start=1, stop=length(PID[:, 1]), step=10)

time = ts1[tsub] / (60 * 60)
energy = exp.(PID[tsub, 1])
cg_vect = sqrt.(PID[tsub, 2] .^ 2 + PID[tsub, 3] .^ 2)


subtitle = "reset to windsea every $DT seconds\n"

p1 = plot(time, energy , marker=3, title=subtitle * "energy", xlabel="time (hours)", ylabel="e", label="V4")

p2 = plot(PID[tsub, 2], PID[tsub, 3], marker=3, markershape=:square, title="cg vector", xlabel="x", ylabel="y", label="V4")


axlim = 10
plot!(p2, xlims=(-axlim, axlim), ylims=(-axlim, axlim))
plot!(p2, [0, 0], [-axlim, axlim], color=:black, linewidth=1, label=nothing)
plot!(p2, [-axlim, axlim], [0, 0], color=:black, linewidth=1, label=nothing)


#quiver!(p2, [0], [0], quiver=( [u.(0, 0, 0)], [v.(0, 0, 0)]), color=:red, linewidth=2, scale_units=:data, label="wind")

# position
p3 = plot(PID[tsub, 4] / 1e3, PID[tsub, 5] / 1e3, marker=3, title="position", ylabel="postition", label="v4") #|> display

tsubx = range(start=1, stop=length(PID[:, 1]), step=100)
time_sub = ts1[tsubx] #/ (60 * 60)
#plot quivers every qstep2

quiver!(p3, PID[tsubx, 4] / 1e3, PID[tsubx, 5] / 1e3, quiver=(u.(0, 0, time_sub) / 1, v.(0, 0, time_sub) / 1), color=:red, linewidth=2)#, label="wind")


axlim = 400#1300
plot!(p3, xlims=(-axlim, axlim), ylims=(-axlim, axlim))
plot!(p3, [0, 0], [-axlim, axlim], color=:black, linewidth=1, label=nothing)
plot!(p3, [-axlim, axlim], [0, 0], color=:black, linewidth=1, label=nothing)

p4 = plot(cg_vect, exp.(PID[tsub, 1]), marker=3, title="e (x)", xlabel="cg (m/s)", ylabel="e (m^2 / s^2)", label="V4") #|> display

plot(p1, p2, p3, p4, layout=(2, 2), legend=true, size=(800, 800))

# subtitle = "u$(U10)_v$(V10)_reset_to_windsea_dt$(DT)"
# savefig(joinpath(plot_path_base, subtitle * "_continous_foreward.png"))

# %%

"""
----------- InitParticleInstance is updated acoording to the new forcing definitions
"""
PI = InitParticleInstance(particle_system, ParticleState, ODE_settings, Forcing, (0, 0), (1, 2), false, true)
PI.ODEIntegrator

@info PI.ODEIntegrator.forcing

function set_u_and_t!(integrator, u_new, t_new)
    integrator.u = u_new
    integrator.t = t_new
end

function time_step_local!(PI, DT)
    "take 1 step over DT"

    #@info "proposed dt", get_proposed_dt(PI.ODEIntegrator) / 60
    # step!(PI.ODEIntegrator, DT, true)
    ts, us = solve!(PI.ODEIntegrator, DT; forcing=Fcollection, save=true, saveat=60, maxiters=10^7)

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
    # u_modified!(PI.ODEIntegrator, true)

    # add_saveat!(PI.ODEIntegrator, PI.ODEIntegrator.t)
    # savevalues!(PI.ODEIntegrator)

    return ts, us
end

#ts, us = time_step_local!(PI, DT/10)
# PI.ODEIntegrator.dt
# PI.ODEIntegrator.u
# PI.ODEIntegrator.t

# %%
using DataFrames

for i in range(1, 8)
    ts, us =time_step_local!(PI, DT*i)
    df = DataFrame(permutedims(hcat(us...)), :auto)
    PID = insertcols!(df, 1, :t => ts)


    gr(display_type=:inline)
    # plit each row in PID and a figure

    tsub = range(start=1, stop=length(PID[:, 1]), step=5)

    subtitle = "reset to windsea every $DT seconds\n"
    p1 = plot(PID[tsub, 1] / (60 * 60), exp.(PID[tsub, 2]), marker=3, title=subtitle * "energy", xlabel="time (hours)", ylabel="e", label="V4") #|> display
    p2 = plot(PID[tsub, 3], PID[tsub, 4], marker=3, markershape=:square, title="cg vector", xlabel="x", ylabel="y", label="V4") #|> display

    axlim = 10
    plot!(p2, xlims=(-axlim, axlim), ylims=(-axlim, axlim))
    plot!(p2, [0, 0], [-axlim, axlim], color=:black, linewidth=1, label=nothing)
    plot!(p2, [-axlim, axlim], [0, 0], color=:black, linewidth=1, label=nothing)

    tsubx = range(start=1, stop=length(PID[:, 1]), step=200)
    time_sub = PID[tsubx, 1]
    #plot quivers every qstep2
    quiver!(p2, PID[tsubx, 3], PID[tsubx, 4], quiver=(u.(0, 0, time_sub) / 2, v.(0, 0, time_sub) / 2), color=:red, linewidth=2)#, label="wind")

    p3 = plot(PID[tsub, 5] / 1e3, PID[tsub, 6] / 1e3, marker=3, title="position", ylabel="postition", label="v4") #|> display

    axlim = 200#1300
    plot!(p3, xlims=(-axlim, axlim), ylims=(-axlim, axlim))
    plot!(p3, [0, 0], [-axlim, axlim], color=:black, linewidth=1, label=nothing)
    plot!(p3, [-axlim, axlim], [0, 0], color=:black, linewidth=1, label=nothing)

    p4 = plot(PID[tsub, 5] / 1e3, exp.(PID[tsub, 2]), marker=3, title="e (x)", xlabel="x (km)", ylabel="e", label="V4") #|> display

    plot(p1, p2, p3, p4, layout=(4, 1), legend=true, size=(600, 1600))

    # subtitle = "u$(U10)_v$(V10)_reset_to_windsea_dt$(DT)"
    # savefig(joinpath(plot_path_base, subtitle * "_continous_foreward.png"))
    display(plot(p1, p2, p3, p4, layout=(2, 2), legend=true, size=(1200, 1200)))

end

# display(plot(p1, p2, p3, p4, layout=(2, 2), legend=true, size=(1200, 1200)))

# %%
