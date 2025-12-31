using ModelingToolkit, DifferentialEquations
using Plots
using Setfield

using PiCLES.ParticleSystems: particle_waves_v3beta as PW3
using PiCLES.ParticleSystems: particle_waves_v4 as PW4
using PiCLES.ParticleSystems: particle_waves_v5 as PW5

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleValues, InitParticleInstance
using PiCLES.ParticleMesh: TwoDGrid, TwoDGridNotes

# % Parameters
@register_symbolic u(x, y, t)
@register_symbolic v(x, y, t)

U10, V10 = 8.0, 8.0
dt_ODE_save = 60 * 20 # 3 min
DT = Float64(60 * 60) * 12 # seconds
T = 24 * 24 * 60 * 60 # seconds

# version 3
r_g0 = 0.85

# function to define constants for grouwth and dissipation
Const_ID = PW4.get_I_D_constant()
#
Const_Scg = PW4.get_Scg_constants()

# %
#u(x, y, t) = 0.01 - U10 * sin(t / (6 * 60 * 60 * 2π))
#v(x, y, t) = 0.01 - V10 * cos(t / (6 * 60 * 60 * 2π))

u(x, y, t) = - (U10 * cos(t *2  / (6 * 60 * 60 * 2π)) + 0.01)
v(x, y, t) = (V10 * cos(t *0.2 / (6 * 60 * 60 * 2π)) + 0.01)

# u(x, y, t) = - U10
# v(x, y, t) = + V10

winds = (u=u, v=v)

Revise.retry()

# define variables based on particle equation
t, x, y, c̄_x, c̄_y, lne, Δn, Δφ_p, r_g, C_α, C_φ, g, C_e = PW4.init_vars()

particle_equations3 = PW3.particle_equations_vec5(u, v, u, v, γ=Const_ID.γ, q=Const_ID.q)
@named particle_system3 = ODESystem(particle_equations3)

particle_equations4 = PW4.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)
@named particle_system4 = ODESystem(particle_equations4)

particle_system5 = PW5.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)

# define V4 parameters absed on Const NamedTuple:
default_ODE_parameters = Dict(r_g => r_g0, C_α => Const_Scg.C_alpha,
    C_φ => Const_ID.c_β, C_e => Const_ID.C_e, g => 9.81)

# %%
# define simple callback
condition(u, t, integrator) = 0.9 * u[1] > log(17)
affect!(integrator) = terminate!(integrator)
cb = ContinuousCallback(condition, affect!)

# define standard initial conditions
cg_u_local = sign(u(0, 0, 0)) * FetchRelations.c_g_U_tau(abs(u(0, 0, 0)), DT) / 100
cg_v_local = sign(v(0, 0, 0)) * FetchRelations.c_g_U_tau(abs(v(0, 0, 0)), DT) / 100
lne_local = log(FetchRelations.Eⱼ(0.01 * sqrt(u(0, 0, 0)^2 + v(0, 0, 0)^2), DT))

ODE_settings =  PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=lne_local,#log(FetchRelations.Eⱼ(0.1, DT)),
    #maximum energy threshold
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=dt_ODE_save,
    timestep=DT,
    total_time=T,
    callbacks=cb,
    save_everystep=false,
    maxiters=1e4,
    adaptive = true,
    dt = 10, 
    dtmin = 1e-3
)

grid = TwoDGrid(3, 3, 3, 3)
particle_defaults = ParticleDefaults(ODE_settings.log_energy_minimum, cg_u_local, cg_v_local, 0.0, 0.0)
ParticleState = copy(particle_defaults)

PI3 = InitParticleInstance(particle_system3, ParticleState, ODE_settings, (0, 0), false, true)
PI4 = InitParticleInstance(particle_system4, ParticleState, ODE_settings, (0, 0), false, true)
PI5 = InitParticleInstance(particle_system5, ParticleState, ODE_settings, (0, 0), false, true)
#ODE_parameters_NT = NamedTuple{Tuple(Symbol.(keys(ODE_settings.Parameters)))}(values(ODE_settings.Parameters))
#particle_system5(PI4.ODEIntegrator.u, PI4.ODEIntegrator.u, ODE_parameters_NT , PI4.ODEIntegrator.t)
#particle_system4(PI4.ODEIntegrator.u, PI4.ODEIntegrator.u, ODE_parameters_NT, PI4.ODEIntegrator.t)


function set_u_and_t!(integrator, u_new, t_new)
    integrator.u = u_new
    integrator.t = t_new
end

#clock_time = 0.0
for i in Base.Iterators.take(PI3.ODEIntegrator, 5)
    @info "t:", PI3.ODEIntegrator.t
    #@info "u:", PI.ODEIntegrator.u
    #@info "i:", i

    @info "proposed dt v3", get_proposed_dt(PI3.ODEIntegrator) / 60
    @info "proposed dt v4", get_proposed_dt(PI4.ODEIntegrator) / 60
    @info "proposed dt v5", get_proposed_dt(PI5.ODEIntegrator) / 60

    step!(PI3.ODEIntegrator, DT, true)
    step!(PI4.ODEIntegrator, DT, true)
    step!(PI5.ODEIntegrator, DT, true)

    #@info "u:", PI4.ODEIntegrator.u
    #clock_time += DT
    last_t3 = PI3.ODEIntegrator.t
    last_t4 = PI4.ODEIntegrator.t
    last_t5 = PI5.ODEIntegrator.t

    ## define here the particle state at time of resetting
    #ui = [log(exp(PI.ODEIntegrator.u[1]) * 0.5), PI.ODEIntegrator.u[2] / 2, PI.ODEIntegrator.u[3] / 2, 0.0, 0.0]
    #ui = [lne_local, cg_u_local, cg_v_local, 0.0, 0.0]
    #ui = PI.ODEIntegrator.u
    #ui = [lne_local, cg_u_local*0.1, cg_v_local*0.1, PI.ODEIntegrator.u[4]/2, PI.ODEIntegrator.u[5]/2]
    ui = [log(exp(PI3.ODEIntegrator.u[1]) * 1), PI3.ODEIntegrator.u[2], PI3.ODEIntegrator.u[3], 0.0, 0.0]
    set_u_and_t!(PI3.ODEIntegrator, ui, last_t3)

    ui = [log(exp(PI4.ODEIntegrator.u[1]) * 1), PI4.ODEIntegrator.u[2], PI4.ODEIntegrator.u[3], 0.0, 0.0]
    set_u_and_t!(PI4.ODEIntegrator, ui, last_t4)

    ui = [log(exp(PI5.ODEIntegrator.u[1]) * 1), PI5.ODEIntegrator.u[2], PI5.ODEIntegrator.u[3], 0.0, 0.0]
    set_u_and_t!(PI5.ODEIntegrator, ui, last_t5)

    # #set_u!(PI.ODEIntegrator, ui)
    #reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)
    #reinit!(PI3.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)

    # #set_t!(PI.ODEIntegrator, last_t )
    u_modified!(PI3.ODEIntegrator, true)
    u_modified!(PI4.ODEIntegrator, true)
    u_modified!(PI5.ODEIntegrator, true)

end


# %%

PID3 = ParticleTools.ParticleToDataframe(PI3)
PID4 = ParticleTools.ParticleToDataframe(PI4)
PID5 = ParticleTools.ParticleToDataframe(PI5)

gr(display_type=:inline)
# plit each row in PID and a figure
tsub = range(start=1, stop=length(PID4[:, 1]), step=5)
tsub3 = range(start=1, stop=length(PID3[:, 1]), step=5)
tsub5 = range(start=1, stop=length(PID5[:, 1]), step=5)

p1 = plot(PID4[tsub, 1] / (60 * 60), exp.(PID4[tsub, 2]), marker=3, title="e", xlabel="time (hours)", ylabel="e", label="V4") #|> display
p2 = plot(PID4[tsub, 3], PID4[tsub, 4], marker=3, markershape=:square, title="cg vector", xlabel="x", ylabel="y", label="V4") #|> display

plot!(p1, PID3[tsub3, 1] / (60 * 60), exp.(PID3[tsub3, 2]), marker=2, title="e", ylabel="e", label="V3") #|> display
plot!(p1, PID5[tsub5, 1] / (60 * 60), exp.(PID5[tsub5, 2]), marker=2, title="e", ylabel="e", label="V5") #|> display

dx_shift = 1
plot!(p2, PID3[tsub3, 3] .+ dx_shift, PID3[tsub3, 4], marker=2, title="cg vector", xlabel="x", ylabel="y", label="V3") #|> display
plot!(p2, PID5[tsub5, 3] .+ (dx_shift * 2), PID5[tsub5, 4], marker=2, title="cg vector", xlabel="x", ylabel="y", label="V5") #|> display

# add origin coordinates in plot2
axlim = 15
plot!(p2, xlims=(-axlim, axlim), ylims=(-axlim, axlim))
plot!(p2, [0, 0], [-axlim, axlim], color=:black, linewidth=1, label=nothing)
plot!(p2, [-axlim, axlim], [0, 0], color=:black, linewidth=1, label=nothing)

#quiver!(p2, [0], [0], quiver=( [u.(0, 0, 0)], [v.(0, 0, 0)]), color=:red, linewidth=2, scale_units=:data, label="wind")

p3 = plot(PID4[tsub, 5] / 1e3, PID4[tsub, 6] / 1e3, marker=3, title="position", ylabel="postition", label="v4") #|> display
dx_shift = 1e2
plot!(p3, PID3[tsub3, 5] / 1e3 .+ dx_shift , PID3[tsub3, 6] / 1e3, marker=3, title="position", ylabel="postition", label="v3") #|> display
plot!(p3, PID5[tsub5, 5] / 1e3 .+ (dx_shift * 2), PID5[tsub5, 6] / 1e3, marker=3, title="position", ylabel="postition", label="v5") #|> display

# add origin coordinates in plot3
axlim = 500#1300
plot!(p3, xlims=(-axlim, axlim), ylims=(-axlim, axlim))
plot!(p3, [0, 0], [-axlim, axlim], color=:black, linewidth=1, label=nothing)
plot!(p3, [-axlim, axlim], [0, 0], color=:black, linewidth=1, label=nothing)



p4 = plot(PID4[tsub, 5] / 1e3, exp.(PID4[tsub, 2]), marker=3, title="e (x)", xlabel="x (km)", ylabel="e", label="V4") #|> display

plot(p1, p2, p3, p4, layout=(4, 1), legend=true, size=(600, 1600))

# # add legend ot p1
# plot!(p1, legend=:topleft)

