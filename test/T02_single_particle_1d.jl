import Plots


#using PiCLES.ParticleSystems: particle_waves_v3beta as PW3
using PiCLES.ParticleSystems: particle_waves_v5 as PW5

import PiCLES: FetchRelations

using PiCLES.Operators.core_1D: ParticleDefaults, InitParticleValues, InitParticleInstance
using PiCLES.ParticleMesh: OneDGrid, OneDGridNotes

using DifferentialEquations

using PiCLES.Utils.ParticleTools 
using Plots

using Oceananigans.Units

# % Parameters

plot_path_base = "plots/tests/plot_path_base/"
mkpath(plot_path_base)
# @register_symbolic u(x, t)
# @register_symbolic u_x(x, t)

U10 = 10.1
r_g0 = 0.85
c_β = 4e-2
C_e0 = (2.35 / r_g0) * 2e-3 * c_β
γ = 0.88

dt_ODE_save = Float64(60 * 2) # 3 min
DT = Float64(60 * 30) * 2 #* 24 # seconds
T = 24 * 2 * 60 * 60 # seconds


## u must be always a function of x and t !!!
u(x, t) = x .* 0 + t * 0 + U10

Revise.retry()
particle_system = PW5.particle_equations(u, γ=γ)

# define variables based on particle equation
#t, x, c̄_x, lne, r_g, C_α, g, C_e = PW3.init_vars_1D()

# define storing stucture and populate inital conditions
default_ODE_parameters = (
    r_g = 0.85,
    C_α = -1.41,
    g = 9.81,
    C_e = C_e0,
)

typeof(default_ODE_parameters)

condition(u, t, integrator) = 0.9 * u[1] > log(17)
affect!(integrator) = terminate!(integrator)
cb = ContinuousCallback(condition, affect!)

# define initial conditions
WindSeamin = FetchRelations.get_initial_windsea(u(0, 0), 5minutes)
#WindSeamin = FetchRelations.MinimalWindsea( u(0, 0), 5minutes)

Revise.retry()

ODE_settings = PW5.ODESettings(
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
particle_defaults = ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar"], 0.0, 0.0, 0.0)
#particle_defaults = ParticleDefaults(ODE_settings.log_energy_minimum, cg_local, 1.51)

# initialize particle given the wind conditions:


ij = 2
xx = OneDGridNotes(grid1d).x[ij]
ParticleState, particle_on = InitParticleValues(particle_defaults, xx, u(xx, 0), DT)
PI4 = InitParticleInstance(particle_system, ParticleState, ODE_settings, ij, false, particle_on)

# %%
function set_u_and_t!(integrator, u_new, t_new)
    integrator.u = u_new
    integrator.t = t_new
end

clock_time = 0
NDT = 6
for i in Base.Iterators.take(PI4.ODEIntegrator, NDT)
    @info "x:", PI4.ODEIntegrator.u[3]
    @info "t:", PI4.ODEIntegrator.t, clock_time

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


# PI.ODEIntegrator.u
# affect!(PI.ODEIntegrator)
# %

#fig = figure(resolution=(800, 900))
PID4 = ParticleTools.ParticleToDataframe(PI4)

gr(display_type=:inline)
# plit each row in PID and a figure
p1  = plot(PID4[:, 4] / 1e3, exp.(PID4[:, 2]), color=:red, marker=2, label="PW4") #|> display

p2 = plot(PID4[:, 4] / 1e3, PID4[:, 3], marker=2, title="cg", ylabel="cg", xlabel="x (km)", label="PW3") #|> display

p3 = plot(PID4[:, 1] / 60 / 60, PID4[:, 4] / 1e3, marker=3, title="Hofmoeller", xlabel="time (hours)", ylabel="x (km)", label="PW3", bottom_margin=5*Plots.mm )
display

plot(p1, p2, p3, layout=(3, 1), legend=true, size=(600, 1200), left_margin=10*Plots.mm )

title!("PW4")
# save figure

savefig(plot_path_base*"PW3_positive4.png")

@info "c_g min,max", minimum(PID4[:, 3]), maximum(PID4[:, 3])


# %%  same for negatie winds:
u(x, t) = x .* 0 + t * 0 - 10.0

WindSeamin = FetchRelations.get_initial_windsea(u(0, 0), 5minutes)
particle_defaults = ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar"], 0.0)

ij = 2
xx = OneDGridNotes(grid1d).x[ij]
ParticleState, particle_on = InitParticleValues(copy(particle_defaults), xx, u(xx, 0), DT)
PI4 = InitParticleInstance(particle_system, ParticleState, ODE_settings, ij, false, particle_on)

clock_time = 0
NDT = 6
for i in Base.Iterators.take(PI4.ODEIntegrator, NDT)
    @info "x:", PI4.ODEIntegrator.u[3]
    @info "t:", PI4.ODEIntegrator.t, clock_time

    step!(PI4.ODEIntegrator, DT, true)
    clock_time += DT
    clock_time = PI4.ODEIntegrator.t
    #ui = [log(exp(PI4.ODEIntegrator.u[1]) * 1), PI4.ODEIntegrator.u[2], PI4.ODEIntegrator.u[3] * 0]
    ui = [PI4.ODEIntegrator.u[1] , PI4.ODEIntegrator.u[2], PI4.ODEIntegrator.u[3] * 0]

    #ui = [log(WindSeamin["E"]), WindSeamin["cg_bar"], 0.0]
    #ui = [lne_local, cg_local, PI4.ODEIntegrator.u[3]]

    #set_u!(PI4.ODEIntegrator, ui)
    #reinit!(PI4.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)

    set_u_and_t!(PI4.ODEIntegrator, ui, clock_time)

    #set_t!(PI4.ODEIntegrator, clock_time)
    u_modified!(PI4.ODEIntegrator, true)

end

PID4 = ParticleTools.ParticleToDataframe(PI4)

gr(display_type=:inline)
# plit each row in PID and a figure
p1 = plot(PID4[:, 4] / 1e3, exp.(PID4[:, 2]), color=:red, marker=2, label="PW4") #|> display
 
p2 = plot(PID4[:, 4] / 1e3, PID4[:, 3], marker=2, title="cg", ylabel="cg", xlabel="x (km)", label="PW3") #|> display

p3 = plot(PID4[:, 1] / 60 / 60, PID4[:, 4] / 1e3, marker=3, title="Hofmoeller", xlabel="time (hours)", ylabel="x (km)", label="PW3", bottom_margin=5 * Plots.mm)
display

plot(p1, p2, p3, layout=(3, 1), legend=true, size=(600, 1200), left_margin=10 * Plots.mm)

title!("PW4")
# save figure
savefig(plot_path_base* "PW4_negtive4.png")

@info "c_g min,max", minimum(PID4[:, 3]), maximum(PID4[:, 3])




####################################
# %% test with PIC remeshing algorithm ####
# define model -
using PiCLES: WaveGrowthModels1D, reset_boundary!
using Oceananigans.Units
using PiCLES.Simulations
using PiCLES.Plotting

Revise.retry()

u10 = 10
u(x, t) = x .* 0 + t * 0 + u10

# redefine model 
wave_model = WaveGrowthModels1D.WaveGrowth1D(; grid= OneDGrid(0, 30e3, 50),
    winds=u,
    ODEsys=PW5.particle_equations(u, γ=γ),
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",  # default_ODE_parameters
    minimal_particle=FetchRelations.MinimalParticle(2, 0, DT), #
    periodic_boundary=true,
    boundary_type="same"  # "default" #
    )

plot_path_base = "plots/tests/T02_1D_fetch/with_merge_rule/"
#mkdir(plot_path_base)
@info "experiment 1: positive winds, periodic \n"

wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=3hours)
initialize_simulation!(wave_simulation)
run!(wave_simulation, store=false, cash_store=true, debug=false)

Plotting.plot_results(wave_simulation, title="$u10 m/s, periodic=" * string(wave_model.periodic_boundary))
#title!("titlde")
savefig(joinpath(plot_path_base, "PW4_u$(u10)_per_" * string(wave_model.periodic_boundary) * ".png") )


# %%
@info "experiment 1: positive winds, non-periodic \n"
Revise.retry()

wave_model.periodic_boundary = false
# reset boundary array in model
reset_boundary!(wave_model)



wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=3hours)
initialize_simulation!(wave_simulation)
run!(wave_simulation, store=false, cash_store=true, debug=false)
Plotting.plot_results(wave_simulation, title="$u10 m/s, periodic=" * string(wave_model.periodic_boundary))
savefig(joinpath(plot_path_base, "PW4_u$(u10)_per_" * string(wave_model.periodic_boundary) * ".png"))


# %%
@info "experiment 1: nagative winds, perodic \n"
u10 = -10
u(x, t) = x .* 0 + t * 0 + u10
wave_model.periodic_boundary = true
# reset boundary array in model
reset_boundary!(wave_model)


wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=3hours)
initialize_simulation!(wave_simulation)
run!(wave_simulation, store=false, cash_store=true, debug=false)
Plotting.plot_results(wave_simulation, title="$u10 m/s, periodic=" * string(wave_model.periodic_boundary))
savefig(joinpath(plot_path_base, "PW4_u$(u10)_per_" * string(wave_model.periodic_boundary) * ".png"))

# %%
#Revise.retry()
@info "experiment 1: nagative winds, non-perodic \n"
u10 = -10
u(x, t) = x .* 0 + t * 0 + u10
wave_model.periodic_boundary = false
# reset boundary array in model
reset_boundary!(wave_model)


wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=3hours)
initialize_simulation!(wave_simulation)
run!(wave_simulation, store=false, cash_store=true, debug=false)
Plotting.plot_results(wave_simulation, title="$u10 m/s, periodic=" * string(wave_model.periodic_boundary))
savefig(joinpath(plot_path_base, "PW4_u$(u10)_per_" * string(wave_model.periodic_boundary) * ".png"))

@info "... finished\n"


