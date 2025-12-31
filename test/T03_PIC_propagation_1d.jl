using Plots
using BenchmarkTools
using Revise

using IfElse
using Statistics


using HDF5, JLD2

using DocStringExtensions
using DifferentialEquations

using PiCLES.ParticleMesh: OneDGrid, OneDGridNotes
using PiCLES.Operators.core_1D: ParticleDefaults
using PiCLES: Simulation, WindEmulator, WaveGrowthModels1D, FetchRelations
using PiCLES.Simulations

using PiCLES: ParticleInCell

#using PiCLES.Debugging
using PiCLES.ParticleSystems: particle_waves_v5 as PW

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units

using PiCLES.Operators.core_1D: GetParticleEnergyMomentum, init_z0_to_State! 
# %
# Default values
save_path_base = "data/1D_gaussian/"
plot_path_base = "plots/tests/T03_PIC_propagation-1d/with_merging_rules/"
mkpath(plot_path_base)
#savefig(joinpath(plot_path_base, "PW3_vs_PW4.png"))

# parametric wind forcing
U10, V = 10, 5 #m/s

#  rescale parameters for the right units.
T = 24 * 2 * 60 * 60 # seconds
Lx = 300 * 10e3  # km
DT = Float64(20 * 60) # seconds
Nx = 40
dt_ODE_save = 10 # 3 min

r_g0 = 0.85
# function to define constants for grouwth and dissipation

#
Const_Scg = PW.get_Scg_constants()


grid1d = OneDGrid(1e3, Lx - 1e3, Nx)
# create ID and save name
ID = "empl_eq_v4"
#save_path = joinpath( "data/1D_static/", parsed_args["ID"] )
#save_path = save_path_base * parset * "/" * ID * "/"
#plot_path = plot_path_base * parset * "/"

##mkpath(save_path)
# %
@info "Init Forcing Field\n"
# create wind test fucntion

#@register_symbolic u(x, t)

x_scale = 600e3
t_scale = (60 * 60 * 24 * 1.5)
dx = 5e3

u(x, t) = U10 + x * 0 + t * 0# 

# -------------- start model definition -------------------------
# %% Load Particle equations and derive ODE system
#using PiCLES.ParticleInCell: push_to_grid!
Revise.revise(ParticleInCell)

Revise.retry()
############ try v3 again first and check ways to import module
#particle_equations = PW.particle_equations(u, γ=γ,input=true, dissipation=true, peak_shift=true )
particle_system = PW.particle_rays()

# define variables based on particle equation
#t, x, c̄_x, lne, r_g, C_α, g, C_e = nothing

default_ODE_parameters = (r_g =r_g0, C_α = Const_Scg.C_alpha, C_e = Const_ID.C_e)


WindSeamin = FetchRelations.MinimalWindsea(U10, DT)
particle_defaults = ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar"], 0.0)
#particle_defaults = ParticleDefaults(log(FetchRelations.Eⱼ(5.0, DT)), 1e-4, 0.0)

ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),
    #maximum energy threshold
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=dt_ODE_save,
    timestep=DT,
    total_time=T,
    adaptive=true,
    dt=1e-3, #60*10, 
    dtmin=1e-12, #60*5, 
    force_dtmin=true,
)


wave_model = WaveGrowthModels1D.WaveGrowth1D(; grid=grid1d,
    winds=u,
    ODEsys=particle_system,
    ODEvars=nothing,
    layers=1,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",  # default_ODE_parameters
    periodic_boundary=false,
    boundary_type="same"
)




function convert_state_store_to_array(store::Vector{Any})
    store_data = cat(store..., dims=3)
    store_waves_data = permutedims(store_data, (3, 1, 2))
    wave_x = OneDGridNotes(wave_model.grid).x
    wave_time = collect(range(0, wave_simulation.stop_time + wave_simulation.Δt, step=wave_simulation.Δt))
    return (data=store_waves_data, x=wave_x, time=wave_time)
end

function plot_results(wave_simulation; title_m="title")
    output = convert_state_store_to_array(wave_simulation.store.store)

    store_waves_energy = output.data[:, :, 1]
    store_waves_mx = output.data[:, :, 2]
    #store_waves_my = output.data[:, :, 3];
    cg = store_waves_energy ./ store_waves_mx ./ 2

    # ## make a three panel plot for energy, mx, and cg 
    plot(heatmap(output.x / dx, output.time / DT, store_waves_energy, levels=20, colormap=:blues),
        heatmap(output.x / dx, output.time / DT, store_waves_mx, levels=20, colormap=:blues),
        heatmap(output.x / dx, output.time / DT, cg, levels=40, colormap=:blues),
        layout=(3, 1), size=(400, 800), title=[title_m * "  | energy" "mx" "cg"], xlabel="x (dx)", ylabel="time (DT)") |> display
end

# % run simulations

##### avection to the right #########
wave_simulation = Simulation(wave_model, Δt=20minutes, stop_time=24hours)
initialize_simulation!(wave_simulation)

WindSea2 = FetchRelations.get_initial_windsea(U10, 3hours)
cg_bar = round(WindSea2["cg_bar"])
cg_bar_min = sign(WindSeamin["cg_bar"])

for i in range(10, Integer(floor(length(wave_model.ParticleCollection) * 1 / 2)), step=1)
    PII = wave_model.ParticleCollection[i].ODEIntegrator
    ui = [WindSea2["E"], WindSea2["cg_bar"], PII.u[3]]
    PII.u = ui
    u_modified!(PII, true)
    init_z0_to_State!(wave_model.State, i, GetParticleEnergyMomentum(ui))
    @info i, wave_model.ParticleCollection[i].ODEIntegrator.u
end

run!(wave_simulation, store=false, cash_store=true, debug=false)

output = convert_state_store_to_array(wave_simulation.store.store)
#output.data
plot_results(wave_simulation, title_m="cg=$cg_bar m/s, env sign =$(cg_bar_min)\n")
savefig(joinpath(plot_path_base, "test1_forward.png"))

#ParticleInCell.merge!( wave_simulation.model.State[3, :] , [10.0, 3.0, 0], verbose=true)

# %%
##### avection to the left #########

wave_simulation = Simulation(wave_model, Δt=20minutes, stop_time=24hours)
initialize_simulation!(wave_simulation)

WindSea2 = FetchRelations.get_initial_windsea(-U10, 3hours)
cg_bar = round(WindSea2["cg_bar"])
cg_bar_min = sign(WindSeamin["cg_bar"])

for i in range(10, Integer(floor(length(wave_model.ParticleCollection) * 1 / 2)), step=1)
    PII = wave_model.ParticleCollection[i].ODEIntegrator
    ui = [WindSea2["E"], WindSea2["cg_bar"], PII.u[3]]
    PII.u = ui
    u_modified!(PII, true)
    init_z0_to_State!(wave_model.State, i, GetParticleEnergyMomentum(ui))
    @info i, wave_model.ParticleCollection[i].ODEIntegrator.u
end

run!(wave_simulation, store=false, cash_store=true, debug=false)

plot_results(wave_simulation, title_m="cg=$cg_bar m/s, env sign =$(cg_bar_min)\n")
savefig(joinpath(plot_path_base, "test1_negative.png"))



# %%
##### avection to the left #########

# Default values for particle
WindSeamin = FetchRelations.MinimalWindsea(-U10, DT)
particle_defaults = ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar"], 0.0)

#particle_defaults = ParticleDefaults(log((hs / 4)^2), 0.001, 0.0)
#particle_defaults = ParticleDefaults(log(FetchRelations.Eⱼ(5.0, DT)), 1e-4, 0.0)

wave_model = WaveGrowthModels1D.WaveGrowth1D(; grid=grid1d,
    winds=u,
    ODEsys=particle_system,
    ODEvars=nothing,
    layers=1,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",  # default_ODE_parameters
    periodic_boundary=false,
    boundary_type="same"
)

wave_simulation = Simulation(wave_model, Δt=20minutes, stop_time=24hours)
initialize_simulation!(wave_simulation)

WindSea2 = FetchRelations.get_initial_windsea(U10, 3hours)
cg_bar = round(WindSea2["cg_bar"])
cg_bar_min = sign(WindSeamin["cg_bar"])

for i in range(10, Integer(floor(length(wave_model.ParticleCollection) * 1 / 2)), step=1)
    PII = wave_model.ParticleCollection[i].ODEIntegrator
    ui = [WindSea2["E"], cg_bar, PII.u[3]]
    #ui = [log(energy), c_g, PII.u[3]]
    PII.u = ui
    u_modified!(PII, true)
    init_z0_to_State!(wave_model.State, i, GetParticleEnergyMomentum(ui))
    @info i, wave_model.ParticleCollection[i].ODEIntegrator.u
end

run!(wave_simulation, store=false, cash_store=true, debug=false)
#title!("cg=-2 m/s, env cg = 0.001")
plot_results(wave_simulation, title_m="cg=$cg_bar m/s, env sign =$(cg_bar_min)\n")
savefig(joinpath(plot_path_base, "test2_forward.png"))

# %%
wave_simulation = Simulation(wave_model, Δt=20minutes, stop_time=24hours)
initialize_simulation!(wave_simulation)

WindSea2 = FetchRelations.get_initial_windsea(-U10, 3hours)
cg_bar = round(WindSea2["cg_bar"])
cg_bar_min = sign(WindSeamin["cg_bar"])

for i in range(10, Integer(floor(length(wave_model.ParticleCollection) * 1 / 2)), step=1)
    PII = wave_model.ParticleCollection[i].ODEIntegrator
    ui = [WindSea2["E"], WindSea2["cg_bar"], PII.u[3]]
    PII.u = ui
    u_modified!(PII, true)
    init_z0_to_State!(wave_model.State, i, GetParticleEnergyMomentum(ui))
    @info i, wave_model.ParticleCollection[i].ODEIntegrator.u
end

run!(wave_simulation, store=false, cash_store=true, debug=false)

plot_results(wave_simulation, title_m="cg=$cg_bar m/s, env sign =$(cg_bar_min)\n")
savefig(joinpath(plot_path_base, "test2_negative.png"))

@info "... finished\n"
# %%
