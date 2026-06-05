
using Plots
using BenchmarkTools
using Revise
using Setfield

using Statistics

using JLD2, Printf,IfElse
# using ModelingToolkit: Num, @unpack, @register_symbolic, Symbolics, @named, ODESystem

# %%

using ParticleMesh: OneDGrid, OneDGridNotes
using PiCLES.Operators.core_1D: ParticleDefaults
using PiCLES: Simulation, WindEmulator, WaveGrowthModels1D, FetchRelations
using PiCLES.Simulations
using PiCLES.Plotting

#using PiCLES.Debugging
using PiCLES.ParticleSystems: particle_waves_v6 as PW

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units

# %%
# parametric wind forcing
U10, V = 5, 5 #m/s
#  rescale parameters for the right units.
T = 2days#24 * 2 * 60 * 60 # seconds
Lx = 1500kilometer# * 10e3  # km
DT = 20minutes #Float64(20 * 60) # seconds
Nx = 40
dt_ODE_save = 3minutes # 3 min
grid1d = OneDGrid(1e3, Lx - 1e3, Nx)

r_g0 = 0.85
# function to define constants for grouwth and dissipation
Const_ID = PW.get_I_D_constant() 
#@set Const_ID.γ = 0.88
Const_Scg = PW.get_Scg_constants()


@info "Init Forcing Field\n"
# create wind test fucntion
x_scale = 300kilometer
t_scale = 0.6 * 1days #(60 * 60 * 24 * 0.6)
dx = 5kilometer
#fake_winds(x, t) = U10 + x * 0 + t * 0# 
#fake_winds(x, t) = 0.01 + t * 0 + U10 * IfElse.ifelse.((x .> Lx / 3.0) & (x .< Lx * 2 / 3.0), 1 + 0 .* x, 0 .* x) * IfElse.ifelse.((t .> T / 4.0), 1 + 0 .* t, 0 .* t)
fake_winds(x, t) = WindEmulator.slopped_blob(x, t, U10, V, T, x_scale, t_scale)# <: WindEmulationFunction2

wind_grid = WindEmulator.IdealizedWindGrid(fake_winds, (Lx=Lx, T=T), (dx=dx, dt=DT))
# % define wind forcing globally as interp functions
# and convert to registered_symbolic need foir ODESystem
@register_symbolic u(x, t)
u(x, t) = WindEmulator.wind_interpolator(wind_grid).u(x, t)

contourf(wind_grid.x / dx, wind_grid.t / DT, transpose(wind_grid.u))
plot!(xlabel="x", ylabel="time") |> display

# -------------- start model definition -------------------------

# %% Load Particle equations and derive ODE system
particle_equations = PW.particle_equations(u, γ=Const_ID.γ, q=Const_ID.q)
@named particle_system = ODESystem(particle_equations)

# define variables based on particle equation
t, x, c̄_x, lne, r_g, C_α, g, C_e = PW.init_vars_1D()

# %% define storing stucture and populate inital conditions
default_ODE_parameters = Dict(r_g => r_g0, C_α => Const_Scg.C_alpha, C_e => Const_ID.C_e)

ODE_settings = PW.ODESettings(
        Parameters=default_ODE_parameters,
        # define mininum energy threshold
        log_energy_minimum=log(FetchRelations.Eⱼ(0.1, DT)),
        #maximum energy threshold
        log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
        saving_step=dt_ODE_save,
        timestep=DT,
        total_time=T,
        adaptive=true,
        dt=1e-3, #60*10, 
        dtmin=1e-9, #60*5, 
        force_dtmin=true,
)

# Default values for particle
particle_defaults = ParticleDefaults(log(FetchRelations.Eⱼ(0.5, DT)), 2e-1, 0.0)

# Define wavemodel 
wave_model = WaveGrowthModels1D.WaveGrowth1D(; grid=grid1d,
        winds=u,
        ODEsys=particle_system,
        ODEvars=PW.init_vars_1D(),
        layers=1,
        ODEsets=ODE_settings,  # ODE_settings
        ODEdefaults=particle_defaults,  # default_ODE_parameters
        periodic_boundary=false,
        boundary_type="default" #"wind_sea" 
)

# %% initialize Simulation 
wave_simulation = Simulation(wave_model, Δt=20minutes, stop_time=40hours)
initialize_simulation!(wave_simulation)#, particle_initials=nothing)#wave_model.ODEdefaults)

# run simulation
run!(wave_simulation, store=false, cash_store=true, debug=false)
@info "... finished\n"

Plotting.plot_results(wave_simulation)

plot_path_base = "plots/PW4/1D/"
mkpath(plot_path_base)
savefig(joinpath(plot_path_base, "PW3_vs_PW4.png"))

