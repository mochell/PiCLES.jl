ENV["JULIA_INCREMENTAL_COMPILE"]=true
using Pkg
Pkg.activate(".")

import Plots as plt
using Setfield, IfElse

using PiCLES.ParticleSystems: particle_waves_v6 as PW

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D_spread: ParticleDefaults, InitParticleInstance, GetGroupVelocity, PointSource
using PiCLES.Operators: TimeSteppers
using PiCLES.Simulations
using PiCLES.Operators.TimeSteppers: time_step!, movie_time_step!

using PiCLES.ParticleMesh: TwoDGrid, TwoDGridNotes, TwoDGridMesh
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, TwoDCartesianGridStatistics
using PiCLES.Models.GeometricalOpticsModels

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units
import Oceananigans.Utils: prettytime

using PiCLES.Architectures

using PiCLES.Operators.core_2D_spread: GetGroupVelocity, speed

using Revise

save_path = "plots/test_case_1/"
mkpath(save_path)
mkpath(save_path*"data/")
mkpath(save_path*"plots/")
touch(save_path*"data/simu_info.csv")
touch(save_path*"data/sigma.csv")

# % Parameters
U10,V10           = 10.0, 10.0
dt_ODE_save       = 30minutes
DT                = 2minutes
# version 3
r_g0              = 0.85

# function to define constants 
Const_ID = PW.get_I_D_constant()
@set Const_ID.γ = 0.88
Const_Scg = PW.get_Scg_constants(C_alpha=- 1.41, C_varphi=1.81e-5)

u_func(x, y, t) = 0.
v_func(x, y, t) = 0.

u(x, y, t) = u_func(x, y, t)
v(x, y, t) = v_func(x, y, t)
winds = (u=u, v=v)

typeof(winds.u)
typeof(winds.u(1e3, 1e3, 11))

grid =TwoDCartesianGridMesh(22.5e4, 151, 22.5e4, 151; periodic_boundary=(false, true))
# mesh = TwoDGridMesh(grid, skip=1);
# gn = TwoDGridNotes(grid);

Revise.retry()

# define variables based on particle equation

particle_system = PW.particle_equations(u, v, γ=0.88, q=Const_ID.q, input=true, dissipation=false, peak_shift=false);

# define V4 parameters absed on Const NamedTuple:
default_ODE_parameters = (r_g = r_g0, C_α = Const_Scg.C_alpha, 
                                    C_φ = Const_ID.c_β, C_e = Const_ID.C_e, g= 9.81 );
#plt.scalefontsizes(1.75)

# define setting and standard initial conditions
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT );
lne_local = log(WindSeamin["E"])

ODE_settings    = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=lne_local,
    #maximum energy threshold
    log_energy_maximum=log(27),
    saving_step=dt_ODE_save,
    timestep=DT,
    total_time=T=6days,
    adaptive=true,
    dt=1e-3, #60*10, 
    dtmin=1e-4, #60*5, 
    force_dtmin=true,
    callbacks=nothing,
    save_everystep=false)


default_particle = ParticleDefaults(1, 0.0, 22, 112500, 15000., π)
Revise.retry()

wave_model = GeometricalOpticsModels.GeometricalOptics(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type=default_particle,
    periodic_boundary=false,
    boundary_type="same",
    movie=true,
    plot_steps=true,
    save_particles=true,
    plot_savepath="plots/test_case_1",
    angular_spreading_type="nonparametric",
    proba_covariance_init = [1e-50 0 0 0;
                             0 1e-50 0 0;
                             0 0 1e-50 0
                             0 0 0 1e-50],
    n_particles_launch=150000
    )

wave_simulation = Simulation(wave_model, Δt=0.75minutes, stop_time=60minutes)
initialize_simulation!(wave_simulation)


@time run!(wave_simulation, cash_store=false, debug=false)
