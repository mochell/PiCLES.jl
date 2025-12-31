#using Plots

using Pkg
Pkg.activate("PiCLES/")

import Plots as plt
using Setfield, IfElse

using PiCLES.ParticleSystems: particle_waves_v5 as PW

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleInstance, GetGroupVelocity
using PiCLES.Operators: TimeSteppers
using PiCLES.Simulations
using PiCLES.Operators.TimeSteppers: time_step!, movie_time_step!

# using PiCLES.ParticleMesh: TwoDGrid, TwoDGridNotes, TwoDGridMesh
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, TwoDCartesianGridStatistics
using PiCLES.Models.WaveGrowthModels2D

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units
import Oceananigans.Utils: prettytime

using PiCLES.Architectures
using GLMakie

using PiCLES.Operators.core_2D: GetGroupVelocity, speed
using PiCLES.Plotting.movie: init_movie_2D_box_plot


# %%
save_path = "plots/tests/T04_2D_regtest/"
mkpath(save_path)

##### basic parameters

# timestep
DT = 10minutes
# Characterstic wind velocities and std
U10, V10 = 5.0, 5.0

# Define basic ODE parameters

ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)

# define grid
grid = TwoDCartesianGridMesh(120e3, 31, 120e3, 31)

# example user function
u_func(x, y, t) = U10 + x * 0 + y * 0 + t * 0
v_func(x, y, t) = V10 + x * 0 + y * 0 + t * 0

# provide function handles for ODE and Simulation in the right format
u(x, y, t) = u_func(x, y, t)
v(x, y, t) = v_func(x, y, t)
winds = (u=u, v=v)


# define ODE system and parameters
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q);

default_ODE_parameters = (r_g=0.85, C_α=Const_Scg.C_alpha,
    C_φ=Const_ID.c_β, C_e=Const_ID.C_e, g=9.81)

Revise.retry()
# Default initial conditions based on timestep and chaeracteristic wind velocity
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT )
default_particle = ParticleDefaults(WindSeamin["lne"], WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0)

# ... and ODESettings
ODE_settings    =  PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=WindSeamin["lne"],
    #maximum energy threshold
    log_energy_maximum=log(17),#log(17),  # correcsponds to Hs about 16 m
    saving_step=DT,
    timestep=DT,
    total_time=T=6days,
    adaptive=true,
    dt=1e-3, #60*10, 
    dtmin=1e-4, #60*5, 
    force_dtmin=true,
    callbacks=nothing,
    save_everystep=false)



# %%
function make_reg_test(wave_model, save_path; plot_name="dummy", N=36)

    ### build Simulation
    wave_simulation = Simulation(wave_model, Δt=DT, stop_time=1hour)#1hours)
    initialize_simulation!(wave_simulation)

    # run simulation
    #run!(wave_simulation, cash_store=true, debug=true)
    # or, alternatively, make movie

    fig, n = init_movie_2D_box_plot(wave_simulation)

    #wave_simulation.stop_time += 1hour
    #N = 36
    #plot_name = "dummy"
    record(fig, save_path * plot_name * ".gif", 1:N, framerate=10) do i
        @info "Plotting frame $i of $N..."
        @info wave_simulation.model.clock
        movie_time_step!(wave_simulation.model, wave_simulation.Δt)

        n[] = 1
    end

end

# %%
Revise.retry()

# loop over U10 and V10 range
# gridmesh = [(i, j, per) for i in -10:10:10, j in -10:10:10, per in [false, true]]
gridmesh = [(i, j, per) for i in -10:10:10, j in -10:10:10, per in [true, false]]

#for I in CartesianIndices(gridmesh)
for (U10, V10, per) in gridmesh
    periodic = per# == true
    @show U10, V10, periodic

    u_func(x, y, t) = U10 + x * 0 + y * 0 + t * 0 #+ sign.(rand(-1:1)) .* 0.1
    v_func(x, y, t) = V10 + x * 0 + y * 0 + t * 0 #+ sign.(rand(-1:1)) .* 0.1
    u(x, y, t) = u_func(x, y, t)
    v(x, y, t) = v_func(x, y, t)
    winds = (u=u, v=v)

    #winds, u, v  =convert_wind_field_functions(u_func, v_func, x, y, t)

    particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)

    ## Define wave model
    wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
        winds=winds,
        ODEsys=particle_system,
        ODEsets=ODE_settings,  # ODE_settings
        ODEinit_type="wind_sea",  # default_ODE_parameters
        periodic_boundary=periodic,
        boundary_type="same",
        minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT), #
        movie=true)

    make_reg_test(wave_model, save_path, plot_name="T02_2D_periodic"*string(periodic)*"_U"*string(U10)*"_V"*string(V10))
end


# %% half domain tests
gridmesh = [(i, j, per) for i in -10:10:10, j in -10:10:10, per in [true, false]]
#for I in CartesianIndices(gridmesh)
for (U10, V10, per) in gridmesh
    periodic = per == true
    @show U10, V10, periodic

    # u_func(x, y, t) = U10 + x * 0 + y * 0 + t * 0 + sign.(rand(-1:1)) .* 0.1
    # v_func(x, y, t) = V10 + x * 0 + y * 0 + t * 0 + sign.(rand(-1:1)) .* 0.1

    # u_func(x, y, t) = IfElse.ifelse.(x .<60e3, U10, sign.(rand(-1:1)) .* 0.1) + y * 0 + t * 0
    # v_func(x, y, t) = (IfElse.ifelse.(x .< 60e3, V10, sign.(rand(-1:1)) .* 0.1) + y * 0) .* cos(t * 3 / (1 * 60 * 60 * 2π))
    u_func(x, y, t) = IfElse.ifelse.(x .< 60e3, U10, 0.0) + y * 0 + t * 0
    v_func(x, y, t) = (IfElse.ifelse.(x .< 60e3, V10, 0.0) + y * 0) .* cos(t * 3 / (1 * 60 * 60 * 2π))

    u(x, y, t) = u_func(x, y, t)
    v(x, y, t) = v_func(x, y, t)
    winds = (u=u, v=v)

    #winds, u, v  =convert_wind_field_functions(u_func, v_func, x, y, t)

    particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)

    ## Define wave model
    wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
        winds=winds,
        ODEsys=particle_system,
        ODEsets=ODE_settings,  # ODE_settings
        ODEinit_type="wind_sea",  # default_ODE_parameters
        periodic_boundary=periodic,
        boundary_type="same",
        minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT), #
        movie=true)

    make_reg_test(wave_model, save_path, plot_name="T02_2D_hald_domain_periodic" * string(periodic) * "_U" * string(U10) * "_V" * string(V10), N= 72)
end
