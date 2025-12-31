

import Plots as plt
using Setfield, IfElse

using PiCLES.ParticleSystems: particle_waves_v5 as PW

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleInstance, GetGroupVelocity
using PiCLES.Operators: TimeSteppers
using PiCLES.Simulations
using PiCLES.Operators.TimeSteppers: time_step!, movie_time_step!

using PiCLES.ParticleMesh: TwoDGrid, TwoDGridNotes, TwoDGridMesh
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
#sign.(rand(-1:1, 10, 10))


save_path = "plots/tests/T04_2D_growing_decaying_winds/"
mkpath(save_path)

##### basic parameters

# timestep
DT = 20minutes
# Characterstic wind velocities and std
U10, V10 = 10.0, 10.0

# Define basic ODE parameters
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)


# define grid
grid = TwoDGrid(260e3, 66, 80e3, 21)
mesh = TwoDGridMesh(grid, skip=1);
gn = TwoDGridNotes(grid);

# example user function
u_func(x, y, t) = U10 + x * 0 + y * 0 + t * 0
v_func(x, y, t) = V10 + x * 0 + y * 0 + t * 0

# provide function handles for ODE and Simulation in the right format
u(x, y, t) = u_func(x, y, t)
v(x, y, t) = v_func(x, y, t)
winds = (u=u, v=v)


# define ODE system and parameters
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q);

Revise.retry()
# Default initial conditions based on timestep and chaeracteristic wind velocity
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT)
default_particle = ParticleDefaults(WindSeamin["lne"], WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0)

# ... and ODESettings
ODE_settings = PW.ODESettings(
    Parameters=ODEpars,
    # define mininum energy threshold
    log_energy_minimum=WindSeamin["lne"],
    #maximum energy threshold
    log_energy_maximum=log(27),#log(17),  # correcsponds to Hs about 16 m
    saving_step=DT,
    timestep=DT,
    total_time=T = 6days,
    adaptive=true,
    dt=1e-3, #60*10, 
    dtmin=1e-4, #60*5, 
    force_dtmin=true,
    callbacks=nothing,
    save_everystep=false)


# %%

Revise.retry()

function make_reg_test(wave_model, save_path; plot_name="dummy", N=36, axline=0)

    ### build Simulation
    wave_simulation = Simulation(wave_model, Δt=DT, stop_time=1hour)#1hours)
    initialize_simulation!(wave_simulation)

    # run simulation
    #run!(wave_simulation, cash_store=true, debug=true)
    # or, alternatively, make movie

    fig, n = init_movie_2D_box_plot(wave_simulation; resolution=(1300, 800), name_string=plot_name, aspect=3, axline=axline)

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


# % half domain tests
Revise.retry()
gridmesh = [(i, j) for i in [-10,10], j in  [0]]
#gridmesh = [(i, j) for i in [10], j in [0]]

#for I in CartesianIndices(gridmesh)
for (U10, V10) in gridmesh
    @show U10, V10

    x0 =50e3
    Lx = (gn.Nx - 1) * gn.dx
    # u_func(x, y, t) = IfElse.ifelse.(x .< x0, U10, U10 * (1 -x/Lx) ) + y * 0 + t * 0
    # v_func(x, y, t) = IfElse.ifelse.(x .< x0, V10, V10 * (1 -x/Lx) ) + y * 0 + t * 0

    u_func(x, y, t) = IfElse.ifelse.(x .< x0, x*0+ 0.1, U10 * (x - x0) / (Lx-x0)) + y * 0 + t * 0
    v_func(x, y, t) = IfElse.ifelse.(x .< x0, x*0+ 0.1, V10 * (x - x0) / (Lx-x0)) + y * 0 + t * 0

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
        periodic_boundary=false,
        boundary_type="same",
        minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT), #
        # minimal_state=FetchRelations.MinimalState(2, 2, DT) * 1,
        movie=true)

    make_reg_test(wave_model, save_path, plot_name="T02_2D_growing_U" * string(U10) * "_V" * string(V10), N=20, axline=x0/1e3)
end

# %%

gridmesh = [(i, j) for i in [-10,10], j in 0:5:5]
#for I in CartesianIndices(gridmesh)
for (U10, V10) in gridmesh
    @show U10, V10

    x0 =50e3
    Lx = (gn.Nx - 1) * gn.dx
    u_func(x, y, t) = IfElse.ifelse.(x .< x0, U10, U10 * (1 -x/Lx) ) + y * 0 + t * 0 + 0.1
    v_func(x, y, t) = IfElse.ifelse.(x .< x0, V10, V10 * (1 -x/Lx) ) + y * 0 + t * 0 + 0.1

    # u_func(x, y, t) = IfElse.ifelse.(x .< x0, x*0+ 0.1, U10 * (x - x0) / (Lx-x0)) + y * 0 + t * 0
    # v_func(x, y, t) = IfElse.ifelse.(x .< x0, x*0+ 0.1, V10 * (x - x0) / (Lx-x0)) + y * 0 + t * 0
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
        periodic_boundary=false,
        boundary_type="same",
        minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT), #
        # minimal_state=FetchRelations.MinimalState(2, 2, DT) * 1,
        movie=true)

    make_reg_test(wave_model, save_path, plot_name="T02_2D_decaying_U" * string(U10) * "_V" * string(V10), N=60, axline=x0/1e3)
end


# %%
Revise.retry()
gridmesh = [(i, j) for i in [-10,0,10], j in 0:5:5]
#for I in CartesianIndices(gridmesh)
for (U10, V10) in gridmesh
    @show U10, V10

    x0 =130e3
    Lx = (gn.Nx - 1) * gn.dx
    u_func(x, y, t) = IfElse.ifelse.(x .< x0, U10, -U10 ) + y * 0 + t * 0 + 0.1
    v_func(x, y, t) = IfElse.ifelse.(x .< x0, V10, -V10 ) + y * 0 + t * 0 + 0.1

    # u_func(x, y, t) = IfElse.ifelse.(x .< x0, x*0+ 0.1, U10 * (x - x0) / (Lx-x0)) + y * 0 + t * 0
    # v_func(x, y, t) = IfElse.ifelse.(x .< x0, x*0+ 0.1, V10 * (x - x0) / (Lx-x0)) + y * 0 + t * 0
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
        periodic_boundary=false,
        boundary_type="same",
        minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT), #
        # minimal_state=FetchRelations.MinimalState(2, 2, DT) * 1,
        movie=true)

    make_reg_test(wave_model, save_path, plot_name="T02_2D_divergence_U" * string(U10) * "_V" * string(V10), N=100, axline=x0/1e3)
end

