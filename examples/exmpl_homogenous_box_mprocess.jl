#JULIA_NUM_THREADS = 4
#ENV["JULIA_NUM_THREADS"] = 4

using Base.Threads
nthreads()

import Plots as plt

# CPU initialization
using Distributed
addprocs(4)


@everywhere begin
    using Setfield
    using PiCLES.ParticleSystems: particle_waves_v6 as PW

    import PiCLES: FetchRelations, ParticleTools
    using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleInstance, GetGroupVelocity
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
    using PiCLES.Plotting.movie: init_movie_2D_box_plot
end
# debugging:
#using ProfileView

# %%
save_path = "plots/examples/homogenous_box/"
mkpath(save_path)

@everywhere begin
    # % Parameters
    U10, V10 = 10.0, 10.0
    dt_ODE_save = 30minutes
    DT = 30minutes
    # version 3
    r_g0 = 0.85

    # function to define constants 
    Const_ID = PW.get_I_D_constant()
    @set Const_ID.γ = 0.88
    Const_Scg = PW.get_Scg_constants(C_alpha=-1.41, C_varphi=1.81e-5)


    u(x, y, t) = U10 #* sin(t / (6 * 60 * 60 * 2π)) * sin(x / 50e3) * sin(y / 50e3)
    v(x, y, t) = V10 #* cos(t / (6 * 60 * 60 * 2π)) * sin(x / 50e3) * sin(y / 50e3)

    winds = (u=u, v=v)

    grid = TwoDGrid(100e3, 51, 100e3, 51)
    #grid = TwoDGrid(20e3, 21, 20e3, 21)
    mesh = TwoDGridMesh(grid, skip=1);
    gn = TwoDGridNotes(grid);

end

Revise.retry()

# define variables based on particle equation

#ProfileView.@profview 
#ProfileView.@profview 
@everywhere begin
    particle_system = PW.particle_equations(u, v, γ=0.88, q=Const_ID.q, input=true, dissipation=true);
    #particle_equations = PW3.particle_equations_vec5(u, v, u, v, γ=0.88, q=Const_ID.q);

    # define V4 parameters absed on Const NamedTuple:
    default_ODE_parameters = (r_g=r_g0, C_α=Const_Scg.C_alpha,
        C_φ=Const_ID.c_β, C_e=Const_ID.C_e, g=9.81);

    # define setting and standard initial conditions
    WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT);
    #WindSeamin = FetchRelations.MinimalWindsea(u(0, 0, 0), v(0, 0, 0), DT / 2)
    #WindSeamin = FetchRelations.get_initial_windsea(u(0, 0, 0), v(0, 0, 0), DT/5)
    lne_local = log(WindSeamin["E"])
    cg_u_local = WindSeamin["cg_bar_x"]
    cg_v_local = WindSeamin["cg_bar_y"]

    ODE_settings = PW.ODESettings(
        Parameters=default_ODE_parameters,
        # define mininum energy threshold
        log_energy_minimum=lne_local,#log(FetchRelations.Eⱼ(0.1, DT)),
        #maximum energy threshold
        log_energy_maximum=log(27),#log(17),  # correcsponds to Hs about 16 m
        saving_step=dt_ODE_save,
        timestep=DT,
        total_time=T = 6days,
        adaptive=true,
        dt=1e-3, #60*10, 
        dtmin=1e-4, #60*5, 
        force_dtmin=true,
        callbacks=nothing,
        save_everystep=false)


    default_particle = ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0)
end

Revise.retry()
wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",  # default_ODE_parameters
    periodic_boundary=false,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

typeof(wave_model.State)


#global wave_model.ParticleCollection
### build Simulation
#wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=4hours)#1hours)
wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=6hour)#1hours)
initialize_simulation!(wave_simulation)

# %% 
using PiCLES.Operators: mapping_2D
using PiCLES.Operators.TimeSteppers: mean_of_state
import PiCLES.Operators.TimeSteppers: time_step!
# run single timestep


advance_wrapper(f, state, Fcol, grid, winds, dt, emax, windmin, boundary, defaults) = x -> f(x, state, Fcol, grid, winds, dt, emax, windmin, boundary, defaults)
remesh_wrapper(f, state, winds, time, sets, dt, minpar, minstate, defaults) = x -> f(x, state, winds, time, sets, dt, minpar, minstate, defaults)

######## things to try:
# - ParticleCollection should be a global variable
# - try threads instead of processes for remeshing
# - try @async @sync for remeshing -- doesn't do much because task is still done by the same core.


"""
time_step!(model, Δt; callbacks=nothing)

advances model by 1 time step:
1st) the model.ParticleCollection is advanced and then 
2nd) the model.State is updated.
clock is ticked by Δt

callbacks are not implimented yet

"""
function time_step!(model::Abstract2DModel, Δt::Float64, workers::WorkerPool ; callbacks=nothing, debug=false)

    # temporary FailedCollection to store failed particles
    FailedCollection = Vector{AbstractMarkedParticleInstance}([])

    #print("mean energy before advance ", mean_of_state(model), "\n")
    if debug
        @info "before advance"
        @info maximum(model.State[:, :, 1]), maximum(model.State[:, :, 2]), maximum(model.State[:, :, 3])
        model.FailedCollection = FailedCollection
    end

    @info "advance"
    # @time @threads for a_particle in model.ParticleCollection
    #     #@info a_particle.position_ij
    #     mapping_2D.advance!(a_particle, model.State, FailedCollection,
    #         model.grid, model.winds, Δt,
    #         model.ODEsettings.log_energy_maximum,
    #         model.ODEsettings.wind_min_squared,
    #         model.periodic_boundary,
    #         model.ODEdefaults)
    # end

    advance_wrap = advance_wrapper(mapping_2D.advance!,
            model.State, FailedCollection,
            model.grid, model.winds, Δt,
            model.ODEsettings.log_energy_maximum,
            model.ODEsettings.wind_min_squared,
            model.periodic_boundary,
            model.ODEdefaults)

    @time pmap(advance_wrap, workers, model.ParticleCollection)


    print("mean energy after advance ", mean_of_state(model), "\n")

    if debug
        @info "advanced: "
        @info maximum(model.State[:, :, 1]), maximum(model.State[:, :, 2]), maximum(model.State[:, :, 3])
        #@info model.State[8:12, 1], model.State[8:12, 2]
        @info model.clock.time, model.ParticleCollection[10].ODEIntegrator.t
        @info "winds:", model.winds.u(model.ParticleCollection[10].ODEIntegrator.u[4], model.ParticleCollection[10].ODEIntegrator.u[5], model.ParticleCollection[10].ODEIntegrator.t)
    end

    @info "re-mesh"
    @threads for a_particle in model.ParticleCollection
        mapping_2D.remesh!(a_particle, model.State,
            model.winds, model.clock.time,
            model.ODEsettings, Δt,
            model.minimal_particle,
            model.minimal_state,
            model.ODEdefaults)
    end
    #model.ParticleCollection = fetch(model.ParticleCollection)

    # remesh_wrap = remesh_wrapper(mapping_2D.remesh!,
    #         model.State,
    #         model.winds, model.clock.time,
    #         model.ODEsettings, Δt,
    #         model.minimal_particle,
    #         model.minimal_state,
    #         model.ODEdefaults)
    # pmap(remesh_wrap, workers, model.ParticleCollection);

    if debug
        @info "remeshed: "
        #@info model.State[8:12, 1], model.State[8:12, 2]
        @info maximum(model.State[:, :, 1]), maximum(model.State[:, :, 2]), maximum(model.State[:, :, 3])
        @info model.clock.time, model.ParticleCollection[10].ODEIntegrator.t

    end
    print("mean energy after remesh ", mean_of_state(model), "\n")

    tick!(model.clock, Δt)
end

# external definitions
advance_workers = WorkerPool([2, 3, 4, 5])
workers2 = WorkerPool([2, 3])
workers2b = WorkerPool([1, 2])
workers1 = WorkerPool([1])

# %%
for i in 1:3
    
    wave_simulation.model.State .= 0.0

    time_step!(wave_simulation.model, wave_simulation.Δt, workers1, debug=false)
    #time_step!(wave_simulation.model, wave_simulation.Δt, debug=false)
    istate = wave_simulation.model.State;

    p1 = plt.heatmap(gn.x / 1e3, gn.y / 1e3, istate[:, :, 1])
    display(p1)
    sleep(0.5)
end
# %%
@fetchfrom 2 InteractiveUtils.varinfo()
using CUDA
@sync_gpu
# %

@time time_step!(wave_simulation.model, wave_simulation.Δt, advance_workers, debug=false)

@time time_step!(wave_simulation.model, wave_simulation.Δt, workers1, debug=false)


#wave_simulation.model.ParticleCollection
#fetch(wave_simulation.model.ParticleCollection)

# code to make parallel map



#init_state_store!(wave_simulation, save_path)
#wave_simulation.model.MovieState = wave_simulation.model.State

@everywhere run!(wave_simulation, cash_store=true, debug=true)
#reset_simulation!(wave_simulation)
# run simulation
#ProfileView.@profview run!(wave_simulation, cash_store=true, debug=true)


istate = wave_simulation.store.store[end];
p1 = plt.heatmap(gn.x / 1e3, gn.y / 1e3, istate[:, :, 1])





# %% Movie, just for fun ...
Revise.retry()
wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",  # default_ODE_parameters
    periodic_boundary=false,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

wave_simulation = Simulation(wave_model, Δt=5minutes, stop_time=4hours)#1hours)
initialize_simulation!(wave_simulation)

#reset_simulation!(wave_simulation, particle_initials=copy(wave_model.ODEdefaults))
fig, n = init_movie_2D_box_plot(wave_simulation, name_string="Rotating Stationary Winds")

#wave_simulation.stop_time += 1hour
N = 10
record(fig, save_path * "homogenous_box_nonper.gif", 1:N, framerate=10) do i
    @info "Plotting frame $i of $N..."
    #@time for _ = 1:10
    #run!(wave_simulation, store=false)
    @info wave_simulation.model.clock
    movie_time_step!(wave_simulation.model, wave_simulation.Δt)
    #coupled_time_step!(ocean_simulation, ice_simulation)
    #end
    n[] = 1
end
