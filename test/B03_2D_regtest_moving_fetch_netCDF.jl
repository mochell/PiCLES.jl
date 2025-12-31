

#using Plots
using Pkg
Pkg.activate("PiCLES/")
# Pkg.activate("../")

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
#using GLMakie

using PiCLES.Operators.core_2D: GetGroupVelocity, speed
# using PiCLES.Plotting.movie: init_movie_2D_box_plot

using NCDatasets
using Interpolations
using Dates: Dates as Dates
using Revise

# %%
save_path = "plots/tests/B03_2D_regtest_moving_fetch_netCDF/"
mkpath(save_path)

save_path_data = "data/work/B03_2D_regtest_moving_fetch_netCDF_local/"
mkpath(save_path_data)

load_path = "data/work/wind_data_moving_fetch/"
# load_path = "/glade/work/mhell/2022_particle_waves/wind_data_moving_fetch/"
#load_path = "data/work/wind_data_SWAMP/"

##### basic parameters
# timestep
DT = 30minutes

# Define basic ODE parameters
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)
Revise.retry()

function interpolate_winds(ds, multiplyer=0)

    Nxx = Int(ceil(ds.attrib["Nx"] * sqrt(2)^multiplyer))
    Nyy = Int(ceil(ds.attrib["Ny"] * sqrt(2)^multiplyer))

    # define grid based on
    grid = TwoDGrid(ds["x"][end], Nxx, ds["y"][end], Nyy)
    grid_mesh = TwoDGridMesh(grid, skip=1)
    gn = TwoDGridNotes(grid)

    # define time
    time_rel = (ds["time"][:] - ds["time"][1]) ./ convert(Dates.Millisecond, Dates.Second(1))
    T = time_rel[end]

    nodes = (ds["x"][:], ds["y"][:], time_rel)
    u_grid = LinearInterpolation(nodes, permutedims(ds["u10m"], [1, 2, 3]), extrapolation_bc=Flat())
    v_grid = LinearInterpolation(nodes, permutedims(ds["v10m"], [1, 2, 3]), extrapolation_bc=Flat())

    return grid, grid_mesh, gn, T, u_grid, v_grid
end

# define ODE system and parameters
#particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q);

Revise.retry()
# Default initial conditions based on timestep and chaeracteristic wind velocity
WindSeamin = FetchRelations.MinimalWindsea(2, 2, DT)

function make_reg_test_store(wave_model, save_path_name)

    ### build Simulation
    wave_simulation = Simulation(wave_model, Δt=DT, stop_time=wave_model.ODEsettings.total_time)#1hours)
    initialize_simulation!(wave_simulation)

    # run simulation & save simulation
    init_state_store!(wave_simulation, save_path_name)
    #run!(wave_simulation, cash_store=true, debug=true)
    run!(wave_simulation, store=true, cash_store=false, debug=false)
    close_store!(wave_simulation)

end

# or, alternatively, make movie
function make_reg_test_movie(wave_model, save_path_name; N=36)

    wave_simulation = Simulation(wave_model, Δt=DT, stop_time=wave_model.ODEsettings.total_time)#1hours)
    initialize_simulation!(wave_simulation)

    fig, n = init_movie_2D_box_plot(wave_simulation, resolution=(1800, 1200), name_string="T01")
    #wave_simulation.stop_time += 1hour
    #N = 36
    #plot_name = "dummy"
    record(fig, save_path_name * ".gif", 1:N, framerate=10) do i
        @info "Plotting frame $i of $N..."
        @info wave_simulation.model.clock
        movie_time_step!(wave_simulation.model, wave_simulation.Δt)

        n[] = 1
    end

end


# %%
# case = "SWAMP_Case_III"
# ncfile = load_path * case * ".nc"
# ds = Dataset(ncfile, "r")

# %%
# loop over U10 and V10 range
# case_list = ["MF_Case_I"]#, "MF_Case_II", "MF_Case_III", "MF_Case_IV" ]
case_list = ["MF_Case_I", "MF_Case_II", "MF_Case_III", "MF_Case_IV"]
#case_list = [ "SWAMP_Case_VIII"]
#for I in CartesianIndices(gridmesh)
for case in case_list
    # load netCDF file
    #ncfile = "data/work/wind_data/" * case * ".nc"
    ncfile = load_path * case * ".nc"
    ds = Dataset(ncfile, "r")

    grid, grid_mesh, gn, T, u_grid, v_grid = interpolate_winds(ds)

    @info "total time ", T

    u(x, y, t) = u_grid(x, y, t)
    v(x, y, t) = v_grid(x, y, t)
    winds = (u=u, v=v)

    #winds, u, v  =convert_wind_field_functions(u_func, v_func, x, y, t)
    particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)

    # ... and ODESettings
    ODE_settings = PW.ODESettings(
        Parameters=ODEpars,
        # define mininum energy threshold
        log_energy_minimum=WindSeamin["lne"],
        #maximum energy threshold
        log_energy_maximum=log(27),#log(17),  # correcsponds to Hs about 16 m
        saving_step=DT,
        timestep=DT,
        total_time=T,
        adaptive=true,
        dt=1e-3, #60*10, 
        dtmin=1e-4, #60*5, 
        force_dtmin=true,
        save_everystep=false)

    ## Define wave model
    wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid,
        winds=winds,
        ODEsys=particle_system,
        ODEsets=ODE_settings,  # ODE_settings
        ODEinit_type="wind_sea",  # default_ODE_parameters
        periodic_boundary=false,
        boundary_type="same",
        minimal_particle=FetchRelations.MinimalParticle(2, 2, DT), #
        movie=true)


    NN = Int(floor(wave_model.ODEsettings.total_time / wave_model.ODEsettings.timestep))

    # for saving data
    # when saving data
    save_path_select = save_path_data
    mkpath(save_path_select * case)
    make_reg_test_store(wave_model, save_path_select * case)

    # when plotting data
    # save_path_select = save_path
    # make_reg_test_movie(wave_model, save_path * case, N=NN)
end
# %%
