ENV["JULIA_INCREMENTAL_COMPILE"] = true
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

using PiCLES.ParticleMesh: TwoDGrid, TwoDGridNotes, TwoDGridMesh
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, ProjectionKernel, TwoDCartesianGridStatistics

using PiCLES.Models.WaveGrowthModels2D

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units
import Oceananigans.Utils: prettytime

using PiCLES.Architectures
using PiCLES.Architectures: AbstractGridStatistics, CartesianGridStatistics

using PiCLES.Operators.core_2D: ParticleDefaults as ParticleDefaults2D

using Plots
using PiCLES.Operators.core_2D: GetGroupVelocity, speed
using PiCLES.Plotting.movie: init_movie_2D_box_plot

using Revise

using StaticArrays
using StructArrays
using BenchmarkTools

using PiCLES.Grids

using PiCLES.Operators.TimeSteppers: time_step!
using DifferentialEquations
using PiCLES.Operators: mapping_2D


# %%
using NCDatasets
using Dates: Dates as Dates
using Interpolations


load_path = "data/wind_data/era5_surfacewinds_202301_10days_6hourly_1deg.nc"
#load nc file
#ncfile = "data/wind_data/era5_surfacewinds_202301_10days_6hourly.nc"
ds = Dataset(load_path)


# Flat(), Periodic()
function interpolate_winds(ds, time_range = 1:20)

    # define time
    time_rel = (ds["time"][time_range] - ds["time"][1]) ./ convert(Dates.Millisecond, Dates.Second(1))
    #T = time_rel[end]

    nodes = (ds["lon"][:], ds["lat"][:], time_rel)
    u_grid = LinearInterpolation(nodes, permutedims(ds["U10N"][:, :, time_range], [1, 2, 3]), extrapolation_bc=Periodic())
    v_grid = LinearInterpolation(nodes, permutedims(ds["V10N"][:, :, time_range], [1, 2, 3]), extrapolation_bc=Periodic())

    return u_grid, v_grid, time_rel
end

u_grid, v_grid, time_rel = interpolate_winds(ds);

# %%
Revise.retry()
save_path = "plots/tests/T03_PIC_tripolar_land/"
mkpath(save_path)
pwd()
# % Parameters
U10, V10 = 15.0, -10.0
DT = 20minutes

ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)
# u(x, y, t) = IfElse.ifelse.(x .< 250e3, U10, 0.00) + y * 0.0 + t * 0.0
# v(x, y, t) = IfElse.ifelse.(x .< 250e3, V10, 0.00) + y * 0.0 + t * 0.0 .* cos(t * 5 / (1 * 60 * 60 * 2π))

# u(x, y, t) = U10 + x * 0.0 + y * 0.0 + t * 0.0
# v(x, y, t) = (V10 + x * 0.0 + y * 0.0) .* cos(t * 5 / (1 * 60 * 60 * 2π))

u(x, y, t) = u_grid(x, y, t)
v(x, y, t) = v_grid(x, y, t)
winds = (u=u, v=v)

load_path = "PiCLES/src/Grids/files/";
Grid = TripolarGridMOM6.MOM6GridMesh(load_path * "ocean_hgrid_221123.nc", 8; MaskFile=load_path * "ocean_topo_tx2_3v2_240501.nc");


heatmap( transpose(Grid.data.mask))
heatmap(Grid.data.x[:, 1], Grid.data.y[1, :], transpose(Grid.data.mask))

heatmap(Grid.data.x[:, 1], Grid.data.y[1, :], transpose(u.(Grid.data.x, Grid.data.y, 0)))
heatmap(Grid.data.x[:, 1], Grid.data.y[1, :], transpose(v.(Grid.data.x, Grid.data.y, 10)))
heatmap(Grid.data.x[:, 1], Grid.data.y[1, :], transpose(v.(Grid.data.x, Grid.data.y, 0)))

v_diff = v.(Grid.data.x, Grid.data.y, 1.0) - v.(Grid.data.x, Grid.data.y, 410400.0);
v_diff = v.(Grid.data.x, Grid.data.y, 1.0) - v.(Grid.data.x, Grid.data.y, 6hours);

heatmap(Grid.data.x[:, 1], Grid.data.y[1, :], transpose(v_diff))

# %%
Revise.retry()
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q,
    propagation=true,
    input=false,
    dissipation=false,
    peak_shift=false,
    direction=false,
);

default_ODE_parameters = (r_g=ODEpars.r_g, C_α=Const_Scg.C_alpha,
    C_φ=Const_ID.c_β, C_e=Const_ID.C_e, g=9.81);#, M=M);

# define setting and standard initial conditions
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT);
lne_local = log(WindSeamin["E"])
cg_u_local = WindSeamin["cg_bar_x"]
cg_v_local = WindSeamin["cg_bar_y"]

ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=lne_local,#log(FetchRelations.Eⱼ(0.1, DT)),
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



function plot_particle_collection(wave_model)
    particles = wave_model.ParticleCollection
    p = plot(layout=(3, 2), size=(1200, 1000))
    xx, yy = Grid.data.x[:, 1], Grid.data.y[1, :]
    # xx, yy = Grid.data.x, Grid.data.y
    heatmap!(p, xx, yy, transpose(particles.on), subplot=1, title="on | iter=" * string(wave_model.clock.iteration))
    heatmap!(p, xx, yy, transpose(particles.boundary), subplot=2, title="boundary")
    heatmap!(p, xx, yy, transpose(wave_model.State[:, :, 1]), subplot=3, title="State: Energy", clims=(0, 6), color=:blues)
    heatmap!(p, xx, yy, transpose(wave_model.State[:, :, 2]), subplot=4, title="State: x momentum ", clims=(-0.25, 0.25), color=:balance)
    heatmap!(p, xx, yy, transpose(wave_model.State[:, :, 3]), subplot=6, title="State: y momentum ", clims=(-0.25, 0.25), color=:balance)

    u_speed = sqrt.(wave_model.winds.u.(Grid.data.x, Grid.data.y, wave_model.clock.time) .^ 2 + wave_model.winds.v.(Grid.data.x, Grid.data.y, wave_model.clock.time) .^ 2)
    heatmap!(p, transpose(u_speed), subplot=5, title="wind speed", clims=(0, NaN))
    # title = plot!(title="Plot title", grid=false, showaxis=false, bottom_margin=-50Plots.px)
    display(p)
end

# %%

Revise.retry()

default_windsea = FetchRelations.get_initial_windsea(U10, V10, DT, particle_state=true)
default_particle = ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0)
wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=Grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",#ParticleDefaults2D(default_windsea[1], default_windsea[2], default_windsea[3], 0.0, 0.0),
    #ParticleDefaults2D(log(2), 0.0, 0.0, 0.0, 0.0), #"wind_sea",  # default_ODE_valuves
    periodic_boundary=false,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

Revise.retry()

particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q,
    propagation=true,
    input=true,
    dissipation=true,
    peak_shift=true,
    direction=true,
);

# default_windsea = FetchRelations.get_initial_windsea(U10*0.1, V10*0.1, DT, particle_state=true)
# default_particle = ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0)
wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=Grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    ODEinit_type="wind_sea",#ParticleDefaults2D(FetchRelations.MinimalParticle(2, 2, DT)),
    # ODEinit_type="wind_sea",
    periodic_boundary=false,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

#wave_model.ODEsettings.wind_min_squared = 0.0
#wave_model.minimal_state = 2 * wave_model.minimal_state

# ### build Simulation
wave_simulation = Simulation(wave_model, Δt=1hour, stop_time=2days)#1hours)
# wave_simulation = Simulation(wave_model, Δt=1hour, stop_time=2hours)#1hours)

initialize_simulation!(wave_simulation)
plot_particle_collection(wave_model)

run!(wave_simulation, cash_store=true, debug=false)
plot_particle_collection(wave_simulation.model)



# %%
using PiCLES.Plotting: PlotState_DoubleGlobe, PlotState_SingleGlobe, PlotState_DoubleGlobeSeam, OrthographicTwoMaps

fig = PlotState_DoubleGlobe(wave_simulation.model)
fig

Revise.retry()

