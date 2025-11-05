# %%
# ENV["JULIA_INCREMENTAL_COMPILE"] = true
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

# ds["lon"][:]
# ds["lat"][:]
# Grid.stats.xmin
# Grid.stats.xmax
# heatmap(transpose(v_grid.(Grid.data.x, Grid.data.y, 0)))
# heatmap(transpose(u_grid.(Grid.data.x, Grid.data.y, 0)))

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
Grid = TripolarGridMOM6.MOM6GridMesh(load_path * "ocean_hgrid_221123.nc", 2; MaskFile=load_path * "ocean_topo_tx2_3v2_240501.nc");
Grid = TripolarGridMOM6.MOM6GridMesh(load_path * "ocean_hgrid_221123.nc", 2; MaskFile=load_path * "ocean_topo_tx2_3v2_240501.nc");


heatmap( transpose(Grid.data.mask))
heatmap(Grid.data.x[:, 1], Grid.data.y[1, :], transpose(Grid.data.mask))

heatmap(Grid.data.x[:, 1], Grid.data.y[1, :], transpose(u.(Grid.data.x, Grid.data.y, 0)))
heatmap(Grid.data.x[:, 1], Grid.data.y[1, :], transpose(v.(Grid.data.x, Grid.data.y, 10)))
heatmap(Grid.data.x[:, 1], Grid.data.y[1, :], transpose(v.(Grid.data.x, Grid.data.y, 0)))

v_diff = v.(Grid.data.x, Grid.data.y, 1.0) - v.(Grid.data.x, Grid.data.y, 410400.0);
v_diff = v.(Grid.data.x, Grid.data.y, 1.0) - v.(Grid.data.x, Grid.data.y, 6hours);

heatmap(Grid.data.x[:, 1], Grid.data.y[1, :], transpose(v_diff))


# # % Make fake mask
# mask = ones(Bool, size(grid.data.x)) # 1 is ocean, 0 is land (?)
# mask[10:20, 5:10] .= 0
# #mask  = .!mask # to make one active block
# gridstats_mask = TwoDCartesianGridMesh(grid.stats; mask=mask)
# grid = TwoDCartesianGridMesh(gridstats_mask, grid.stats, ProjectionKernel)

# heatmap(transpose(v.(grid.data.x, grid.data.y, 0)))

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

#wave_model.ODEsettings.wind_min_squared = 0.0
#wave_model.minimal_state = 2 * wave_model.minimal_state

# ### build Simulation
wave_simulation = Simulation(wave_model, Δt=1hour, stop_time=2day)#1hours)
wave_simulation = Simulation(wave_model, Δt=1hour, stop_time=2day)#1hours)
initialize_simulation!(wave_simulation)
plot_particle_collection(wave_simulation.model)



#heatmap(wave_simulation.model.State[:,:,1])

# % timstepper test
Revise.retry()
plot_particle_collection(wave_model)

for i in 1:1:40
    TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt)

    if i % 3 == 0
        plot_particle_collection(wave_simulation.model)
        sleep(0.4)
    end
    wave_simulation.model.State[:, :, :] .= 0.0

end


# %%
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
wave_simulation = Simulation(wave_model, Δt=30minutes, stop_time=1.5days)#1hours)
wave_simulation = Simulation(wave_model, Δt=30minutes, stop_time=1.5days)#1hours)
# wave_simulation = Simulation(wave_model, Δt=1hour, stop_time=2hours)#1hours)



initialize_simulation!(wave_simulation)
plot_particle_collection(wave_model)

wave_model.ParticleCollection.on[Grid.data.mask.==1] .= 1
plot_particle_collection(wave_model)

run!(wave_simulation, cash_store=true, debug=false)
plot_particle_collection(wave_simulation.model)


# surface of a sphere
radius_earth = 6371e3  # in meters
surface_area_earth = 4 * π * radius_earth^2  # in square meters
surface_area_earth_km2 = surface_area_earth / 1e6  # in square kilometers

sqrt( surface_area_earth_km2 / (Grid.stats.Nx.N * Grid.stats.Ny.N) )

# %%
# particles = wave_model.ParticleCollection
# heatmap(transpose(Grid.data.mask))
# #particles.on .=1
# particles.on[Grid.data.mask .==1] .= 1
# particles[1,1]
# heatmap(transpose(particles.on))



# %%
using PiCLES.Plotting: PlotState_DoubleGlobe, PlotState_SingleGlobe, PlotState_DoubleGlobeSeam, OrthographicTwoMaps

fig = PlotState_DoubleGlobe(wave_simulation.model)
fig
Revise.retry()

# %%

using CairoMakie
lons, lats = wave_simulation.model.grid.data.x, wave_simulation.model.grid.data.y;

fig = Figure(size=(900, 1200), fontsize=22)
Energy = wave_simulation.model.State[:, :, 2];
ax1 = OrthographicTwoMapsSeam(fig[1,:], lons, lats, Energy)
fig

PlotState_DoubleGlobeSeam

# %%
fig = PlotState_DoubleGlobeSeam(wave_simulation.model)


# %%
# nodes = (model.grid.data.x, model.grid.data.y, 1:3)
# u_grid = LinearInterpolation(nodes, permutedims(model.State, [1, 2, 3]), extrapolation_bc=Periodic())

# nodes = (model.grid.data.x, model.grid.data.y)
# S1_grid = LinearInterpolation(nodes, permutedims(model.State[:,:,1], [1, 2]), extrapolation_bc=Periodic())

# interpolate(nodes, nodes, model.State[:,:,1], Gridded(Linear()))


# # %%
# A = rand(8, 20)
# nodes = ([x^2 for x = 1:8], [0.2y for y = 1:20])
# nodes = ([x^2 for x = 1:8], [0.2y for y = 1:20])
# node_pairs = [(x, y) for x in nodes[1], y in nodes[2]]

# itp = interpolate( node_pairs[:] , A[:])#, Gridded(Linear()))
# itp(4, 1.2)  # approximately A[2,6]

# # %%


# p = PolyharmonicSpline(3, [x y z], a)  # 3 is the order

# A = rand(8, 20)

# nodes = ([x^2 for x = 1:8], [0.2y for y = 1:20])
# pp = linear_interpolation(nodes, A)
# pp(1, 2)

# # %%
# model  = wave_simulation.model
# nodes = (model.grid.data.x[:], model.grid.data.y[:])
# nodes[1]
# nodes[2]
# zdata = model.State[:, :, 1][:]

# S1_grid = linear_interpolation(nodes, transpose(model.State[:, :, 1]), extrapolation_bc=Periodic())

# # %%
# using Dierckx

# nodes_grid = (model.grid.data.x, model.grid.data.y)

# nodes_grid[1][model.ocean_points]
# nodes_grid[2][model.ocean_points]

# N= 1000

# spl = Spline2D(nodes_grid[1][model.ocean_points][1:N], nodes_grid[2][model.ocean_points][1:N], zdata[1:N]; kx=1, ky=1, s=0.0)
# spl = Spline2D(nodes_grid[1][model.ocean_points], nodes_grid[2][model.ocean_points], model.State[:, :, 1][model.ocean_points]; kx=1, ky=1, s=0.0)

# model.State[:,:,1][model.ocean_points]

# maximum(nodes[1][1:300])
# minimum(nodes[1][1:300])

# xx = -180:1:180
# yy = -90:1:90
# node_pairs = [evaluate(spl,x, y) for x in xx, y in yy]

# heatmap(node_pairs)


# %% plot for proposal 

mask_state = copy(wave_model.State[:, :, :]);
#mask_state .= mask_state .* particles.on;
particles = wave_model.ParticleCollection


mask_state[particles.on.==0, :] .= NaN

size(mask_state)

e, m_x, m_y = mask_state[:, :, 1], mask_state[:, :, 2], mask_state[:, :, 3]
m_amp = sqrt.(m_x .^ 2 + m_y .^ 2)

m_amp[m_amp .< 0.005] .= 0.005
c_x = m_x .* e ./ ( 2 .* m_amp.^2)
c_y = m_y .* e ./ (2 .* m_amp.^2)
#significant wave height
hs = 4 .* sqrt.(e)

theta = tan.(c_y ./ c_x)

c_g_amp = sqrt.(c_x .^ 2 .+ c_y .^ 2)
peak_period = (4 * pi * c_g_amp) / 9.81
# %%

cg_lim =12
#function plot_particle_collection2(wave_model)
p = plot(layout=(2, 2), size=(1000, 600))
xx, yy = Grid.data.x[:, 1], Grid.data.y[1, :]
# xx, yy = Grid.data.x, Grid.data.y

u_speed = sqrt.(wave_model.winds.u.(Grid.data.x, Grid.data.y, wave_model.clock.time) .^ 2 + wave_model.winds.v.(Grid.data.x, Grid.data.y, wave_model.clock.time) .^ 2)

u_speed[particles.on.==0] .= NaN
heatmap!(p, xx, yy, transpose(particles.on), subplot=1, title="on | iter=" * string(wave_model.clock.iteration))
heatmap!(p, xx, yy, transpose(u_speed), subplot=1, title="Wind Speed (m/sec)", clims=(0, NaN), color=:dense)

#heatmap!(p, xx, yy, transpose(m_amp), subplot=2, title="boundary")
heatmap!(p, xx, yy, transpose(m_x), subplot=2, title="Zonal Momentum (m/s) ", clims=(-0.25, 0.25), color=:balance)
heatmap!(p, xx, yy, transpose(c_x), subplot=4, title="Zonal Group Velocity (m/s)", clims=(-cg_lim, cg_lim), color=:balance)

heatmap!(p, xx, yy, transpose(hs), subplot=3, title="Significant Wave Height Hs (meters)", clims=(0, 10), color=:dense)
# heatmap!(p, xx, yy, transpose(m_y), subplot=4, title="Meridional Momentum ", clims=(-0.25, 0.25), color=:balance)

# heatmap!(p, xx, yy, transpose(m_amp), subplot=4, title="Zonal Group Velocity ", color=:balance)

# heatmap!(p, xx, yy, transpose(c_y), subplot=4, title="Meridional Group Velocity (m/s)", clims=(-cg_lim, cg_lim), color=:balance)
# heatmap!(p, xx, yy, transpose(peak_period), subplot=6, title="Meridional Group Velocity", clims=(0, 20), color=:balance)

# title = plot!(title="Plot title", grid=false, showaxis=false, bottom_margin=-50Plots.px)
display(p)
# end

# plot_particle_collection2(wave_simulation.model)

# %%
# Example data
x = 1:10
y = 1:10
z = rand(10, 10)

# Available colormaps
colormaps = [:viridis, :plasma, :inferno, :magma, :cividis, :blues, :reds, :greens, :grays, :jet, :hsv, :rainbow]

# Plot with different colormaps
for cmap in colormaps
    p = plot(size=(1000, 600))
    heatmap(p, x, y, z, colormap=cmap, title=string(cmap))
    display(p)
end

# %%

