# %%
ENV["JULIA_INCREMENTAL_COMPILE"] = true
# ENV["JULIA_NUM_THREADS"] = "14"
using Pkg
Pkg.activate("PiCLES/")

using PiCLES.ParticleSystems: particle_waves_v5 as PW
using PiCLES.ParticleSystems: particle_waves_fake as PW_fake


using PiCLES.Simulations
using PiCLES.Models.WaveGrowthModels2D

using Oceananigans.Units

using PiCLES.Operators: TimeSteppers

using PiCLES.Operators.core_2D: ParticleDefaults
using PiCLES.Grids
using PiCLES

using Revise
using BenchmarkTools

using PiCLES.Plotting: PlotState_DoubleGlobe, PlotState_SingleGlobe, PlotState_DoubleGlobeSeam, OrthographicTwoMaps, OrthographicTwoMapsSeam
using Plots

using NCDatasets
using Interpolations
using Dates: Dates as Dates

using Base.Threads
@info "Num. of threads", nthreads()
# %%

plot_path_base = "plots/tests/T03_sphere_moving_box/"
mkpath(plot_path_base)
pwd()

# load_path = "data/work/wind_data_SWAMP/"
load_path ="/home/momme.hell/2022_particle_waves//wind_data/"

function intialize_winds(ds)
    Nx, Ny = ds.attrib["Nx"], ds.attrib["Ny"]
    lon_min, lon_max = ds["lon"][1], ds["lon"][end]
    lat_min, lat_max = ds["lat"][1], ds["lat"][end]
    


    gridd = Grids.SphericalGrid.TwoDSphericalGridMesh(lon_min, lon_max, Nx, lat_min, lat_max, Ny; periodic_boundary=(false, false))

    # time 
    time_rel = (ds["time"][:] - ds["time"][1]) ./ convert(Dates.Millisecond, Dates.Second(1)) # time in seconds relative to start time
    T_end = time_rel[end]# / (60 * 60 * 24) # days
    Ntime = length(time_rel)

    @info "resolution degrees", gridd.stats.dx_deg, gridd.stats.dy_deg
    @info "resolution km     ", gridd.stats.dx_deg * 110, gridd.stats.dy_deg * 110
    @info "time days        ", T_end / (60 * 60 * 24)
    @info "time steps       ", length(time_rel)


    nodes = (ds["lon"][:], ds["lat"][:], time_rel)
    u_grid = LinearInterpolation(nodes, permutedims(ds["u10m"], [1, 2, 3]), extrapolation_bc=Flat())
    v_grid = LinearInterpolation(nodes, permutedims(ds["v10m"], [1, 2, 3]), extrapolation_bc=Flat())

    wind_attrs = (Nx=Nx, Ny=Ny, 
                lon_min=lon_min, lon_max=lon_max, 
                lat_min=lat_min, lat_max=lat_max,
                T_end=T_end, Ntime=Ntime)

    return gridd, u_grid, v_grid, wind_attrs
end

case = "Test01_moving_patch"
ncfile = load_path * case * ".nc"
ds = Dataset(ncfile, "r")


# %%
grid_data, u_grid, v_grid, wind_attrs = intialize_winds(ds)

u(x, y, t) = u_grid(x, y, t)
v(x, y, t) = v_grid(x, y, t)
winds = (u=u, v=v)

close(ds)

# %%

p = Plots.heatmap(transpose(u.(grid_data.data.x, grid_data.data.y, 60 * 60 * 000)))
display(p)


p = Plots.heatmap(transpose(u.(grid_data.data.x, grid_data.data.y, 60 * 60 * 200)))
display(p)

# %%

u.(grid_data.data.x, grid_data.data.y, 60 * 60 * 000)


# %%

# T = (ds["time"][end] - ds["time"][1])/convert(Dates.Millisecond, Dates.Day(1))
T = 18.25days
DT = 10minutes #5minutes * 6
U10, V10 = -1.0, 1.0

# %% define grid
# minium radius: 18.5 km
# maximum windspeed 58 m/s
# Nx_data, Ny_data = length(ds["lon"]), length(ds["lat"])

# final resolution
# Nx, Ny = Int(ceil(Nx_data*2.2)), Int(ceil(Ny_data*2.2))

# okey resolution
# Nx, Ny = Int(ceil(Nx_data*1.5)), Int(ceil(Ny_data*1.5))

# test resolution
Nx, Ny = Int(ceil(wind_attrs.Nx * 1)), Int(ceil(wind_attrs.Ny * 1))

grid_model = Grids.SphericalGrid.TwoDSphericalGridMesh(wind_attrs.lon_min, wind_attrs.lon_max, Nx, wind_attrs.lat_min, wind_attrs.lat_max, Ny; periodic_boundary=(false, false))
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)

@info "resolution degrees", grid_model.stats.dx_deg, grid_model.stats.dy_deg
@info "resolution km     ", grid_model.stats.dx_deg * 110, grid_model.stats.dy_deg * 110


## ------------------ how interpolation works here is not not correct here ..   
p = Plots.contourf(grid_model.data.x[:, 2], grid_model.data.y[2, :], transpose(u.(grid_model.data.x, grid_model.data.y, 60 * 60 *40)), title="Interpolated u at t=200 hours", xlabel="Longitude", ylabel="Latitude")
# p = Plots.heatmap(transpose(u.(grid_model.data.x, grid_model.data.y, 60 *60 * 200)))
display(p)
# %%

function plot_particle_collection(wave_model)
    particles = wave_model.ParticleCollection
    p = plot(layout=(3, 2), size=(1200, 1200))
    heatmap!(p, transpose(particles.on), subplot=1, title="on | iter=" * string(wave_model.clock.iteration) * " | time=" * string(round(wave_model.clock.time / 60 / 60)) * "hours")

    xi, yi = wave_model.grid.data.x, wave_model.grid.data.y
    ui = wave_model.winds.u.(xi, yi, wave_model.clock.time)
    vi = wave_model.winds.v.(xi, yi, wave_model.clock.time)
    xt_idx = 1:50:size(xi, 1)
    yt_idx = 1:50:size(yi, 2)
    xt_lbl = Int.(round.(xi[xt_idx, 1]; digits=0))
    yt_lbl = Int.(round.(yi[1, yt_idx]; digits=0))


    step = 5
    heatmap!(p, transpose(sqrt.(ui .^ 2 + vi .^ 2)), subplot=2, title="wind forcing", clims=(0, NaN), cmap=:jet)

    # quiver!(p, xi[1:step:end, 1:step:end], yi[1:step:end, 1:step:end], quiver=(ui[1:step:end, 1:step:end], vi[1:step:end, 1:step:end]), subplot=2, color=:black, alpha=0.7)

    sE = wave_model.State[:, :, 1]
    sE[wave_model.grid.data.mask.==0] .= NaN
    sE[wave_model.grid.data.mask.==2] .= NaN
    heatmap!(p, transpose(sE), subplot=3, title="State: Energy", clims=(0, NaN))
    # contour!(p, sE' , subplot=3, title="State: Energy", clims=(0, NaN))

    heatmap!(p, transpose(log.(sE)), subplot=5, title="State: log Energy", clims=(NaN, NaN))
    # contour!(p, sE' , subplot=3, title="State: Energy", clims=(0, NaN))


    sm1 = wave_model.State[:, :, 2]
    sm1[wave_model.grid.data.mask.==0] .= NaN
    sm1[wave_model.grid.data.mask.==2] .= NaN
    heatmap!(p, transpose(sm1), subplot=4, title="State: x momentum ")#, clims=(0, NaN))

    sm2 = wave_model.State[:, :, 3]
    sm2[wave_model.grid.data.mask.==0] .= NaN
    sm2[wave_model.grid.data.mask.==2] .= NaN
    heatmap!(p, transpose(sm2), subplot=6, title="State: y momentum ")
    # title = plot!(title="Plot title", grid=false, showaxis=false, bottom_margin=-50Plots.px)


    for sp in [1, 2, 3, 4, 5, 6]
        plot!(p,
            subplot=sp,
            xticks=(xt_idx, xt_lbl),
            yticks=(yt_idx, yt_lbl),
            xrotation=45
        )
    end

    display(p)
end

# plot_particle_collection(wave_simulation.model)

# %%
Revise.retry()
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q,
    propagation=true,
    input=true,
    dissipation=true,
    peak_shift=true,
    direction=true,
);

default_ODE_parameters = (r_g=ODEpars.r_g, C_α=Const_Scg.C_alpha,
    C_φ=Const_ID.c_β, C_e=Const_ID.C_e, g=9.81);#, M=M);

# define setting and standard initial conditions
WindSea = PiCLES.FetchRelations.get_initial_windsea(3, 3, 60minutes)
@show WindSea

WindSeamin = PiCLES.FetchRelations.MinimalWindsea(U10, V10, 10minutes);
@show WindSeamin
lne_local = log(WindSeamin["E"])

ODE_settings = PW.ODESettings(
    Parameters=ODEpars,
    # define mininum energy threshold
    log_energy_minimum=WindSea["lne"],#log(FetchRelations.Eⱼ(0.1, DT)),
    #maximum energy threshold
    log_energy_maximum=log(27),#log(17),  # correcsponds to Hs about 16 m
    saving_step=6000,
    timestep=DT,
    total_time=T,
    dt=1e-3,
    dtmin=1e-4,
    dtmax=20minutes)


Revise.retry()

wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=grid_model,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    #ODEinit_type=ParticleDefaults(default_windsea[1], default_windsea[2], default_windsea[3], 0.0, 0.0),
    ODEinit_type="wind_sea",#ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0),
    #ParticleDefaults2D(log(2), 0.0, 0.0, 0.0, 0.0), #"wind_sea",  # default_ODE_valuves
    periodic_boundary=false,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

# %% build Simulation
T_int = 48hours*2
wave_simulation = Simulation(wave_model, Δt=DT, stop_time=T_int)
initialize_simulation!(wave_simulation)
plot_particle_collection(wave_model)

run!(wave_simulation, cash_store=true, debug=false);

plot_particle_collection(wave_model)

# %%

#482
T_int / DT # days
time_sel = 60 * 60 * 0 # seconds
time_sel = DT * 190

p = plot(size=(1200, 200))

# wind on grid
Ni_data=30
Plots.plot!(p, grid_data.data.y[Ni_data, :], u.(grid_data.data.x[Ni_data, :], grid_data.data.y[Ni_data, :], time_sel), 
            label="u on U grid at t=" * string(time_sel) * " sec")

# wind in model 
Ni_model = 46
Plots.plot!(p, grid_model.data.y[Ni_model, :], wave_model.winds.u.(grid_model.data.x[Ni_model, :], grid_model.data.y[Ni_model, :], time_sel),
            label="u on M grid at t=" * string(time_sel) * " sec")
#Plots.plot!(p, wave_model.winds.u.(grid_model.data.x[46, :], grid_model.data.y[46, :], time_rel[end]))

itime = Int(floor(time_sel / DT))
energy = wave_simulation.store.store[itime][:,:,1]
energy_sel= energy[Ni_model, :] #/ 0.01#maximum(energy[Ni_model, :])

Plots.plot!(p, grid_model.data.y[Ni_model, :], energy_sel,
    label="E on M grid at t=" * string(time_sel) * " sec")

Plots.plot!(p, grid_model.data.y[Ni_model, :], log.(energy_sel),
label="log(E) on M grid at t=" * string(time_sel) * " sec")


#wave_simulation.store.wind_u[Ni_model, :, 1]

display(p)

# %%

Plots.heatmap(u.(grid_data.data.x[46, :], grid_data.data.y[46, :], time_sel))




# %% manual testing
for i in 1:1:300
    TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt)

    if i % 5 == 0
        plot_particle_collection(wave_simulation.model)
        # fig = PlotState_DoubleGlobeSeam(wave_simulation.model, scaled=false)
        # fig
        sleep(0.2)
    end
    wave_simulation.model.State[:, :, :] .= 0.0

end
# %%
@info wave_simulation.model.ParticleCollection[1, 1]

wave_simulation.model.ParticleCollection[20, 20].ODEIntegrator.t/60/60


wave_simulation.model.ParticleCollection[20, 20].ODEIntegrator.t / 60 / 60

# %%
times_list = vec([wave_simulation.model.ParticleCollection[i, j].ODEIntegrator.t for i in 1:size(wave_simulation.model.ParticleCollection, 1) for j in 1:size(wave_simulation.model.ParticleCollection, 2)])
times = reshape([wave_simulation.model.ParticleCollection[i, j].ODEIntegrator.t for i in 1:size(wave_simulation.model.ParticleCollection, 1) for j in 1:size(wave_simulation.model.ParticleCollection, 2)], size(wave_simulation.model.ParticleCollection))


p = Plots.histogram(times / 60 / 60, xlabel="Time (hours)", ylabel="Count", title="Particle Collection Times", bins=100)
display(p)

Plots.heatmap(times / 60 / 60, xlabel="Time (hours)", ylabel="Count", title="Particle Collection Times")

wave_simulation.model.clock.time/60/60
# %%
# TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt)

#%%
Revise.retry()
using GLMakie
using PiCLES.Operators.core_2D: GetGroupVelocity

using PiCLES.Plotting.movie: init_movie_2D_simple

# or, alternatively, make movie
save_path = "plots/hurricanes/TC_Lee/"
plot_name = "moving_box_test_run"
# N = 100
# N = Int(time_rel[end] / DT / 2)
N = Ntime

fig, n = init_movie_2D_simple(wave_simulation; resolution=(1350, 1200), name_string=plot_name, aspect=1)

record(fig, save_path * plot_name * ".gif", 1:N, framerate=10) do i
    @info "Plotting frame $i of $N..."
    @info wave_simulation.model.clock

    # wave_simulation.model.State .= 0.0
    # TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt)
    # wave_simulation.model.State .= 0.0
    # TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt)
    # wave_simulation.model.State .= 0.0
    TimeSteppers.movie_time_step!(wave_simulation.model, wave_simulation.Δt)
    # Set the current index or iteration counter to 1, likely initializing or resetting
    # a variable that tracks the frame number, time step, or current position in a sequence
    n[] = 1
end

wave_simulation.model.clock.time / (60 * 60 * 24) # hours


Ntime * DT / (60 * 60 * 24) # hours

# %%