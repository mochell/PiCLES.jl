# %%
ENV["JULIA_INCREMENTAL_COMPILE"] = true
using Pkg
Pkg.activate("PiCLES/")

import Plots as plt
using Setfield, IfElse

using PiCLES.ParticleSystems: particle_waves_v5 as PW

import PiCLES: FetchRelations
using PiCLES.Simulations
using PiCLES.Operators.TimeSteppers: time_step!, movie_time_step!
using PiCLES.Models.WaveGrowthModels2D

using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Units

using PiCLES.Operators: TimeSteppers

using PiCLES.Operators.core_2D: ParticleDefaults
using PiCLES.Grids

using Revise
using BenchmarkTools

using PiCLES.Plotting: PlotState_DoubleGlobe, PlotState_SingleGlobe, PlotState_DoubleGlobeSeam, OrthographicTwoMaps, OrthographicTwoMapsSeam
using Plots

# %%

Revise.retry()
plot_path_base = "plots/tests/T03_PIC_Sphere/"
mkpath(plot_path_base)
pwd()
# % Parameters
U10, V10 = -20.0, 1.0
DT = 20minutes

ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)

u_std = 20 * 1
v_std = 20 * 1
u_center = 90
v_center = 40
u(x, y, t) = U10 * exp(-(x - u_center)^2 / u_std^2) * exp(-(y - v_center)^2 / v_std^2) #* abs(cos(t * 1.2 / (1 * 60 * 60 * 2π)))
v(x, y, t) = V10 * exp(-(x - u_center)^2 / u_std^2) * exp(-(y - v_center)^2 / v_std^2) #* abs(cos(t * 1.2 / (1 * 60 * 60 * 2π)) )

# u(x, y, t) = U10 + x * 0.0 + y * 0.0 + t * 0.0
# v(x, y, t) = (V10 + x * 0.0 + y * 0.0) .* cos(t * 5 / (1 * 60 * 60 * 2π))
winds = (u=u, v=v)

Revise.retry()

gridd = Grids.SphericalGrid.TwoDSphericalGridMesh(0.0, 180.0, 91, 0, 80.0, 61; periodic_boundary=(true, false))

gridd.stats

gridd.data[end, 5]


# %%

# using CairoMakie

# fig = Figure(size=(900, 500), fontsize=22)
# OrthographicTwoMapsSeam(fig, gridd.data.x, gridd.data.y, gridd.data.mask)
# # heatmap(gridd.data.angle_dx)
# fig

# %%


Plots.heatmap(gridd.data.x[:, 1], gridd.data.y[1, :], transpose(gridd.data.mask))

# # % Make fake mask
mask = ones(Bool, size(gridd.data.x)) # 1 is ocean, 0 is land (?)
mask[45:50, 45:61] .= 0

# #mask  = .!mask # to make one active block
gridstats_mask = Grids.SphericalGrid.TwoDSphericalGridMesh(gridd.stats; mask=mask)
gridd = Grids.SphericalGrid.TwoDSphericalGridMesh(gridstats_mask, gridd.stats, gridd.ProjetionKernel, gridd.PropagationCorrection)

Plots.heatmap(transpose(v.(gridd.data.x, gridd.data.y, 120)))

# %%

# function plot_particle_collection(wave_model)
#     particles = wave_model.ParticleCollection
#     p = plot(layout=(3, 2), size=(1200, 1000))
#     heatmap!(p, transpose(particles.on), subplot=1, title="on | iter=" * string(wave_model.clock.iteration))
#     heatmap!(p, transpose(particles.boundary), subplot=2, title="boundary")
#     heatmap!(p, transpose(wave_model.State[:, :, 1]), subplot=3, title="State: Energy", clims=(0, NaN))
#     heatmap!(p, transpose(wave_model.State[:, :, 2]), subplot=4, title="State: x momentum ", clims=(0, NaN))
#     heatmap!(p, transpose(wave_model.State[:, :, 3]), subplot=6, title="State: y momentum ")
#     # title = plot!(title="Plot title", grid=false, showaxis=false, bottom_margin=-50Plots.px)
#     display(p)
# end


function plot_particle_collection(wave_model)
    particles = wave_model.ParticleCollection
    p = plot(layout=(3, 2), size=(1200, 1000))
    heatmap!(p, transpose(particles.on), subplot=1, title="on | iter=" * string(wave_model.clock.iteration) * " | time=" * string(wave_model.clock.time))
    heatmap!(p, transpose(particles.boundary), subplot=2, title="boundary")

    sE = wave_model.State[:, :, 1]
    sE[wave_model.grid.data.mask.==0] .= NaN
    sE[wave_model.grid.data.mask.==2] .= NaN
    heatmap!(p, transpose(sE), subplot=3, title="State: Energy", clims=(0, NaN))

    sm1 = wave_model.State[:, :, 2]
    sm1[wave_model.grid.data.mask.==0] .= NaN
    sm1[wave_model.grid.data.mask.==2] .= NaN
    heatmap!(p, transpose(sm1), subplot=4, title="State: x momentum ", clims=(0, NaN))

    sm2 = wave_model.State[:, :, 3]
    sm2[wave_model.grid.data.mask.==0] .= NaN
    sm2[wave_model.grid.data.mask.==2] .= NaN
    heatmap!(p, transpose(sm2), subplot=6, title="State: y momentum ")
    # title = plot!(title="Plot title", grid=false, showaxis=false, bottom_margin=-50Plots.px)
    display(p)
end


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


Revise.retry()

# default_windsea = FetchRelations.get_initial_windsea(0.0, -30.0, DT*10, particle_state=true)
#default_particle = ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0)
wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=gridd,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,  # ODE_settings
    #ODEinit_type=ParticleDefaults(default_windsea[1], default_windsea[2], default_windsea[3], 0.0, 0.0),
    ODEinit_type=ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0),
    #ParticleDefaults2D(log(2), 0.0, 0.0, 0.0, 0.0), #"wind_sea",  # default_ODE_valuves
    periodic_boundary=false,
    boundary_type="same",
    #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

#wave_model.ODEsettings.wind_min_squared = 0.0
#wave_model.minimal_state = 2 * wave_model.minimal_state

# ### build Simulation

wave_simulation = Simulation(wave_model, Δt=120minutes, stop_time=8days)#1hours)
initialize_simulation!(wave_simulation)
plot_particle_collection(wave_model)

# wave_simulation.model.ParticleCollection
# wave_simulation.model.ParticleCollection.position_ij
# wave_simulation.model.ParticleCollection[4, 5]
# heatmap(wave_simulation.model.ParticleCollection.on)

# % timstepper test
# plot_particle_collection(wave_model)

5days/ 30minutes

for i in 1:1:100
    TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt)

    if i % 4 == 0
        plot_particle_collection(wave_simulation.model)
        # fig = PlotState_DoubleGlobeSeam(wave_simulation.model, scaled=false)
        # fig
        sleep(0.2)
    end
    wave_simulation.model.State[:, :, :] .= 0.0

end


fig = PlotState_DoubleGlobeSeam(wave_simulation.model, scaled=false)
#Makie.save(joinpath(plot_path_base, subtitle * "_shift_sphere.png"), fig)




# %%
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q,
    propagation=true,
    input=true,
    dissipation=true,
    peak_shift=true,
    direction=true,
);

# default_windsea = FetchRelations.get_initial_windsea(U10*0.1, V10*0.1, DT, particle_state=true)

using Makie
Revise.retry()
for (u10, v10, name) in [
    (5.0, 0.0, "zonal"),
    (-5.0, 0.0, "zonal_neg"),
    (0.0, 5.0, "meridional"),
    (0.0, -5.0, "meridional_neg"),
    (5.0, 5.0, "diagonal"),
    (-5.0, - 5.0, "diagonal_neg"),
]
    @info u10, v10, name
    default_windsea = FetchRelations.get_initial_windsea(u10, v10, DT, particle_state=true)
    default_particle = ParticleDefaults(lne_local, cg_u_local, cg_v_local, 0.0, 0.0)
    wave_model = WaveGrowthModels2D.WaveGrowth2D(; grid=gridd,
        winds=winds,
        ODEsys=particle_system,
        ODEsets=ODE_settings,  # ODE_settings
        ODEinit_type=ParticleDefaults(default_windsea[1], default_windsea[2], default_windsea[3], 0.0, 0.0),
        #ParticleDefaults2D(log(2), 0.0, 0.0, 0.0, 0.0), #"wind_sea",  # default_ODE_valuves
        periodic_boundary=false,
        boundary_type="same",
        #minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
        movie=true)

    subtitle = "TripolarGrid_aqua_test_u$(u10)_v$(v10)_"

    wave_simulation = Simulation(wave_model, Δt=40minutes, stop_time=12hours*6)#1hours)
    initialize_simulation!(wave_simulation)

    plot_particle_collection(wave_model)
    savefig(joinpath(plot_path_base, subtitle * "_initialvalues_plane.png"))

    # fig = PlotState_DoubleGlobeSeam(wave_simulation.model)
    # Makie.save(joinpath(plot_path_base, subtitle * "_initialvalues_sphere.png"), fig)

    run!(wave_simulation, cash_store=true, debug=false)

    plot_particle_collection(wave_simulation.model)
    savefig(joinpath(plot_path_base, subtitle * "_shift_plane.png"))

    fig = PlotState_DoubleGlobeSeam(wave_simulation.model, scaled=false)
    Makie.save(joinpath(plot_path_base, subtitle * "_shift_sphere.png"), fig)
end


# %%

 fig = Figure(size=(900, 500), fontsize=22)
OrthographicTwoMapsSeam(fig,gridd.data.x, gridd.data.y, gridd.data.angle_dx) 
# heatmap(gridd.data.angle_dx)
fig
Makie.save(joinpath(plot_path_base, "grid" * "_angle_definition.png"), fig)


gridd.data.angle_dx

# %%