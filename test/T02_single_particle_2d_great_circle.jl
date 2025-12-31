# %%
using Pkg
Pkg.activate("PiCLES/")

using DifferentialEquations
using Plots
using Setfield
using IfElse

using PiCLES.ParticleSystems: particle_waves_v5 as PW
using PiCLES.Utils: Init_Standard

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: InitParticleInstance
using Oceananigans.Units

using PiCLES.Grids.TripolarGridMOM6: TripolarGridMOM6

using PiCLES.Grids: SphericalPropagationCorrection
using Statistics

plot_path_base = "plots/tests/T04_2D_single_particle/"
mkpath(plot_path_base)

# %%

using CairoMakie, GeoMakie

#using Makie
using PiCLES.Plotting: PlotState_DoubleGlobe, PlotState_SingleGlobe, PlotState_DoubleGlobeSeam, OrthographicTwoMaps, OrthographicTwoMapsSeam

using Revise
Revise.retry()

u(x::Number, y::Number, t::Number) = 0.0
v(x::Number, y::Number, t::Number) = 0.0
Const_ID = PW.get_I_D_constant()
ParticleState, default_ODE_parameters, WindSeamin, Const_ID = Init_Standard(u(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), 5hours)
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)
# %%

R = 6.3710E+6

deg_lat = -89.9:0.1:89.9
Gg = SphericalPropagationCorrection.(deg_lat)
G_test = [Gg_i(10) for Gg_i in Gg]
Plots.plot(deg_lat, G_test)



# define Particle Position
# %%
using PiCLES.Operators.core_2D: ParticleDefaults

function M_geocoords_deg_func(y_lat)
    return [
        1/(R*cos(y_lat * pi / 180)*pi/180) 0;
        0 1/(R*pi/180)
    ]
end


function init_propagation_particle(x_lon, y_lat, u10, v10, DT; M=[1 0; 0 1])
    # proforma intialization
    u(x::Number, y::Number, t::Number) = 0.0#(10.0 * cos(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
    v(x::Number, y::Number, t::Number) = 0.0#-(10.0 * sin(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0

    ParticleState = ParticleDefaults(FetchRelations.get_initial_windsea(u10, v10, DT, particle_state=true))

    particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q,
        propagation=true,
        input=false,
        dissipation=false,
        peak_shift=false,
        direction=false,
    );

    used_ODE_params = (default_ODE_parameters..., x=x_lon, y=y_lat, PC=SphericalPropagationCorrection(y_lat), M=M)

    # test execution
    ODE_settings = PW.ODESettings(
        Parameters=used_ODE_params,
        # define mininum energy threshold
        log_energy_minimum=log(WindSeamin["E"]),
        #maximum energy threshold
        log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
        saving_step=10minutes,
        timestep=DT,
        total_time=T = 30days,
        save_everystep=false,
        maxiters=1e4,
        adaptive=true,
        dt=10,#60*10, 
        dtmin=1,#60*5, 
        force_dtmin=true,)

    PI = InitParticleInstance(particle_system, ParticleState, ODE_settings, (0,0), (x_lon, y_lat), false, true)

    return PI
end

lon0, lat0 = 10.0, 50.0
PI = init_propagation_particle(10, 50, 0.0, 20.0, 5hours; M=M_geocoords_deg_func(lat0))
# %%

# propagation is in meters zonal and meridinal, there is no projection
Revise.retry()

function time_step_local!(PI, DT)
    "take 1 step over DT"

    # x_delta = M_geocoords_deg * PI.ODEIntegrator.u[4:5]
    # x_lon = PI.position_xy[1] + x_delta[1]
    # y_lat = PI.position_xy[2] + x_delta[2]

    # update particle position on the globe
    x_lon = PI.position_xy[1] + PI.ODEIntegrator.u[4] #/ (110e3 * cos(PI.position_xy[2]))
    y_lat = PI.position_xy[2] + PI.ODEIntegrator.u[5] #/ 110e3

    # reset particle local position
    PI.ODEIntegrator.u[4:5] .= 0.0
    M_i = M_geocoords_deg_func(y_lat)
    used_ODE_params = (default_ODE_parameters..., x=x_lon, y=y_lat, PC=SphericalPropagationCorrection(y_lat), M=M_i)

    # test execution
    ODE_settings = PW.ODESettings(
        Parameters=used_ODE_params,
        # define mininum energy threshold
        log_energy_minimum=log(WindSeamin["E"]),
        #maximum energy threshold
        log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
        saving_step=10minutes,
        timestep=DT,
        total_time=T = 30days,
        save_everystep=false,
        maxiters=1e4,
        adaptive=true,
        dt=10,#60*10, 
        dtmin=1,#60*5, 
        force_dtmin=true,)

    ParticleState = ParticleDefaults(PI.ODEIntegrator.u)
    PI = InitParticleInstance(particle_system, ParticleState, ODE_settings, (0,0), (x_lon, y_lat), false, true)

    #@info "proposed dt", get_proposed_dt(PI.ODEIntegrator) / 60
    step!(PI.ODEIntegrator, DT, true)

    return PI
end


# %% Plot trajectory for different directions
# initial position on the sphere
x_lon0, y_lat0 = 0, 45.0

fig = Figure(size=(1200, 500), fontsize=16)
proj_string = "+proj=ortho +lon_0=40 +lat_0=" * string(y_lat0-20)

ax_left = GeoAxis(fig[1, 1]; dest=proj_string, xticks=-180:30:180, yticks=-90:30:60)
ax_right = Axis(fig[1, 2:3]; title="Ray tracing on a sphere with spherical propagation coordinates", ylabel="lat", xlabel="lon")


for dir_alpha in -80:20:80
    @info dir_alpha
    u10, v10 = 20.0 * cos(dir_alpha * pi / 180), 20.0 * sin(dir_alpha * pi / 180)
    DT_init = 10hours
    PI = init_propagation_particle(x_lon0, y_lat0, u10, v10, DT_init; M=M_geocoords_deg_func(y_lat0))

    x_list = []
    y_list = []
    cg_x_list = []
    cg_y_list = []
    T = 947.64hours + 2hours
    t = 0
    DT = 2hours
    while t < T
        PI = time_step_local!(PI, DT)
        # @info PI.ODEIntegrator.u
        push!(x_list, PI.position_xy[1])
        push!(y_list, PI.position_xy[2])
        push!(cg_x_list, PI.ODEIntegrator.u[2])
        push!(cg_y_list, PI.ODEIntegrator.u[3])
        t += DT
    end

    # sphere plot
    sp1 = GeoMakie.scatter!(ax_left, x_list, y_list, cg_y_list, color="black")
    # flat plot
    lines!(ax_right, x_list, y_list, color="black", linewidth=2)

    speed = mean(sqrt.(cg_x_list .^ 2 + cg_y_list .^ 2))
    @info "hour to travel around the globe:" (2 * pi * R / speed) / (60 * 60)

end
fig

savefig(joinpath(plot_path_base, "T02_2D_single_particle_raytracing_on_a_spherical.png"))

# %% ------------------- Repeat with local coordinates -------------------


# standard local cartesian coordinates
M_cart_local= [1 0; 0 1]

lon0, lat0 = 10.0, 50.0
PI = init_propagation_particle(10, 50, 0.0, 20.0, 5hours; M=M_cart_local)

function time_step_local2!(PI, DT)
    "take 1 step over DT"

    # update particle position on the globe
    x_delta_deg = M_geocoords_deg_func(PI.position_xy[2]) * PI.ODEIntegrator.u[4:5]
    x_lon = PI.position_xy[1] + x_delta_deg[1]
    y_lat = PI.position_xy[2] + x_delta_deg[2]

    # reset particle local position
    PI.ODEIntegrator.u[4:5] .= 0.0

    used_ODE_params = (default_ODE_parameters..., x=x_lon, y=y_lat, PC=SphericalPropagationCorrection(y_lat), M=M_cart_local)

    # test execution
    ODE_settings = PW.ODESettings(
        Parameters=used_ODE_params,
        # define mininum energy threshold
        log_energy_minimum=log(WindSeamin["E"]),
        #maximum energy threshold
        log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
        saving_step=10minutes,
        timestep=DT,
        total_time=T = 30days,
        save_everystep=false,
        maxiters=1e4,
        adaptive=true,
        dt=10,#60*10, 
        dtmin=1,#60*5, 
        force_dtmin=true,)

    ParticleState = ParticleDefaults(PI.ODEIntegrator.u)
    PI = InitParticleInstance(particle_system, ParticleState, ODE_settings, (0,0), (x_lon, y_lat), false, true)

    #@info "proposed dt", get_proposed_dt(PI.ODEIntegrator) / 60
    step!(PI.ODEIntegrator, DT, true)

    return PI
end


# %% Plot trajectory for different directions
# initial position on the sphere
x_lon0, y_lat0 = 0, 45.0

fig = Figure(size=(1200, 500), fontsize=16)
proj_string = "+proj=ortho +lon_0=40 +lat_0=" * string(y_lat0 - 20)

ax_left = GeoAxis(fig[1, 1]; dest=proj_string, xticks=-180:30:180, yticks=-90:30:60)
ax_right = Axis(fig[1, 2:3]; title="Ray tracing on a sphere with local propagation coordinates", ylabel="lat", xlabel="lon")

for dir_alpha in -80:20:80
# for dir_alpha in 0:20:40
    @info dir_alpha
    u10, v10 = 20.0 * cos(dir_alpha * pi / 180), 20.0 * sin(dir_alpha * pi / 180)
    DT_init = 10hours
    PI = init_propagation_particle(x_lon0, y_lat0, u10, v10, DT_init; M=M_cart_local)

    x_list = []
    y_list = []
    cg_x_list = []
    cg_y_list = []
    T = 947.06hours - 0hours
    t = 0
    DT = 2hours
    while t < T
        PI = time_step_local2!(PI, DT)
        # @info PI.ODEIntegrator.u
        push!(x_list, PI.position_xy[1])
        push!(y_list, PI.position_xy[2])
        push!(cg_x_list, PI.ODEIntegrator.u[2])
        push!(cg_y_list, PI.ODEIntegrator.u[3])
        t += DT
    end

    # sphere plot
    sp1 = GeoMakie.scatter!(ax_left, x_list, y_list, cg_y_list, color="black")
    # flat plot
    lines!(ax_right, x_list, y_list, color="black", linewidth=2)

    speed = mean(sqrt.(cg_x_list .^ 2 + cg_y_list .^ 2))
    @info "hour to travel around the globe:" (2 * pi * R / speed) / (60 * 60)

end
fig
savefig(joinpath(plot_path_base, "T02_2D_single_particle_raytracing_on_a_spherical_local.png"))



# %%


# %%
# % test single integration step
# PI = init_propagation_particle(x_lon0, y_lat0, u10, v10, DT_init)

# PI.position_xy, PI.ODEIntegrator.u
# x_lon, y_lat = PI.position_xy;
# PI = time_step_local!(PI, 40hours,  u10, v10);
# PI.position_xy, PI.ODEIntegrator.u

# PID = ParticleTools.ParticleToDataframe(PI);

# gr(display_type=:inline)
# # plit each row in PID and a figure
# tsub = range(start=1, stop=length(PID[:, 1]), step=10)
# p3 = Plots.plot( x_lon  .+  PID[tsub, 5], y_lat .+  PID[tsub, 6], marker=3, title="position on Sphere ", ylabel="lat", xlabel= "lon", label="1 integration") #|> display

# # local coordinate version
# # p3 = Plots.plot( x_lon  .+  PID[tsub, 5] / (110e3 * cos(y_lat)), y_lat .+  PID[tsub, 6] / 110e3, marker=3, title="position on Sphere ", ylabel="lat", xlabel= "lon", label="v4") #|> display

# Plots.plot!(p3,  [x_lon]  , [y_lat] , marker=10, title="lon ", ylabel="lat", label="origin") #|> display

# Plots.plot!(x_list, y_list, title="position on Sphere ", ylabel="lat", xlabel="lon", label="piecewise integration", width=3, marker=3) #|> display

# p3
# subtitle = "u$(U10)_v$(V10)_reset_to_windsea_dt$(DT)"
# subtitle = "u10_v10_reset_to_windsea_dt$(DT)"
# savefig(joinpath(plot_path_base, subtitle*"_continous_foreward.png"))

