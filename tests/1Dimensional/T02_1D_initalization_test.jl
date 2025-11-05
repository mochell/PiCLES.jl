# %%

using Pkg
Pkg.activate("PiCLES/")

using Plots

using DocStringExtensions
using DifferentialEquations

using PiCLES.ParticleMesh: OneDGrid, OneDGridNotes
using PiCLES.Operators.core_1D: ParticleDefaults
using PiCLES: Simulation, WindEmulator, WaveGrowthModels1D, FetchRelations
using PiCLES.Simulations

using PiCLES.ParticleSystems: particle_waves_v5 as PW

using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Units

using Statistics
using HDF5, JLD2
using Measures

using IfElse
# %%


function convert_state_store_to_array(store::Vector{Any}, grid; time_iteration_interval=1, t_start=0)
    store_data = cat(store..., dims=3)
    store_waves_data = permutedims(store_data, (3, 1, 2))
    wave_x = OneDGridNotes(grid).x
    wave_time = collect(range(0, wave_simulation.stop_time + wave_simulation.Δt, step=wave_simulation.Δt * time_iteration_interval))
    wave_time = (wave_time .- t_start)
    return (data=store_waves_data, x=wave_x, time=wave_time)
end





# %%


Revise.retry()

# Default values
save_path_base = "data/1D_gaussian/"
plot_path_base = "plots/tests/T02_1D_initialization/"
mkpath(plot_path_base)


# parametric wind forcing
x_scale = 1000e3 / 1.5
t_scale = 3.6days / 1
U10 = 15#m/s

T = t_scale * 1.5 #5

dt = 10minutes 
dx = 5e3

U10 * dt / dx

xend = x_scale * 3.5

# just for plotting
xx = collect(0:10e3:xend);
tt = collect(0:30minutes:T);
xx_grid, tt_grid = [x for x in xx, t in tt], [t for x in xx, t in tt];

#u(x, t) = IfElse.ifelse.((x_scale / 2 .< x) & (x .< x_scale * 3) & (t .< t_scale * 2), U10, 0.0) + t * 0

t_start = 12hours 
u(x, t) = IfElse.ifelse.((t .> t_start) & (x .< x_scale * 3) , x / x_scale + (t - t_start) / t_scale, 0.0)

wind_map = u.(xx_grid, tt_grid);

heatmap(xx / 1e3, tt / 1day, transpose(wind_map), label="t=0")


# %

# change output interval by setting output_step_interval = Integer(60minutes / dt)

function init_test(u, grid1d, dt, T; wind_min_squared=sqrt(2), output_step_interval=1)

    ODEpars, Const_ID, _ = PW.ODEParameters(r_g=0.85)

    particle_system = PW.particle_equations(u, γ=Const_ID.γ, q=Const_ID.q,
        propagation=true,
        input=true,
        dissipation=true,
        peak_shift=true,
    );

    WindSeamin = FetchRelations.MinimalWindsea(U10, dt)

    ODE_settings = PW.ODESettings(
        Parameters=ODEpars,
        # define mininum energy threshold
        log_energy_minimum=log(WindSeamin["E"]),
        #maximum energy threshold
        log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
        wind_min_squared=wind_min_squared, 
        saving_step=T,
        timestep=dt,
        total_time=T,
        adaptive=true,
        dt=1e-3, #60*10, 
        dtmin=1e-12, #60*5,
        # abstol=1e-10, 
        force_dtmin=true,
    )


    wave_model = WaveGrowthModels1D.WaveGrowth1D(; grid=grid1d,
        winds=u,
        ODEsys=particle_system,
        ODEvars=nothing,
        ODEsets=ODE_settings, 
        ODEinit_type="wind_sea",  # default_ODE_parameters
        periodic_boundary=false,
        boundary_type="same",
        output_step_interval=output_step_interval,
    )

    wave_simulation = Simulation(wave_model, Δt=dt, stop_time=T)
    return wave_simulation
end

function plot_wave_simulation(output, wind_map, xx, tt, dx, dt, WindSeamin; cmap=cgrad(:ocean, rev=true))
    model_time = output.time / dt
    model_x = output.x / dx
    store_Hs = 4 * sqrt.(output.data[:, :, 1])
    store_waves_mx = output.data[:, :, 2] .+ WindSeamin["m"]
    cg = output.data[:, :, 1] ./ (store_waves_mx * 2)
    cg = clamp.(cg, 0, 40)

    p = plot(layout=(3, 1), size=(800, 1000), left_margin=[10mm 0mm])

    contourf!(xx ./ dx, tt / dt, transpose(wind_map), subplot=1, color=cmap, lw=0, levels=10, ylabel="time (days)", title="Wind Speed (U10)")
    contour!(xx ./ dx, tt / dt, transpose(wind_map), subplot=1, linecolor="red", lw=1, levels=[sqrt(2)], ylabel="time (days)")

    #heatmap!(p, model_x, model_time, store_Hs, subplot=2, lw=0, title="Significant Wave Height (Hs)", color=cmap, ylabel="time (days)", clims=(0, 3))
    contour!(p, model_x, model_time, store_Hs, subplot=2, lw=0.5, levels=[0.01, 0.1, 0.5, 1, 2, 3, 4], linecolor="black", ylabel="time (days)")

    contour!(xx ./ dx, tt / dt, transpose(wind_map), subplot=2, linecolor="red", lw=1, levels=[sqrt(2)], ylabel="time (days)")

    contourf!(p, model_x, model_time, cg, subplot=3, lw=0, title="Group Velocity (m/s)", color=cmap, xlabel="x (km)", ylabel="time (days)", clims=(0, 6))
    contour!(p, model_x, model_time, cg, subplot=3, lw=0.5, levels=[0.01, 0.1, 0.5, 1, 2, 3, 4], linecolor="black", ylabel="time (days)")

    contour!(xx ./ dx, tt / dt, transpose(wind_map), subplot=3, linecolor="red", lw=1, levels=[sqrt(2)], ylabel="time (days)")

    return p
end




dts = [10minutes, 20minutes, 60minutes]
dxs = [2.5e3, 5e3, 10e3]
wind_min_squared_values = [0.0001, 0.001, 0.01, 0.1, 1,  sqrt(2)]

# dt, dx, wind_min_squared = 10minutes, 2.5e3, 0.0001

for dt in dts
    for dx in dxs
        for wind_min_squared in wind_min_squared_values

                grid1d = OneDGrid(0, xend, Integer(round(xend / dx)))
                wave_simulation = init_test(u, grid1d, dt, T; wind_min_squared=wind_min_squared, output_step_interval=1)
                initialize_simulation!(wave_simulation)

                run!(wave_simulation, store=false, cash_store=true, debug=false);

                output = convert_state_store_to_array(wave_simulation.store.store, wave_simulation.model.grid;
                    time_iteration_interval=wave_simulation.model.output_step_interval,
                    t_start=0)

                p = plot_wave_simulation(output, wind_map, xx, tt, 1e3, 1day, WindSeamin; cmap=cgrad(:ocean, rev=true))

                savefig(p, joinpath(plot_path_base, "wind_init_gradient_test=$(round(wind_min_squared, sigdigits=3))_dt=$(Integer(dt))_dx=$(Integer(dx)).png"))
                display(p)

        end
    end
end


#"wind_init_gradient_test=$(wind_min_squared)_dt=$(dt)_dx=$(dx).png"