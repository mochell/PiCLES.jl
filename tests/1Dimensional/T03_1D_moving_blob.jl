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

using Base.Threads
# %%

function convert_state_store_to_array(store::Vector{Any}, wave_simulation; time_iteration_interval=1, t_start=0)
    store_data = cat(store..., dims=3)
    store_waves_data = permutedims(store_data, (3, 1, 2))
    wave_x = OneDGridNotes(wave_simulation.model.grid).x
    wave_time = collect(range(0, wave_simulation.stop_time + wave_simulation.Δt, step=wave_simulation.Δt * time_iteration_interval))
    wave_time = (wave_time .- t_start)
    return (data=store_waves_data, x=wave_x, time=wave_time)
end


# %%
Revise.retry()

# Default values
save_path_base = "data/1D_gaussian/"
plot_path_base = "plots/tests/T02_1D_moving_blob/"
mkpath(plot_path_base)
#savefig(joinpath(plot_path_base, "PW3_vs_PW4.png"))


# parametric wind forcing
x_scale = 1000e3/2
t_scale = 3.6days/2
U10, V = 15, 4#m/s

T = t_scale *4 #5

dt = 20minutes 
dx = 20e3 

xend = x_scale *  (3 + V * 2)
# just for plotting
function make_wind_map(xend, tend, U10, V, T, x_scale, t_scale)
    xx = collect(0:10e3:xend);
    tt = collect(0:30minutes:tend);
    xx_grid, tt_grid = [x for x in xx, t in tt], [t for x in xx, t in tt];

    wind_map = WindEmulator.slopped_blob.(xx_grid, tt_grid, U10, V, T, x_scale, t_scale; x0=x_scale/2);
    return (xx=xx, tt=tt, wind_map=wind_map)
end




wind_map1 = make_wind_map(xend, T, U10, V, T, x_scale, t_scale)
maximum(wind_map1.wind_map)

heatmap(wind_map1.xx / 1e3, wind_map1.tt / 1day, transpose(wind_map1.wind_map), label="t=0")

transpose(wind_map1.tt)
# %
ODEpars, Const_id, Const_Scg = PW.ODEParameters(r_g=0.85)
grid1d = OneDGrid(0, xend, Integer(round(xend / dx)))


# create id and save name
@info "Init Forcing Field\n"
# create wind test function
u(x, t) = WindEmulator.slopped_blob.(x, t, U10, V, T, x_scale, t_scale; x0=x_scale)
# u(x, t) = IfElse.ifelse.((x_scale / 2 .< x) & (x .< x_scale * 3) & (t .< t_scale * 2), U10, 0.0) + t * 0


function init_test(u, grid1d, dt, T; wind_min_squared=2, output_step_interval=1)

    ODEpars, Const_id, _ = PW.ODEParameters(r_g=0.85)

    particle_system = PW.particle_equations(u, γ=Const_id.γ, q=Const_id.q,
        propagation=true,
        input=true,
        dissipation=true,
        peak_shift=true,
    )

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


function make_string(astring, normalizer; unit=nothing, digits=0)
    if unit == nothing
        unit = ""
    end
    return "$(round(astring / normalizer, digits= digits) ) $(unit)"
end

# %%

function plot_wave_simulation(output, wind_map, xx, tt, DX, DT, WindSeamin; cmap=cgrad(:ocean, rev=true), wind_min=nothing, name=nothing)

    dx, dt = 1e3, 1day
    model_time = output.time / dt
    model_x = output.x / dx
    store_Hs = 4 * sqrt.(output.data[:, :, 1])
    store_waves_mx = output.data[:, :, 2] .+ WindSeamin["m"]
    cg = output.data[:, :, 1] ./ (store_waves_mx * 2)
    cg = clamp.(cg, 0, 40)
    p = plot(layout=(3, 1), size=(800, 1200), left_margin=[10mm 0mm])

    if name != nothing
       name ="Wind Min: $(wind_min)"
    end
    contourf!(xx ./ dx, tt / dt, transpose(wind_map), subplot=1, color=cmap, lw=0, levels=10, ylabel="time (days)", 
                    title=name * 
                    " dx: $(make_string(DX, 1e3; unit="km")), dt: $(make_string(DT, 1minutes; unit="min", digits=3)),  
                    U10: $(U10), V: $(V) \n Wind Speed (U10)")
    #; 
    max_hs= " (max = $(round(maximum(store_Hs), sigdigits=3)) m)"
    heatmap!(p, model_x, model_time, store_Hs, subplot=2, lw=0, title="Significant Wave Height (Hs) " * max_hs, color=cmap, ylabel="time (days)", clims=(0, 10))
    
    contour!(p, model_x, model_time, store_Hs, subplot=2, lw=0.5, levels=[0.01, 0.1, 0.5, 1, 2, 3, 4], linecolor="black", ylabel="time (days)")

    contourf!(p, model_x, model_time, cg, subplot=3, lw=0, title="Group Velocity (m/s)", color=cmap, xlabel="x (km)", ylabel="time (days)", clims=(0, 10))
    max_cg = " (max = $(round(maximum(cg), sigdigits=3)))"
    heatmap!(p, model_x, model_time, cg, subplot=3, lw=0, title="Group Velocity (m/s) " * max_cg, color=cmap, xlabel="x (km)", ylabel="time (days)", clims=(0, 10))
    contour!(p, model_x, model_time, cg, subplot=3, lw=0.5, levels=[0.01, 0.1, 0.5, 1, 2, 3, 4], linecolor="black", ylabel="time (days)")

    if wind_min != nothing
        for splot in 1:3
            contour!(xx ./ dx, tt / dt, transpose(wind_map), subplot=splot, linecolor="red", lw=1, levels=[wind_min], ylabel="time (days)")
        end
    end

    max_indices = argmax(wind_map)
    for splot in 1:3
        vline!(p, [xx[max_indices[1]] / dx], subplot=splot, color=:orange, lw=2, label=nothing)
        hline!(p, [tt[max_indices[2]] / dt], subplot=splot, color=:orange, lw=2, label=nothing)
        plot!(p, xx / dx, ((xx .- xx[max_indices[1]]) / V .+ tt[max_indices[2]]) / dt, color=:white, lw=1.5, label=nothing, subplot=splot)
        ylims!(p, 0, maximum(tt / dt), subplot=splot)
    end


    return p
end


# %%
wind_min_squared = 1
dt = 20minutes
dx = 20e3

xend = x_scale * (3 + V * 1)

for wind_min_squared in [0.001, 0.1, 0.01, 1, 2, 5]
    for dt in [10minutes, 20minutes, 60minutes]
        @threads for dx in [5e3, 10e3, 30e3]
            grid1d = OneDGrid(0, xend, Integer(round(xend / dx)))
            wave_simulation = init_test(u, grid1d, dt, T; wind_min_squared=wind_min_squared, output_step_interval=Integer(60minutes / dt))
            initialize_simulation!(wave_simulation)

            run!(wave_simulation, store=false, cash_store=true, debug=false);

            output = convert_state_store_to_array(wave_simulation.store.store, wave_simulation;
                time_iteration_interval=wave_simulation.model.output_step_interval,
                t_start=0.0)

            WindSeamin = FetchRelations.MinimalWindsea(U10, dt)
            p = plot_wave_simulation(output, wind_map, xx, tt, grid1d.dx, wave_simulation.Δt, WindSeamin; 
                                    cmap=cgrad(:ocean, rev=true), wind_min=sqrt(wind_min_squared))

            savefig(p, joinpath(plot_path_base, "moving_blob_test_dt=$(Integer(dt))_dx=$(Integer(dx))_windmin_sqt=$(round(wind_min_squared, sigdigits=3)).png"))
            display(p)
        end
    end
end

# %% Make more specific tests with chaning wind forcing

plot_path_base = "plots/tests/T02_1D_moving_blob/cases/"
mkpath(plot_path_base)

function make_wind_map(acase)

    x_scale, t_scale, U10, V= acase.x_scale, acase.t_scale, acase.U10, acase.V
    tend = t_scale * 4 #5
    xend = x_scale * (3 + V * 2)

    xx = collect(0:10e3:xend)
    tt = collect(0:30minutes:tend)
    xx_grid, tt_grid = [x for x in xx, t in tt], [t for x in xx, t in tt]

    return make_wind_map(xend, T, U10, V, T, x_scale, t_scale)
end

function save_wind_data(save_path, wind_map1)
    h5write(joinpath(save_path, "wind_data.h5"), "x", wind_map1.xx)
    h5write(joinpath(save_path, "wind_data.h5"), "time", wind_map1.tt)
    h5write(joinpath(save_path, "wind_data.h5"), "data", wind_map1.wind_map)
    # close store
    # close(joinpath(save_path, "wind_data.h5"))
end


# Realistic case from Hell et al 2021
#95%-width = 2,800 km, 95%-duration = 4 days, max_u =22 m s, and V = 14.1 m s1.
mutable struct WindCase
    x_scale::Float64
    t_scale::Float64
    U10::Float64
    V::Float64
    name::String
end

realistic_case_hell2021 = WindCase(
    2800e3 / 2,
    4days / 2,
    22,
    14.1,
    "realistic_case_hell2021"
)

idealized_case_hell2021_fig5a = WindCase(
    1000e3 / 2,
    3.6days / 2,
    20,
    10,
    "idealized_case_hell2021_fig5a"
)

idealized_case_hell2021_fig5b = WindCase(
    1000e3 / 2,
    3.6days / 2,
    10,
    10,
    "idealized_case_hell2021_fig5b"
)

idealized_case_hell2021_fig5a_4ms = WindCase(
    1000e3 / 2,
    3.6days / 2,
    20,
    4,
    "idealized_case_hell2021_fig5a_4ms"
)

idealized_case_hell2021_fig5b_4ms = WindCase(
    1000e3 / 2,
    3.6days / 2,
    10,
    4,
    "idealized_case_hell2021_fig5b_4ms"
)


climatological_case = WindCase(
    1000e3 / 2,
    3.6days / 2,
    10,
    8.5,
    "climatological_case"
)

climatological_case2 = WindCase(
    1000e3 / 2,
    3.6days / 2,
    25,
    4,
    "climatological_case2"
)

climatological_case3 = WindCase(
    1000e3 / 2,
    3.6days / 2,
    25,
    8.5,
    "climatological_case3"
)

wind_min_squared = 0.01
dt = 30minutes
dx = 30e3

permanent_storing = false#true
# acase = idealized_case_hell2021

# for acase in [idealized_case_hell2021_fig5a_4ms, idealized_case_hell2021_fig5b_4ms, idealized_case_hell2021_fig5a, idealized_case_hell2021_fig5b, climatological_case2, realistic_case_hell2021, climatological_case]

for acase in [climatological_case3] #[climatological_case, climatological_case2, climatological_case3]

    x_scale, t_scale, U10, V, name = acase.x_scale, acase.t_scale, acase.U10, acase.V, acase.name
    # acase.V = V = Vi
    wind_map = make_wind_map(acase)

    T = t_scale * 6 #5
    xend = x_scale * (3 + V * 2)

    u(x, t) = 2.0+  WindEmulator.slopped_blob.(x, t, U10, V, T, x_scale, t_scale; x0=x_scale/2)

    wind_map1 = make_wind_map(xend, T, U10, V, T, x_scale, t_scale)
    p = plot()
    heatmap!(p, wind_map1.xx / 1e3, wind_map1.tt / 1day, transpose(wind_map1.wind_map), label="t=0")

    display(p)    

    grid1d = OneDGrid(0, xend, Integer(round(xend / dx)))
    wave_simulation = init_test(u, grid1d, dt, T; wind_min_squared=wind_min_squared, output_step_interval=Integer(60minutes / dt))
    initialize_simulation!(wave_simulation)

    if permanent_storing
        save_path = save_path_base *name
        mkpath(save_path)
        init_state_store!(wave_simulation, save_path; state=["e", "mx"])
        run!(wave_simulation, store=true, cash_store=false, debug=false)

        @info save_path
        #remove stored file
        if isfile(joinpath(save_path, "wind_data.h5"))
            rm(joinpath(save_path, "wind_data.h5"))
        end
        save_wind_data(save_path, wind_map1)

        close_store!(wave_simulation)

    else
        run!(wave_simulation, store=false, cash_store=true, debug=false)


        output = convert_state_store_to_array(wave_simulation.store.store, wave_simulation;
            time_iteration_interval=wave_simulation.model.output_step_interval,
            t_start=0.0)

        WindSeamin = FetchRelations.MinimalWindsea(U10, dt)
        p = plot_wave_simulation(output, wind_map.wind_map, wind_map.xx, wind_map.tt, grid1d.dx, wave_simulation.Δt, WindSeamin;
            cmap=cgrad(:ocean, rev=true), wind_min=sqrt(wind_min_squared), name=name)

        savefig(p, joinpath(plot_path_base, "moving_blob_test_dt=$(name).png"))
        display(p)

    end


end



# %%


wind_min_squared = 0.01
dt = 30minutes
dx = 30e3

function make_test_wind(x_scale, t_scale, U10)
    test_wind(x, t) = IfElse.ifelse.((x_scale *1 .< x) & (x .< x_scale * 4) & (t_scale/3 .< t) & (t .< t_scale * 3), U10, 0.0) + t * 0
    return test_wind
end
make_test_wind(x_scale, t_scale, U10)

function make_test_wind_map(xend, tend, U10, x_scale, t_scale)
    xx = collect(0:10e3:xend)
    tt = collect(0:30minutes:tend)
    xx_grid, tt_grid = [x for x in xx, t in tt], [t for x in xx, t in tt]

    wind_map = make_test_wind(x_scale, t_scale, U10).(xx_grid, tt_grid)
    return (xx=xx, tt=tt, wind_map=wind_map)
end
make_test_wind_map(xend, T, U10, x_scale, t_scale)

for Vi in [2]#, 4, 6, 8]

    #for Vi in [9]#[-1, 0, 2, 6]#4, 6, 8, 10, 12, 14]
    x_scale, t_scale, U10, V, name = climatological_case.x_scale, climatological_case.t_scale, climatological_case.U10, climatological_case.V, climatological_case.name
    climatological_case.V = V =  Vi

    wind_map = make_wind_map(climatological_case)
    #wind_map = make_test_wind_map(xend, T, U10, x_scale, t_scale)


    T = t_scale * 6 #5
    xend = 9000e3#
    #xend= x_scale * (3 + V * 2)

    u_wind(x, t) = 2.0 +WindEmulator.slopped_blob.(x, t, U10, V, T, x_scale, t_scale; x0=x_scale / 2)
    #u_wind(x, t) = make_test_wind(x_scale, t_scale, U10 * 0.4).( x, t)

    grid1d = OneDGrid(0, xend, Integer(round(xend / dx)))
    wave_simulation = init_test(u_wind, grid1d, dt, T; wind_min_squared=wind_min_squared, output_step_interval=Integer(60minutes / dt))
    initialize_simulation!(wave_simulation)

    run!(wave_simulation, store=false, cash_store=true, debug=false)

    output = convert_state_store_to_array(wave_simulation.store.store, wave_simulation;
        time_iteration_interval=wave_simulation.model.output_step_interval,
        t_start=0.0)

    WindSeamin = FetchRelations.MinimalWindsea(U10, dt)

    p = plot()
    p = plot_wave_simulation(output, wind_map.wind_map, wind_map.xx, wind_map.tt, grid1d.dx, wave_simulation.Δt, WindSeamin;
        cmap=cgrad(:ocean, rev=true), wind_min=sqrt(wind_min_squared), name=name)

    savefig(p, joinpath(plot_path_base, "moving_blob_test_dt=$(name)_V=$(V).png"))
    display(p)

end


