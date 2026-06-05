# %%
# ENV["JULIA_INCREMENTAL_COMPILE"] = true
using Pkg
Pkg.activate("PiCLES/")

"""
Manual diagnostic: 2D PIC propagation from a seeded blob. This script runs a compact case matrix (upper-right, right-only, bottom-only, lower-left propagation; each with dissipation on/off) on a periodic Cartesian grid using simulation-level forcing, advances each case with repeated remeshing, and produces a 3x2 diagnostic figure (binary on/boundary masks, state heatmaps, and total-energy time series) with descriptive metadata in the suptitle, saving one final plot per case to `save_path`.
"""


import Plots as plt
using Setfield, IfElse

using PiCLES.ParticleSystems

import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleInstance, GetGroupVelocity
using PiCLES.Operators: TimeSteppers
using PiCLES.Simulations
using PiCLES.Operators.TimeSteppers: time_step!, movie_time_step!

using PiCLES.ParticleMesh: TwoDGrid, TwoDGridNotes, TwoDGridMesh
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, TwoDCartesianGridStatistics

using PiCLES.Models.WaveGrowthModels2D

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units
import Oceananigans.Utils: prettytime

using PiCLES.Architectures
using PiCLES.Architectures: AbstractGridStatistics, CartesianGridStatistics

using PiCLES.Operators.core_2D: ParticleDefaults as ParticleDefaults2D

#using GLMakie
using Plots

using PiCLES.Operators.core_2D: GetGroupVelocity, speed
using PiCLES.Plotting.movie: init_movie_2D_box_plot

using StaticArrays
using StructArrays
using BenchmarkTools

using PiCLES.Grids
using PiCLES

using PiCLES.Operators.TimeSteppers: time_step!
#using OrdinaryDiffEq
using PiCLES.Operators: mapping_2D



# %%
save_path = "plots/tests/S02_box_2D_mesh_grid/"
mkpath(save_path)
pwd()
# % Parameters
U10, V10 = 0.00, 00.0
DT = 20minutes
ODEpars, Const_ID, Const_Scg = ODEParameters(r_g=0.85)

u(x, y, t) = IfElse.ifelse.(x .< 250e3, U10, 0.00) + y * 0.0 + t * 0.0
v(x, y, t) = IfElse.ifelse.(x .< 250e3, V10, 0.00) + y * 0.0 + t * 0.0 .* cos(t * 5 / (1 * 60 * 60 * 2π))

# u(x, y, t) = U10 + x * 0.0 + y * 0.0 + t * 0.0
# v(x, y, t) = V10 + x * 0.0 + y * 0.0 + t * 0.0
forcing = PiCLES.custom_structures.ForcingCollection(u_wind=u, v_wind=v)

grid = TwoDCartesianGridMesh(400e3, 41, 300e3, 31; periodic_boundary=(true, true))

# %%
# define setting and standard initial conditions
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT);
lne_local  = round( log(WindSeamin["E"]) , digits=4)

ParticleMin = FetchRelations.MinimalParticle(2, 0, DT)
# get_initial_windsea(2, 2, DT,particle_state=true )

ODE_settings = ODESettings(
    Parameters=ODEpars,
    # define mininum energy threshold
    log_energy_minimum=lne_local,#log(FetchRelations.Eⱼ(0.1, DT)),
    #maximum energy threshold
    log_energy_maximum=log(27),#log(17),  # correcsponds to Hs about 16 m
    saving_step=6000,
    timestep=DT,
    total_time=T = 12days,
    dt=1e-3,
    dtmin=1e-4,
    dtmax=20minutes)



# %%

function plot_state(wave_model, energy_time, energy_series; case_title="")
    particles = wave_model.ParticleCollection
    total_energy = sum(wave_model.State[:, :, 1])
    meta_title = "iter=$(wave_model.clock.iteration) | total energy=$(round(total_energy, digits=4))"
    suptitle = isempty(case_title) ? meta_title : "$(case_title) | $(meta_title)"
    ymax = isempty(energy_series) ? 1.0 : maximum(energy_series)
    ymax_plot = max(1e-8, 1.1 * ymax)
    ymin_plot = isempty(energy_series) ? 0.0 : 0.95 * minimum(energy_series)

    p = plot(layout=(3, 2), size=(1000, 1200), plot_title=suptitle)
    heatmap!(p, transpose(particles.on), subplot=1,
        title="on (binary: 1=active, 0=inactive)",
        clims=(0, 1), c=:grays, colorbar=true)
    heatmap!(p, transpose(particles.boundary), subplot=2,
        title="boundary (binary: 1=boundary, 0=interior)",
        clims=(0, 1), c=:grays, colorbar=true)
    heatmap!(p, transpose(wave_model.State[:, :, 1]), subplot=3, title="State: Energy", clims=(0, NaN))
    heatmap!(p, transpose(wave_model.State[:, :, 2]), subplot=4, title="State: x momentum ", clims=(0, NaN))
    plot!(p, energy_time, energy_series, subplot=5,
        title="Total energy over time",
        xlabel="time [h]", ylabel="sum(E)",
        ylims=(ymin_plot, ymax_plot),
        label="total E", color=:black)
    heatmap!(p, transpose(wave_model.State[:, :, 3]), subplot=6, title="State: y momentum ")
    # Keep map-style panels square; let the time-series panel use a free aspect ratio.
    for s in (1, 2, 3, 4, 6)
        plot!(p, subplot=s, aspect_ratio=:equal)
    end
    plot!(p, subplot=5, aspect_ratio=:none)
    display(p)
    return p
end



# %%
using PiCLES.Operators.mapping_2D: reset_PI_u!, ParticleToNode!

function seed_blob!(wave_simulation; cgx, cgy, patch_i=5:15, patch_j=5:15)
    for PI in wave_simulation.model.ParticleCollection[patch_i, patch_j]
        reset_PI_u!(PI, ui=FetchRelations.get_initial_windsea(cgx, cgy, 2hour, particle_state=true))
        ParticleToNode!(PI, wave_simulation.model.State, wave_simulation.model.grid, wave_simulation.model.periodic_boundary)
    end
end

function run_case(; case_title, case_slug, cgx, cgy, dissipation)
    particle_system = particle_equations(
        γ=Const_ID.γ,
        q=Const_ID.q,
        propagation=true,
        input=false,
        dissipation=dissipation,
        peak_shift=false,
        direction=false,
    )

    wave_model = WaveGrowthModels2D.WaveGrowth2D(
        grid=grid,
        ODEsys=particle_system,
        ODEsets=ODE_settings,
        ODEinit_type=ParticleDefaults(ParticleMin),
        periodic_boundary=true,
        boundary_type="same",
        movie=true,
    )

    wave_simulation = Simulation(
        wave_model,
        forcing=forcing,
        Δt=20minutes,
        stop_time=16hours,
    )
    initialize_simulation!(wave_simulation)
    seed_blob!(wave_simulation; cgx=cgx, cgy=cgy)

    energy_time = [wave_simulation.model.clock.time / 3600]
    energy_series = [sum(wave_simulation.model.State[:, :, 1])]

    for i in 1:600
        TimeSteppers.time_step!(wave_simulation.model, wave_simulation.Δt; forcing=wave_simulation.forcing)

        push!(energy_time, wave_simulation.model.clock.time / 3600)
        push!(energy_series, sum(wave_simulation.model.State[:, :, 1]))

        if i % 8 == 0
            plot_state(wave_simulation.model, energy_time[2:end], energy_series[2:end]; case_title=case_title)
            sleep(0.02)
        end

        wave_simulation.model.State[:, :, :] .= 0.0
    end

    p_final = plot_state(wave_simulation.model, energy_time[2:end], energy_series[2:end]; case_title=case_title)
    savefig(p_final, joinpath(save_path, "T03_blob_" * case_slug * ".png"))
end

# %% Case matrix: 4 propagation directions x dissipation on/off
cases = [
    (title="Upper-right propagation | dissipation ON",  slug="ur_diss_on",  cgx=10.0,  cgy=12.0,  dissipation=true),
    (title="Upper-right propagation | dissipation OFF", slug="ur_diss_off", cgx=10.0,  cgy=12.0,  dissipation=false),
    (title="Right-only propagation | dissipation ON",   slug="r_diss_on",   cgx=12.0,  cgy=0.0,   dissipation=true),
    (title="Right-only propagation | dissipation OFF",  slug="r_diss_off",  cgx=12.0,  cgy=0.0,   dissipation=false),
    (title="Bottom-only propagation | dissipation ON",  slug="b_diss_on",   cgx=0.0,   cgy=-12.0, dissipation=true),
    (title="Bottom-only propagation | dissipation OFF", slug="b_diss_off",  cgx=0.0,   cgy=-12.0, dissipation=false),
    (title="Lower-left propagation | dissipation ON",   slug="ll_diss_on",  cgx=-10.0, cgy=-12.0, dissipation=true),
    (title="Lower-left propagation | dissipation OFF",  slug="ll_diss_off", cgx=-10.0, cgy=-12.0, dissipation=false),
]

for case in cases
    @info "Running case: $(case.title)"
    run_case(
        case_title=case.title,
        case_slug=case.slug,
        cgx=case.cgx,
        cgy=case.cgy,
        dissipation=case.dissipation,
    )
end


# %%