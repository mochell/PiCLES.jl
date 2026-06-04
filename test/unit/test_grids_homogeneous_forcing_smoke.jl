"""
Grid smoke tests.

Each available grid type is initialized and run for about one hour with
homogeneous forcing. The tests assert that the run does not fail and that the
domain energy state remains finite.
"""

using Test

using PiCLES
using PiCLES.ParticleSystems: particle_waves_v6 as PW
using PiCLES.Models.WaveGrowthModels2D
using PiCLES.Simulations: Simulation, initialize_simulation!
using PiCLES.Operators: TimeSteppers
using PiCLES.Operators.core_2D: ParticleDefaults
using PiCLES.custom_structures: ForcingCollection
using PiCLES.Grids.TripolarGridMOM6: TripolarGridMOM6
import PiCLES: FetchRelations

const GRID_SMOKE_DT = 20 * 60.0
const GRID_SMOKE_STEPS = 3

homogeneous_u(x, y, t) = 6.0
homogeneous_v(x, y, t) = -2.0

function centered_box_bounds(n::Int)
    center = fld(n, 2)
    half_width = max(1, fld(n, 6))
    lo = clamp(center - half_width, 2, n - 1)
    hi = clamp(center + half_width, 2, n - 1)
    return lo, hi
end

function add_center_box_mask!(grid)
    nx, ny = size(grid.data.mask)
    ix1, ix2 = centered_box_bounds(nx)
    iy1, iy2 = centered_box_bounds(ny)

    ocean_mask = grid.data.mask .!= 0
    ocean_mask[ix1:ix2, iy1:iy2] .= false

    total_mask = PiCLES.Grids.make_boundaries(ocean_mask, grid.stats.Nx, grid.stats.Ny)
    grid.data.mask .= total_mask

    return grid
end

function make_wave_model(grid; periodic_boundary::Bool)
    odepars, const_id, _ = PW.ODEParameters(r_g=0.85)
    particle_system = PW.particle_equations(γ=const_id.γ, q=const_id.q, input=true, dissipation=true)

    windsea = FetchRelations.MinimalWindsea(6.0, 2.0, GRID_SMOKE_DT)
    defaults = ParticleDefaults(log(windsea["E"]), windsea["cg_bar_x"], windsea["cg_bar_y"], 0.0, 0.0)

    ode_settings = PW.ODESettings(
        Parameters=odepars,
        log_energy_minimum=log(windsea["E"]),
        log_energy_maximum=log(27),
        saving_step=GRID_SMOKE_DT,
        timestep=GRID_SMOKE_DT,
        total_time=6 * GRID_SMOKE_DT,
        dt=1e-3,
        dtmin=1e-4,
        dtmax=GRID_SMOKE_DT,
    )

    return WaveGrowthModels2D.WaveGrowth2D(
        grid=grid,
        winds=nothing,
        ODEsys=particle_system,
        ODEsets=ode_settings,
        ODEinit_type=defaults,
        periodic_boundary=periodic_boundary,
        boundary_type="same",
        movie=false,
    )
end

function run_one_hour_smoke!(model)
    forcing = ForcingCollection(u_wind=homogeneous_u, v_wind=homogeneous_v)
    sim = Simulation(model, Δt=GRID_SMOKE_DT, stop_time=GRID_SMOKE_STEPS * GRID_SMOKE_DT, forcing=forcing, verbose=false)
    initialize_simulation!(sim)

    t0 = sim.model.clock.time
    for _ in 1:GRID_SMOKE_STEPS
        TimeSteppers.time_step!(sim.model, sim.Δt; forcing=sim.forcing)
    end

    energy = sim.model.State[:, :, 1]
    return sim.model.clock.time - t0, energy
end

@testset "grid homogeneous forcing smoke" begin
    grid_file = joinpath(@__DIR__, "..", "..", "src", "Grids", "files", "ocean_hgrid_221123.nc")

    grid_builders = [
        ("cartesian", () -> PiCLES.Grids.CartesianGrid.TwoDCartesianGridMesh(20e3, 9, 20e3, 9; periodic_boundary=(true, true)), true),
        ("spherical", () -> PiCLES.Grids.SphericalGrid.TwoDSphericalGridMesh(0.0, 20.0, 15, -10.0, 10.0, 13; periodic_boundary=(true, false)), false),
        ("tripolar", () -> TripolarGridMOM6.MOM6GridMesh(grid_file, 8; mask_radius=5), false),
    ]

    for (label, build_grid, periodic_boundary) in grid_builders
        @testset "grid=$label" begin
            @testset "no center mask" begin
                grid = build_grid()
                model = make_wave_model(grid; periodic_boundary=periodic_boundary)
                elapsed, energy = run_one_hour_smoke!(model)

                @test elapsed ≈ GRID_SMOKE_STEPS * GRID_SMOKE_DT
                @test all(isfinite, energy)
                @test isfinite(sum(energy))
            end

            @testset "center box mask" begin
                grid = build_grid()
                add_center_box_mask!(grid)

                @test any(grid.data.mask .== 0)
                @test any(grid.data.mask .== 2)

                model = make_wave_model(grid; periodic_boundary=periodic_boundary)
                elapsed, energy = run_one_hour_smoke!(model)

                @test elapsed ≈ GRID_SMOKE_STEPS * GRID_SMOKE_DT
                @test all(isfinite, energy)
                @test isfinite(sum(energy))
            end
        end
    end
end