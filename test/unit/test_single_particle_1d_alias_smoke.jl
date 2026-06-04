"""
1D alias smoke test.

The 1D capability is implemented as an alias over the 2D engine: `WaveGrowth1D` builds a
`WaveGrowth2D` on a thin, periodic-in-y `CartesianGridMesh1D` with zero meridional wind. This
test runs a short homogeneous-wind 1D simulation through the canonical particle-system API and
asserts the degenerate-2D run stays genuinely 1D:
  * every y-row of the state is identical (no cross-row leakage),
  * the meridional momentum component stays ~0 (c_y ≡ 0),
  * energy grows under wind input, and
  * the stored output squeezes back to a 1D `(time, x, state)` shape.
"""

using Test

using PiCLES
using PiCLES.ParticleSystems            # canonical, version-agnostic API
using PiCLES.Grids.CartesianGrid: CartesianGridMesh1D, gridnotes_1d
using PiCLES.Models: WaveGrowth1D
using PiCLES.Simulations: Simulation, initialize_simulation!, run!, convert_store_to_tuple
import PiCLES: FetchRelations

const DT_1D = 20 * 60.0

@testset "1D alias smoke" begin
    U10 = 10.0
    uwind(x, t) = 0.0 * x + U10

    odepars, const_id, _ = ODEParameters(r_g=0.85)
    particle_system = particle_equations(γ=const_id.γ, q=const_id.q)

    windsea = FetchRelations.MinimalWindsea(U10, 0.0, DT_1D)
    ode_settings = ODESettings(
        Parameters=odepars,
        log_energy_minimum=log(windsea["E"]),
        log_energy_maximum=log(27),
        saving_step=DT_1D,
        timestep=DT_1D,
        total_time=6 * 3600.0,
        dt=1e-3, dtmin=1e-4, dtmax=DT_1D,
    )

    grid = CartesianGridMesh1D(100e3, 40; Ny=3, periodic_boundary=true)
    @test grid isa PiCLES.Architectures.CartesianGrid2D   # accepted by the 2D engine
    @test length(gridnotes_1d(grid)) == 40

    model = WaveGrowth1D(; grid=grid, winds=uwind, ODEsys=particle_system,
        ODEsets=ode_settings, ODEinit_type="wind_sea",
        periodic_boundary=true, boundary_type="same")

    sim = Simulation(model, Δt=DT_1D, stop_time=2 * 3600.0, verbose=false)
    initialize_simulation!(sim)
    E0 = maximum(model.State[:, 1, 1])
    run!(sim, cash_store=true, debug=false)
    S = model.State

    # all y-rows identical -> the run is genuinely 1D
    @test all(isapprox.(S[:, 1, :], S[:, 2, :]; atol=1e-8))
    @test all(isapprox.(S[:, 1, :], S[:, 3, :]; atol=1e-8))

    # meridional momentum stays ~0 (c_y ≡ 0)
    @test maximum(abs.(S[:, :, 3])) < 1e-6

    # energy grows under wind input and stays finite
    @test maximum(S[:, 1, 1]) > E0
    @test all(isfinite, S)

    # output squeezes back to 1D: (time, x, state)
    out = convert_store_to_tuple(sim.store, sim)
    @test ndims(out.data) == 3
    @test size(out.data, 2) == 40
    @test length(out.x) == 40
end
