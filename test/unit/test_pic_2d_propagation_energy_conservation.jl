"""
2D blob energy-window test (dissipation off).

Builds a compact version of the manual 2D blob setup and checks that, after 100
steps, the final total domain energy remains within +/-2% of the energy at step 3.
"""

using Test

using PiCLES
using PiCLES.ParticleSystems
import PiCLES: FetchRelations
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh
using PiCLES.Models.WaveGrowthModels2D
using PiCLES.Simulations
using PiCLES.Operators: TimeSteppers
using PiCLES.Operators.core_2D: ParticleDefaults
using PiCLES.Operators.mapping_2D: reset_PI_u!, ParticleToNode!
using Oceananigans.Units

function seed_blob!(sim; cgx, cgy, patch_i=5:15, patch_j=5:15)
    for PI in sim.model.ParticleCollection[patch_i, patch_j]
        reset_PI_u!(PI, ui=FetchRelations.get_initial_windsea(cgx, cgy, 2hour, particle_state=true))
        ParticleToNode!(PI, sim.model.State, sim.model.grid, sim.model.periodic_boundary, Val(sim.model.spline_order))
    end
end

function energy_series_for_case(; cgx, cgy, steps=100, spline_order=1)
    U10, V10 = 0.0, 0.0
    DT = Float64(20minutes)

    ODEpars, Const_ID, _ = ODEParameters(r_g=0.85)
    u(x, y, t) = ifelse.(x .< 250e3, U10, 0.00) + y * 0.0 + t * 0.0
    v(x, y, t) = ifelse.(x .< 250e3, V10, 0.00) + y * 0.0 + t * 0.0
    forcing = PiCLES.custom_structures.ForcingCollection(u_wind=u, v_wind=v)

    grid = TwoDCartesianGridMesh(400e3, 41, 300e3, 31; periodic_boundary=(true, true))

    particle_system = particle_equations(
        γ=Const_ID.γ,
        q=Const_ID.q,
        propagation=true,
        input=false,
        dissipation=false,
        peak_shift=false,
        direction=false,
    )

    WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT)
    lne_local = log(WindSeamin["E"])
    ODE_settings = ODESettings(
        Parameters=ODEpars,
        log_energy_minimum=lne_local,
        log_energy_maximum=log(27),
        saving_step=6000,
        timestep=DT,
        total_time=12days,
        dt=1e-3,
        dtmin=1e-4,
        dtmax=DT,
    )

    particle_min = FetchRelations.MinimalParticle(2, 0, DT)

    model = WaveGrowthModels2D.WaveGrowth2D(
        grid=grid,
        ODEsys=particle_system,
        ODEsets=ODE_settings,
        ODEinit_type=ParticleDefaults(particle_min),
        periodic_boundary=true,
        boundary_type="same",
        spline_order=spline_order,
        movie=false,
    )

    sim = Simulation(model; forcing=forcing, Δt=DT, stop_time=16hours)
    initialize_simulation!(sim)
    seed_blob!(sim; cgx=cgx, cgy=cgy)

    energies = Float64[]
    for _ in 1:steps
        TimeSteppers.time_step!(sim.model, sim.Δt; forcing=sim.forcing)
        push!(energies, sum(sim.model.State[:, :, 1]))
        sim.model.State[:, :, :] .= 0.0
    end
    return energies
end

@testset "2D blob total energy window (dissipation off)" begin
    cases = [
        (name="upper-right", cgx=10.0, cgy=12.0),
        (name="right-only", cgx=12.0, cgy=0.0),
        (name="bottom-only", cgx=0.0, cgy=-12.0),
        (name="lower-left", cgx=-10.0, cgy=-12.0),
    ]

    # Sweep B-spline deposition order. Partition of unity makes the deposit conservative at every
    # order, so the same ±2% window must hold for P = 1, 2, 3 (issues #59, #60).
    @testset "spline_order=$(P)" for P in (1, 2, 3)
        for case in cases
            e = energy_series_for_case(; cgx=case.cgx, cgy=case.cgy, steps=100, spline_order=P)
            @test length(e) == 100

            e_ref = e[3]
            e_final = e[end]

            @test isfinite(e_ref) && isfinite(e_final)
            @test e_ref != 0.0

            rel_change = abs(e_final - e_ref) / abs(e_ref)
            @test rel_change <= 0.02
        end
    end
end
