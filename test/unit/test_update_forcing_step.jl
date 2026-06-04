using Test

using PiCLES
using SharedArrays
using PiCLES.ParticleSystems: particle_waves_v6 as PW
using PiCLES.Utils: Init_Standard
using PiCLES.Operators.core_2D: InitParticleInstance, initParticleDefaults
using PiCLES.Operators: mapping_2D
using PiCLES.Solvers.RK35Integrator: ODEIntegrator, step!, update_forcing!
using Oceananigans.Units
using PiCLES.ParticleMesh: TwoDGrid

let
    u(x::Number, y::Number, t::Number) = (5.0 * cos(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
    v(x::Number, y::Number, t::Number) = -(5.0 * sin(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0

    DT = 2hours
    ParticleState, default_ODE_parameters, WindSeamin, Const_ID = Init_Standard(u(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), DT)
    particle_system = PW.particle_equations(γ=Const_ID.γ, q=Const_ID.q)

    ODE_settings = PW.ODESettings(
        Parameters=default_ODE_parameters,
        log_energy_minimum=log(WindSeamin["E"]),
        log_energy_maximum=log(17),
        saving_step=2minutes,
        timestep=DT * 3,
        total_time=6days,
        solver=nothing,
        reltol=1e-3,
        abstol=1e-4,
        dt=6minutes,
        dtmin=0.1,
        dtmax=10minutes,
    )

    z_initials = initParticleDefaults(ParticleState)
    forcing_collection = PiCLES.custom_structures.ForcingCollection(u_wind=u, v_wind=v)

    function make_local_test_integrator()
        forcing0 = PW.ForcingData(u_wind=u(0.0, 0.0, 0.0), v_wind=v(0.0, 0.0, 0.0))
        return ODEIntegrator(particle_system, copy(z_initials), 0.0, ODE_settings.Parameters;
            forcing=forcing0,
            dt=ODE_settings.dt,
            reltol=ODE_settings.reltol,
            abstol=ODE_settings.abstol,
            dtmin=ODE_settings.dtmin,
            dtmax=ODE_settings.dtmax)
    end

    function forcing_tuple(integ, forcing)
        return (
            forcing.u_wind(integ.u[4], integ.u[5], integ.t),
            forcing.v_wind(integ.u[4], integ.u[5], integ.t),
        )
    end

    @testset "update_forcing! refreshes ODEIntegrator before step!" begin
        integ_stale = make_local_test_integrator()
        integ_updated = make_local_test_integrator()

        step!(integ_stale, DT, true)
        step!(integ_updated, DT, true)

        expected_second_step_forcing = forcing_tuple(integ_updated, forcing_collection)
        stale_forcing_before_second_step = (integ_stale.forcing.u_wind, integ_stale.forcing.v_wind)

        step!(integ_stale, DT, true)
        stale_forcing_after_second_step = (integ_stale.forcing.u_wind, integ_stale.forcing.v_wind)

        update_forcing!(integ_updated, forcing_collection)
        updated_forcing_before_second_step = (integ_updated.forcing.u_wind, integ_updated.forcing.v_wind)
        step!(integ_updated, DT, true)

        forcing_updated = all((
            isapprox(updated_forcing_before_second_step[1], expected_second_step_forcing[1]; atol=1e-8, rtol=1e-8),
            isapprox(updated_forcing_before_second_step[2], expected_second_step_forcing[2]; atol=1e-8, rtol=1e-8),
        ))

        forcing_stays_stale_without_refresh = stale_forcing_after_second_step == stale_forcing_before_second_step

        stale_is_not_expected = !all((
            isapprox(stale_forcing_before_second_step[1], expected_second_step_forcing[1]; atol=1e-6, rtol=1e-6),
            isapprox(stale_forcing_before_second_step[2], expected_second_step_forcing[2]; atol=1e-6, rtol=1e-6),
        ))

        state_updated_too = !all((
            isapprox(integ_stale.u[2], integ_updated.u[2]; atol=1e-8, rtol=1e-8),
            isapprox(integ_stale.u[3], integ_updated.u[3]; atol=1e-8, rtol=1e-8),
            isapprox(integ_stale.u[4], integ_updated.u[4]; atol=1e-8, rtol=1e-8),
            isapprox(integ_stale.u[5], integ_updated.u[5]; atol=1e-8, rtol=1e-8),
        ))

        @test forcing_stays_stale_without_refresh
        @test stale_is_not_expected
        @test forcing_updated
        @test state_updated_too
    end

    @testset "mapping_2D.advance! refreshes integrator forcing before step!" begin
        forcing0 = PW.ForcingData(u_wind=u(0.0, 0.0, 0.0), v_wind=v(0.0, 0.0, 0.0))
        particle = InitParticleInstance(
            particle_system,
            ParticleState,
            ODE_settings,
            forcing0,
            (2, 2),
            (1.0, 1.0),
            false,
            true,
        )

        state = SharedArray{Float64,3}(3, 3, 3)
        state[:, :, :] .= 0.0
        failed = Vector{PiCLES.Architectures.AbstractMarkedParticleInstance}()
        grid = TwoDGrid(10.0, 3, 10.0, 3)
        forcing_before = (particle.ODEIntegrator.forcing.u_wind, particle.ODEIntegrator.forcing.v_wind)
        tuple_forcing = (1.25, -2.5)

        mapping_2D.advance!(
            particle,
            state,
            failed,
            grid,
            tuple_forcing,
            10.0,
            ODE_settings.log_energy_maximum,
            ODE_settings.wind_min_squared,
            false,
            nothing,
        )

        forcing_after = (particle.ODEIntegrator.forcing.u_wind, particle.ODEIntegrator.forcing.v_wind)

        @test forcing_before != tuple_forcing
        @test forcing_after == tuple_forcing
    end
end