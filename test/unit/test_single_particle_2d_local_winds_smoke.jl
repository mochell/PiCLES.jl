"""
Smoke test for the 2D single-particle local-winds setup.

This checks two execution paths with the same forcing:
1. direct `ODEIntegrator` + `solve!`
2. `InitParticleInstance` + `solve!`
"""

using Test

using PiCLES
using PiCLES.ParticleSystems: particle_waves_v6 as PW
using PiCLES.Utils: Init_Standard
using PiCLES.Operators.core_2D: InitParticleInstance, initParticleDefaults
using PiCLES.Solvers.RK35Integrator: ODEIntegrator, solve!
using Oceananigans.Units

const LOCAL_WIND_DT = 2hours

local_u(x::Number, y::Number, t::Number) = (5.0 * cos(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
local_v(x::Number, y::Number, t::Number) = -(5.0 * sin(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0

function make_setup()
    particle_state, default_ode_parameters, windseamin, const_id = Init_Standard(local_u(0.0, 0.0, 0.0), local_v(0.0, 0.0, 0.0), LOCAL_WIND_DT)
    particle_system = PW.particle_equations(γ=const_id.γ, q=const_id.q)
    ode_settings = PW.ODESettings(
        Parameters=default_ode_parameters,
        log_energy_minimum=log(windseamin["E"]),
        log_energy_maximum=log(17),
        saving_step=2minutes,
        timestep=LOCAL_WIND_DT * 3,
        total_time=6days,
        solver=nothing,
        reltol=1e-3,
        abstol=1e-4,
        dt=6minutes,
        dtmin=0.1,
        dtmax=10minutes,
    )

    return particle_state, particle_system, ode_settings
end

function make_forcing()
    seed_forcing = PW.ForcingData(u_wind=local_u(0.0, 0.0, 0.0), v_wind=local_v(0.0, 0.0, 0.0))
    forcing_collection = PiCLES.custom_structures.ForcingCollection(u_wind=local_u, v_wind=local_v)
    return seed_forcing, forcing_collection
end

function make_integrator(particle_system, particle_state, ode_settings, seed_forcing, xy)
    z_initials = initParticleDefaults(particle_state)
    z_initials[4] = xy[1]
    z_initials[5] = xy[2]
    parameters = (; ode_settings.Parameters..., x=xy[1], y=xy[2])
    return ODEIntegrator(
        particle_system,
        z_initials,
        0.0,
        parameters;
        forcing=seed_forcing,
        dt=ode_settings.dt,
        reltol=ode_settings.reltol,
        abstol=ode_settings.abstol,
        dtmin=ode_settings.dtmin,
        dtmax=ode_settings.dtmax,
    )
end

function solve_bare_integrator(particle_system, particle_state, ode_settings, seed_forcing, forcing_collection, xy)
    integrator = make_integrator(particle_system, particle_state, ode_settings, seed_forcing, xy)
    ts, zs = solve!(integrator, ode_settings.timestep * 3; forcing=forcing_collection, saveat=ode_settings.saving_step)
    return integrator, ts, zs
end

function solve_wrapped_particle(particle_system, particle_state, ode_settings, seed_forcing, forcing_collection, xy)
    particle = InitParticleInstance(particle_system, particle_state, ode_settings, seed_forcing, (0, 0), xy, false, true)
    ts, zs = solve!(particle.ODEIntegrator, ode_settings.timestep * 3; forcing=forcing_collection, saveat=ode_settings.saving_step)
    return particle, ts, zs
end

@testset "single-particle 2D local-winds smoke" begin
    particle_state, particle_system, ode_settings = make_setup()
    seed_forcing, forcing_collection = make_forcing()
    xy = (1.0, 2.0)

    bare_integrator, bare_ts, bare_zs = solve_bare_integrator(particle_system, particle_state, ode_settings, seed_forcing, forcing_collection, xy)
    wrapped_particle, wrapped_ts, wrapped_zs = solve_wrapped_particle(particle_system, particle_state, ode_settings, seed_forcing, forcing_collection, xy)

    @test bare_ts[end] ≈ ode_settings.timestep * 3
    @test wrapped_ts[end] ≈ ode_settings.timestep * 3
    @test length(bare_ts) == length(wrapped_ts)
    @test all(isfinite, bare_integrator.u)
    @test all(isfinite, wrapped_particle.ODEIntegrator.u)
    @test all(isfinite, bare_zs[end])
    @test all(isfinite, wrapped_zs[end])
    @test bare_integrator.t ≈ wrapped_particle.ODEIntegrator.t
end