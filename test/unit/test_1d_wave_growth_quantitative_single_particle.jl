"""
Quantitative 1D-wave-growth single-particle regression tests (no plotting).

Checks for single-particle 1D/2D integrations:
1) monotonic energy growth,
2) monotonic peak-frequency decrease,
3) agreement with PM limits at t_tilde = 1.5e5.
"""

using Test

using PiCLES
using PiCLES.ParticleSystems
using PiCLES.Grids.CartesianGrid: CartesianGridMesh1D, gridnotes_1d
using PiCLES.Models: WaveGrowth1D
using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleValues, InitParticleInstance, GetParticleEnergyMomentum
using PiCLES.Solvers.RK35Integrator: solve!
import PiCLES: FetchRelations
using PiCLES: FetchRelations as FR

using Oceananigans.Units
using Statistics

const T_TILDE_TARGET = 1.5e5
const MONO_SKIP_STEPS = 5
const PW_TOL_E = 0.10
const PW_TOL_FP = 0.20

function get_fp_from_cg(cg)
    omega_p = 9.81 ./ (2 * cg)
    return omega_p / (2 * pi)
end

function format_particle_data_saved(PI, t_end; saveat=nothing)
    ts, us = solve!(PI.ODEIntegrator, t_end; saveat=saveat, save=true)

    statelist = GetParticleEnergyMomentum.(us)
    statematix = hcat(statelist...)
    u_matrix = hcat(us...)

    return (
        x=u_matrix[4, :],
        y=u_matrix[5, :],
        time=ts,
        cgx=u_matrix[2, :],
        cgy=u_matrix[3, :],
        E=statematix[1, :],
        mx=statematix[2, :],
        my=statematix[3, :],
    )
end

function is_monotonic_increasing(v; skip_first=MONO_SKIP_STEPS)
    length(v) <= skip_first + 1 && return true
    vv = v[(skip_first + 1):end]
    return all(diff(vv) .>= 0)
end

function is_monotonic_decreasing(v; skip_first=MONO_SKIP_STEPS)
    length(v) <= skip_first + 1 && return true
    vv = v[(skip_first + 1):end]
    return all(diff(vv) .<= 0)
end

function assert_pw_at_target(t_tilde, e_tilde, fp_tilde; t_target=T_TILDE_TARGET, tol_e=PW_TOL_E, tol_fp=PW_TOL_FP)
    @test maximum(t_tilde) >= t_target
    idx = argmin(abs.(t_tilde .- t_target))
    pm = FR.PMlimits()

    @test abs(e_tilde[idx] - pm.E_tilde) <= tol_e * abs(pm.E_tilde)
    @test abs(fp_tilde[idx] - pm.f_p_tilde) <= tol_fp * abs(pm.f_p_tilde)
end

@testset "1D wave-growth quantitative single-particle checks" begin
    odepars, const_id, _ = ODEParameters(r_g=0.85)

    u10 = 15.0
    DT = 10minutes
    Nx = 101
    t_target_sec = T_TILDE_TARGET * u10 / 9.81
    stop_time = t_target_sec + DT

    grid = CartesianGridMesh1D(0, 1e7, Nx; Ny=3, periodic_boundary=true)
    windsea = FetchRelations.get_initial_windsea(u10, DT)
    ode_settings = ODESettings(
        Parameters=odepars,
        log_energy_minimum=log(windsea["E"]),
        log_energy_maximum=log(17),
        saving_step=DT,
        timestep=DT,
        total_time=stop_time,
        dt=1e-3,
        dtmin=1e-9,
    )

    uwind(x, t) = x * 0 + t * 0 + u10
    model = WaveGrowth1D(; grid=grid,
        winds=uwind,
        ODEsys=particle_equations(γ=const_id.γ, q=const_id.q, IDConstants=const_id),
        ODEsets=ode_settings,
        ODEinit_type="mininmal",
        minimal_particle=FetchRelations.MinimalParticle(u10, 0, DT),
        periodic_boundary=true,
        boundary_type="same")

    xx = gridnotes_1d(model.grid)[1]

    # 1D particle (embedded in 2D state)
    ws1 = FetchRelations.get_initial_windsea(u10, DT)
    pd1 = ParticleDefaults(log(ws1["E"]), ws1["cg_bar"], 0.0, xx, 0.0)
    ps1, on1 = InitParticleValues(pd1, (xx, 0.0), (u10, 0.0), DT)
    f1 = ForcingData(u_wind=u10, v_wind=0.0)
    pi1 = InitParticleInstance(model.ODEsystem, ps1, ode_settings, f1, (1, 1), (xx, 0.0), false, on1)

    p1 = format_particle_data_saved(pi1, stop_time; saveat=DT)
    p1_t = FR.t_tilde(p1.time, u10)
    p1_E = FR.E_tilde(p1.E, u10)
    p1_fp = FR.f_p_tilde(get_fp_from_cg(p1.cgx / odepars.r_g), u10)

    @test is_monotonic_increasing(p1_E)
    @test is_monotonic_decreasing(p1_fp)
    assert_pw_at_target(p1_t, p1_E, p1_fp)

    # 2D particle
    u2(x, y, t) = x * 0 + t * 0 + u10
    ws2 = FetchRelations.get_initial_windsea(u10, 0, DT)
    pd2 = ParticleDefaults(log(ws2["E"]), ws2["cg_bar_x"], ws2["cg_bar_y"], xx, xx)
    ps2, on2 = InitParticleValues(pd2, (xx, xx), (u2(xx, xx, 0), 0.0), DT)
    sys2 = particle_equations(γ=const_id.γ, q=const_id.q, IDConstants=const_id)
    f2 = ForcingData(u_wind=u10, v_wind=0.0)
    pi2 = InitParticleInstance(sys2, ps2, ode_settings, f2, (1, 1), (xx, xx), false, on2)

    p2 = format_particle_data_saved(pi2, stop_time; saveat=DT)
    p2_t = FR.t_tilde(p2.time, u10)
    p2_E = FR.E_tilde(p2.E, u10)
    p2_fp = FR.f_p_tilde(get_fp_from_cg(p2.cgx / odepars.r_g), u10)

    @test is_monotonic_increasing(p2_E)
    @test is_monotonic_decreasing(p2_fp)
    assert_pw_at_target(p2_t, p2_E, p2_fp)
end
