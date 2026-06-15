"""
Quantitative 1D-wave-growth periodic regression tests (no plotting).

Checks for periodic case sweeps:
1) monotonic energy growth,
2) monotonic peak-frequency decrease,
3) agreement with PM limits at t_tilde = 1.5e5.
"""

using Test

using PiCLES
using PiCLES.ParticleSystems
using PiCLES.Grids.CartesianGrid: CartesianGridMesh1D
using PiCLES.Models: WaveGrowth1D
using PiCLES.Simulations: Simulation, initialize_simulation!, run!, convert_store_to_tuple
import PiCLES: FetchRelations
using PiCLES: FetchRelations as FR

using Oceananigans.Units
using Statistics

const T_TILDE_TARGET = 1.5e5
const MONO_SKIP_STEPS = 5
const PW_TOL_E = 0.10
const PW_TOL_FP = 0.20

squeeze(a) = dropdims(a, dims=tuple(findall(size(a) .== 1)...))

function get_fetch_variables_from_model_output(data_slice, r_g)
    wave_energy = data_slice[:, 1]
    wave_mx = data_slice[:, 2]

    wave_cgbar = wave_energy ./ 2.0 ./ wave_mx
    wave_omega_p = r_g * 9.81 ./ (2 * wave_cgbar)
    wave_fp = wave_omega_p / (2 * pi)
    return (energy=wave_energy, fp=wave_fp)
end

function get_non_dim_data(Pdata, u10, x, t)
    x_tilde = FR.X_tilde(x, u10)
    t_tilde = FR.t_tilde(t, u10)
    E_pic_tilde = FR.E_tilde(Pdata.energy, u10)
    Fp_pic_tilde = FR.f_p_tilde(Pdata.fp, u10)
    return (x_tilde=x_tilde, t_tilde=t_tilde, E_tilde=E_pic_tilde, Fp_tilde=Fp_pic_tilde)
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

@testset "1D wave-growth quantitative periodic checks" begin
    odepars, const_id, _ = ODEParameters(r_g=0.85)

    case_list = [
        (u10 = 15.0, DT = 10minutes, Nx = 21),
        (u10 = 15.0, DT = 10minutes, Nx = 101),
        (u10 = 15.0, DT = 10minutes, Nx = 201),
        (u10 = 15.0, DT = 5minutes,  Nx = 51),
        (u10 = 15.0, DT = 20minutes, Nx = 51),
        (u10 = 5.0, DT = 10minutes,  Nx = 51),
        (u10 = 10.0, DT = 10minutes, Nx = 51),
        (u10 = 20.0, DT = 10minutes, Nx = 51),
        (u10 = 15.0, DT = 10minutes, Nx = 101),
    ]

    # B-spline deposition order sweep. The full case list runs at P=1 (CIC, the established
    # coverage); higher orders P=2,3 are validated on a representative case to keep runtime
    # reasonable while confirming the wave-growth physics is unchanged by the deposition order
    # (issues #59, #60). Index 2 is the u10=15, DT=10min, Nx=101 case.
    cases_for_order = Dict(1 => collect(eachindex(case_list)), 2 => [2], 3 => [2])

    @testset "Periodic case sweep, spline_order=$(P)" for P in (1, 2, 3)
        for i in cases_for_order[P]
            case = case_list[i]
            u10, DT, Nx = case
            t_target_sec = T_TILDE_TARGET * u10 / 9.81
            stop_time = t_target_sec + DT

            grid = CartesianGridMesh1D(0, 1e7, Nx; Ny=3, periodic_boundary=true)
            windsea = FetchRelations.get_initial_windsea(u10, DT)
            ode_settings = ODESettings(
                Parameters=odepars,
                log_energy_minimum=log(windsea["E"]),
                log_energy_maximum=log(17),
                saving_step=5minutes,
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
                boundary_type="same",
                spline_order=P)

            sim = Simulation(model, Δt=DT, stop_time=stop_time, verbose=false)
            initialize_simulation!(sim)
            run!(sim, store=false, cash_store=true, debug=false)

            out = convert_store_to_tuple(sim.store, sim)
            data_periodic_slice = squeeze(mean(out.data[:, :, :], dims=2))
            picles_data = get_fetch_variables_from_model_output(data_periodic_slice, odepars.r_g)
            nd = get_non_dim_data(picles_data, u10, out.x, out.time)

            @testset "case $(i): u10=$(u10), DT=$(DT), Nx=$(Nx)" begin
                @test is_monotonic_increasing(nd.E_tilde)
                @test is_monotonic_decreasing(nd.Fp_tilde)
                assert_pw_at_target(nd.t_tilde, nd.E_tilde, nd.Fp_tilde)
            end
        end
    end
end
