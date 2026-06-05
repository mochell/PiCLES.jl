"""
Manual diagnostic: 1D wave-growth regression checks.

Interactive script for plotting and visual comparison of 1D growth behavior.
"""

import Plots

#using PiCLES.ParticleSystems: particle_waves_v3beta as PW3
import PiCLES.ParticleSystems as PW
import PiCLES: FetchRelations
using Setfield, IfElse

using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleValues, InitParticleInstance

using PiCLES.Grids.CartesianGrid: CartesianGridMesh1D, gridnotes_1d

#using OrdinaryDiffEq

using PiCLES.Utils.ParticleTools
using Plots

using Oceananigans.Units

using PiCLES.Models: WaveGrowth1D
using PiCLES.Simulations
using PiCLES.Solvers.RK35Integrator: step!, solve!
using PiCLES.Plotting
using PiCLES.Operators.core_2D: GetParticleEnergyMomentum

using Statistics
using PiCLES: FetchRelations as FR
using JSON

"""
    format_particle_data_saved(PI, t_end; saveat=nothing)

Advance a particle integrator with history saving and return a NamedTuple matching
the structure produced by `ParticleTools.FormatParticleData`.
"""
function format_particle_data_saved(PI, t_end; saveat=nothing)
    ts, us = solve!(PI.ODEIntegrator, t_end; saveat=saveat, save=true)

    statelist = GetParticleEnergyMomentum.(us)
    statematix = hcat(statelist...)
    u_matrix = hcat(us...)

    if length(us[1]) == 3
        return (
            x=u_matrix[3, :],
            time=ts,
            cgx=u_matrix[2, :],
            E=statematix[1, :],
            mx=statematix[2, :],
        )
    end

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


squeeze(a) = dropdims(a, dims=tuple(findall(size(a) .== 1)...))

function get_fetch_variables_from_model_output(data_slice, r_g)
    wave_energy = data_slice[:, 1]
    wave_mx = data_slice[:, 2]

    wave_cgbar = wave_energy ./ 2.0 ./ wave_mx
    wave_omega_p = r_g * 9.81 ./ (2 * wave_cgbar)
    wave_fp = wave_omega_p / (2 * pi)
    return (energy=wave_energy, fp=wave_fp)
end

function get_fp_from_cg(cg)
    omega_p = 9.81 ./ (2 * cg)
    return omega_p / (2 * pi)
end



function convert_picles_nondim_to_savedict(data, label; attrs=nothing)
    return Dict("x_tilde" => data.x_tilde, "t_tilde" => data.t_tilde,
        "E_tilde" => data.E_tilde, "Fp_tilde" => data.Fp_tilde,
        "label" => label, "attrs" => attrs)
end

function get_non_dim_data(Pdata, u10, x, t)
    x_tilde = FR.X_tilde(x, u10)
    t_tilde = FR.t_tilde(t, u10)
    E_pic_tilde = FR.E_tilde(Pdata.energy, u10)
    Fp_pic_tilde = FR.f_p_tilde(Pdata.fp, u10)
    return (x_tilde=x_tilde, t_tilde=t_tilde, E_tilde=E_pic_tilde, Fp_tilde=Fp_pic_tilde)
end

# % Parameters
plot_path_base = "plots/tests/"
mkpath(plot_path_base)

save_path = "data/processed/FetchRelation_tests/"
mkpath(save_path)

# %%
# function to define constants 
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)


ODEpars
Const_ID
Const_Scg

c_D, c_β, c_e, c_alpha, r_w, C_e, γ, p, q, n =Const_ID.c_D, Const_ID.c_β, Const_ID.c_e, Const_ID.c_alpha, Const_ID.r_w, Const_ID.C_e, Const_ID.γ, Const_ID.p, Const_ID.q, Const_ID.n


# %% 
"""
Run 1D test cases for wind growth, with and without periodic boundaries. Compare to JONSWAP and PM64 fetch relations.
"""

DDcollect = Dict()

# typeof(ODEpars)
T           = 2.5day

# loop over u10= 5:5:20, DT = 5,10,20,30,60 minutes, Nx = 21, 51, 101, 201

DD_PIC_nonper = Dict()
DD_PIC_per    = Dict()
DD_failed     = Dict()

PIC_per      = nothing
PIC_nonper   = nothing
wave_model   = nothing

case_list = [
(u10 = 15.0, DT = 10minutes, Nx = 21),
(u10 = 15.0, DT = 10minutes, Nx = 101),
(u10 = 15.0, DT = 10minutes, Nx = 201),

(u10 = 15.0, DT = 5minutes,  Nx = 51),
(u10 = 15.0, DT = 20minutes, Nx = 51),

(u10 = 5.0, DT = 10minutes,  Nx = 51),
(u10 = 10.0, DT = 10minutes, Nx = 51),
(u10 = 20.0, DT = 10minutes, Nx = 51),

(u10 = 15.0, DT = 10minutes, Nx = 101)
]

u10, DT, Nx = case_list[end]

1e7/ 200 

for case in case_list

    #u10         = 15.0
    # DT          = 10minutes
    # Nx          = 51
    u10, DT, Nx = case

    grid1d = CartesianGridMesh1D(0, 1e7, Nx; Ny=3, periodic_boundary=true)
    grid1d.stats.dx

    Case_dict = Dict("u10" => u10, "DT" => DT, "dx" => grid1d.stats.dx)
    #make string from Case_dict, convert to integers
    Case_str = join(["$(k):$(round(Int, v))" for (k, v) in Case_dict], "_")

    try
        # define initial conditions
        WindSeamin  = FetchRelations.get_initial_windsea(u10, DT)

        ODE_settings = PW.ODESettings(
            Parameters=ODEpars,
            # define mininum energy threshold
            log_energy_minimum=log(WindSeamin["E"]),
            #maximum energy threshold
            log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
            saving_step=5minutes,
            timestep=DT,
            total_time=T,
            dt=1e-3, #60*10, 
            dtmin=1e-9, #60*5, 
        )


        # define model -
        u(x, t) = x .* 0 + t * 0 + u10
        # redefine model 
        wave_model = WaveGrowth1D(; grid=grid1d,
            winds=u,
            ODEsys=PW.particle_equations(γ=Const_ID.γ, q=Const_ID.q, IDConstants=Const_ID),
            ODEsets=ODE_settings,  # ODE_settings
            ODEinit_type="mininmal",  # ODEpars
            minimal_particle=FetchRelations.MinimalParticle(u10, 0, DT), #
            periodic_boundary=false,
            boundary_type="same"  # "default" #
        )

        # non periodic boundary
        wave_simulation = Simulation(wave_model, Δt=DT, stop_time=T)
        initialize_simulation!(wave_simulation)
        run!(wave_simulation, store=false, cash_store=true, debug=false);

        # periodic boundary
        wave_model.periodic_boundary    = true
        wave_simulation_periodic        = Simulation(wave_model, Δt=DT, stop_time=T)
        initialize_simulation!(wave_simulation_periodic)
        run!(wave_simulation_periodic, store=false, cash_store=true, debug=false);
        #Plotting.plot_results(wave_simulation_periodic, title="$u10 m/s, periodic=" * string(wave_model.periodic_boundary))

        data        = Simulations.convert_store_to_tuple(wave_simulation.store, wave_simulation)
        data_slice  = squeeze(maximum(data.data[end-6:1:end, :, :], dims=1))
        PiCLES_data = get_fetch_variables_from_model_output(data_slice, ODEpars.r_g)

        PIC_nonper = get_non_dim_data(PiCLES_data, u10, data.x, data.time)
        DD_PIC_nonper[Case_str] = convert_picles_nondim_to_savedict(PIC_nonper, "PiCLES_nonper", attrs=Case_dict)

        ### E tilde time
        data_periodic = Simulations.convert_store_to_tuple(wave_simulation_periodic.store, wave_simulation_periodic)
        data_periodic_slice = squeeze(mean(data_periodic.data[:, :, :], dims=2))
        PiCLES_data_periodic = get_fetch_variables_from_model_output(data_periodic_slice, ODEpars.r_g)

        PIC_per = get_non_dim_data(PiCLES_data_periodic, u10, data_periodic.x, data_periodic.time)
        DD_PIC_per[Case_str] = convert_picles_nondim_to_savedict(PIC_per, "PiCLES_per", attrs=Case_dict)
    catch err
        DD_failed[Case_str] = sprint(showerror, err)
        @warn "Case failed; skipping" case=Case_str error=DD_failed[Case_str]
        continue
    end

end


DDcollect["PIC_nonper"] = DD_PIC_nonper
DDcollect["PIC_per"]    = DD_PIC_per
DDcollect["failed_cases"] = DD_failed

@info "case summary" succeeded=length(keys(DD_PIC_per)) failed=length(keys(DD_failed))


# %%


# %% # single particle 1D

T_test, save_time_test = 2*24hours , 20minutes

xx                         = gridnotes_1d(wave_model.grid)[1]
WindSeamin                 = FetchRelations.get_initial_windsea(u10, DT)
ODE_settings               = wave_model.ODEsettings
particle_defaults          = ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar"], 0.0, xx, 0.0)
ParticleState, particle_on = InitParticleValues(particle_defaults, (xx, 0.0), (u10, 0.0), DT)
forcing_1d                 = PW.ForcingData(u_wind=u10, v_wind=0.0)
PI4                        = InitParticleInstance(wave_model.ODEsystem, ParticleState, ODE_settings, forcing_1d, (1, 1), (xx, 0.0), false, particle_on)


PI = format_particle_data_saved(PI4, T_test; saveat=save_time_test)

PI4.ODEIntegrator

PI1D_x_tilde = FR.X_tilde(PI.x, u10)
PI1D_t_tilde = FR.t_tilde(PI.time, u10)
PI1D= Dict(   "x_tilde" => PI1D_x_tilde, 
        "t_tilde" => PI1D_t_tilde, 
        "E_tilde" => FR.E_tilde(PI.E, u10),
        "Fp_tilde" => FR.f_p_tilde(get_fp_from_cg(PI.cgx / ODEpars.r_g), u10), 
        "label" => "Single Particle 1D",
        "attrs" => nothing)


plot!(p_diag, PI1D_t_tilde, PI1D["E_tilde"], linestyle=:dash, lw=3, marker=:x, ms=3, label="Single Particle 1D", subplot=1)

plot!(p_diag, PI1D_t_tilde, PI1D["Fp_tilde"], linestyle=:dash, lw=3, marker=:x, ms=3, label="Single Particle 1D", subplot=2)


# % single particle 2D
u2(x, y, t) = x .* 0 + t * 0 + u10
v2(x, y, t) = x .* 0 + t * 0 + y *0 
WindSeamin                 = FetchRelations.get_initial_windsea(u10, 0,  DT)
particle_defaults          = ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], xx, xx)
ParticleState, particle_on = InitParticleValues(particle_defaults, (xx, xx), (u2(xx, xx, 0), 0.0), DT)
ODESystem2D                = PW.particle_equations(γ=Const_ID.γ, q=Const_ID.q, IDConstants=Const_ID)
forcing_2d                 = PW.ForcingData(u_wind=u10, v_wind=0.0)
PI5                        = InitParticleInstance(ODESystem2D, ParticleState, ODE_settings, forcing_2d, (1, 1), (xx, xx), false, particle_on)

PI2 = format_particle_data_saved(PI5, T_test; saveat=save_time_test)

PI2D_x_tilde = FR.X_tilde(PI2.x, u10)
PI2D_t_tilde = FR.t_tilde(PI2.time, u10)
PI2D= Dict(   "x_tilde" => PI2D_x_tilde, 
        "t_tilde" => PI2D_t_tilde, 
        "E_tilde" => FR.E_tilde(PI2.E, u10),
        "Fp_tilde" => FR.f_p_tilde(get_fp_from_cg(PI2.cgx / ODEpars.r_g), u10), 
        "label" => "Single Particle 2D",
        "attrs" => nothing)


# %% final plotting (2x2)


@info "PLots periodic case keys" keys(DD_PIC_per)

case_keys = sort(collect(keys(DD_PIC_per)))

if isempty(case_keys)
    error("No cases found in DD_PIC_per")
end

first_case_key = case_keys[1]
first_per = DD_PIC_per[first_case_key]

t_ref = first_per["t_tilde"]
u10_ref = first_per["attrs"]["u10"]
x_tilde_tau_ref = FR.X_tilde_from_tau.(t_ref)

PM = FR.PMlimits()
case_palette = palette(:tab10, max(length(case_keys), 3))
tick_fs = (Plots.default(:tickfontsize) isa Number ? Int(Plots.default(:tickfontsize)) : 8) + 2
guide_fs = (Plots.default(:guidefontsize) isa Number ? Int(Plots.default(:guidefontsize)) : 11) + 2

p_diag = plot(layout=(2, 2), size=(1350, 1100),
    left_margin=14Plots.mm,
    right_margin=10Plots.mm,
    top_margin=8Plots.mm,
    bottom_margin=14Plots.mm,
    tickfontsize=tick_fs,
    guidefontsize=guide_fs)
plot!(p_diag, subplot=1, legend=false)
plot!(p_diag, subplot=3, legend=false)

for sp in 1:4
    plot!(p_diag, subplot=sp, xlims=(minimum(t_ref), maximum(t_ref)))
end

# Top-left: single-particle comparison (Energy)
plot!(p_diag, t_ref, FR.E_fetch_tilde(x_tilde_tau_ref), label="JONSWAP", lw=3, lc=:green, subplot=1)
plot!(p_diag, t_ref, t_ref * 0 .+ PM.E_tilde, label="PW64", lw=3, lc=:black, subplot=1)
plot!(p_diag, PI1D_t_tilde, PI1D["E_tilde"], linestyle=:dash, lw=3, marker=:x, ms=3,
    label="Single Particle 1D", xlabel="t_tilde", ylabel="E_tilde",
    title="Single Particle Comparison (Energy)", subplot=1)
plot!(p_diag, PI2D_t_tilde, PI2D["E_tilde"], lw=3, marker=:circle, ms=2,
    label="Single Particle 2D", subplot=1)

# Top-right: single-particle comparison (Peak frequency)
plot!(p_diag, t_ref, t_ref * 0 .+ PM.f_p_tilde, label="PW64", lw=3, lc=:black, subplot=2)
plot!(p_diag, t_ref, FR.f_p_tilde(fp_JON_t, u10_ref), label="JONSWAP", lw=3, lc=:green,
    ylim=(0.1, 0.4), subplot=2)
plot!(p_diag, PI1D_t_tilde, PI1D["Fp_tilde"], linestyle=:dash, lw=3, marker=:x, ms=3,
    label="Single Particle 1D", xlabel="t_tilde", ylabel="fp_tilde",
    title="Single Particle Comparison (Peak Frequency)", subplot=2)
plot!(p_diag, PI2D_t_tilde, PI2D["Fp_tilde"], lw=3, marker=:circle, ms=2,
    label="Single Particle 2D", subplot=2)

# Bottom-left: periodic case sweep (Energy)
plot!(p_diag, t_ref, FR.E_fetch_tilde(x_tilde_tau_ref), label="JONSWAP", lw=3, lc=:green, subplot=3)
plot!(p_diag, t_ref, t_ref * 0 .+ PM.E_tilde, label="PW64", lw=3, lc=:black, subplot=3)
for case_key in case_keys
    i_case = findfirst(==(case_key), case_keys)
    case_color = case_palette[i_case]
    per_case = DD_PIC_per[case_key]
    t_tilde = per_case["t_tilde"]
    case_label = string("PiCLES ", case_key)
    plot!(p_diag, t_tilde, per_case["E_tilde"], label=case_label,
        xlabel="t_tilde", ylabel="E_tilde", title="Periodic Case Sweep (Energy)",
        lw=2.4, lc=case_color, alpha=0.9, subplot=3)
end

# Bottom-right: periodic case sweep (Peak frequency)
plot!(p_diag, t_ref, t_ref * 0 .+ PM.f_p_tilde, label="PW64", lw=3, lc=:black, subplot=4)
cg_JON_t = FR.c_p_fetch.(x_tilde_tau_ref, u10_ref) / 2
fp_JON_t = get_fp_from_cg(cg_JON_t)
plot!(p_diag, t_ref, FR.f_p_tilde(fp_JON_t, u10_ref), label="JONSWAP", lw=3, lc=:green,
    ylim=(0.1, 0.4), subplot=4)
for case_key in case_keys
    i_case = findfirst(==(case_key), case_keys)
    case_color = case_palette[i_case]
    per_case = DD_PIC_per[case_key]
    t_tilde = per_case["t_tilde"]
    case_label = string("PiCLES ", case_key)
    plot!(p_diag, t_tilde, per_case["Fp_tilde"], label=case_label,
        xlabel="t_tilde", ylabel="fp_tilde", title="Periodic Case Sweep (Peak Frequency)",
        lw=2.4, lc=case_color, alpha=0.9, subplot=4)
end

display(p_diag)

# save figure
savefig(p_diag, joinpath(plot_path_base, "B01_1D_PW_tuning_space_periodic_u$(u10).png"))

# %%

DDcollect["PI1"] = PI1D
DDcollect["PI2"] = PI2D

open(save_path *"PiCLES_v6_Fetch_parsets.json", "w") do f
    write(f, JSON.json(DDcollect))
end

# %%
