import Plots

#using PiCLES.ParticleSystems: particle_waves_v3beta as PW3
using PiCLES.ParticleSystems: particle_waves_v5 as PW
import PiCLES: FetchRelations
using Setfield, IfElse

using PiCLES.Operators.core_1D: ParticleDefaults, InitParticleValues, InitParticleInstance, GetVariablesAtVertex
using PiCLES.Operators.core_2D: ParticleDefaults as ParticleDefaults_2D, InitParticleValues as InitParticleValues_2D, InitParticleInstance as InitParticleInstance_2D

using PiCLES.ParticleMesh: OneDGrid, OneDGridNotes

using DifferentialEquations

using PiCLES.Utils.ParticleTools
using Plots

using Oceananigans.Units

using PiCLES: WaveGrowthModels1D, reset_boundary!
using PiCLES.Simulations
using PiCLES.Plotting

using Statistics
using PiCLES: FetchRelations as FR
using JSON


squeeze(a) = dropdims(a, dims=tuple(findall(size(a) .== 1)...))

function get_fetch_variables(data_slice, r_g)
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
Revise.retry()
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)

# @info "org. gamma gamma= " Const_ID.γ
# Const_ID.γ = Const_ID.γ   #ODEpars.r_g^2
# @info "gamma= " Const_ID.γ
# PW.e_T_func(Const_ID.γ, Const_ID.p, Const_ID.q, Const_ID.n; C_e=Const_ID.C_e, c_e=Const_ID.c_e, c_α=Const_ID.c_alpha)^2
# ODEpars.C_φ
# ODEpars.C_e

ODEpars
Const_ID
Const_Scg

c_D, c_β, c_e, c_alpha, r_w, C_e, γ, p, q, n =Const_ID.c_D, Const_ID.c_β, Const_ID.c_e, Const_ID.c_alpha, Const_ID.r_w, Const_ID.C_e, Const_ID.γ, Const_ID.p, Const_ID.q, Const_ID.n

#PW.e_T_func(γ, p, q, n; C_e=Const_ID.C_e, c_e=Const_ID.c_e, c_α=Const_ID.c_alpha)^2


# e_T_func(γ::Float64, p::Float64, q::Float64, n::Float64; c_β::Number=2.16e-4, c_D::Number=2e-3, c_e::Float64=1.3e-6, c_α::Float64=11.8) = sqrt(c_e * c_α^(-p / q) / (γ * c_β * c_D )^(1 / n))

# e_T_func(γ, p, q, n, c_β=Const_ID.c_β, c_D=Const_ID.c_D, c_e=Const_ID.c_e, c_α=Const_ID.c_alpha)^2

# r_w / ODEpars.r_g

# γ
# 1- γ
# r_g = 0.87

# (p - q) * c_alpha^(-4) * r_g / (2 * r_w * c_β * c_D)


DDcollect = Dict()

# typeof(ODEpars)
T           = 2.5day

# loop over u10= 5:5:20, DT = 5,10,20,30,60 minutes, Nx = 21, 51, 101, 201

DD_PIC_nonper = Dict()
DD_PIC_per    = Dict()

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

    grid1d = OneDGrid(0, 1e7, Nx)
    grid1d.dx

    Case_dict = Dict("u10" => u10, "DT" => DT, "dx" => grid1d.dx)
    #make string from Case_dict, convert to integers
    Case_str = join(["$(k):$(round(Int, v))" for (k, v) in Case_dict], "_")

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
        save_everystep=false,
        dt=1e-3, #60*10, 
        dtmin=1e-9, #60*5, 
    )


    # define model -
    u(x, t) = x .* 0 + t * 0 + u10
    # redefine model 
    wave_model = WaveGrowthModels1D.WaveGrowth1D(; grid=grid1d,
        winds=u,
        ODEsys=PW.particle_equations(u, γ=Const_ID.γ, q=Const_ID.q, IDConstants=Const_ID),
        ODEsets=ODE_settings,  # ODE_settings
        ODEinit_type="wind_sea",  # ODEpars
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
    data_slice  = squeeze(maximum(data.data[end-6:1:end, :, :], dims=1))[2:end-1, :]
    PiCLES_data = get_fetch_variables(data_slice, ODEpars.r_g)

    PIC_nonper = get_non_dim_data(PiCLES_data, u10, data.x[2:end-1], data.time[2:end-1])
    DD_PIC_nonper[Case_str] = convert_picles_nondim_to_savedict(PIC_nonper, "PiCLES_nonper", attrs=Case_dict)

    ### E tilde time
    data_periodic = Simulations.convert_store_to_tuple(wave_simulation_periodic.store, wave_simulation_periodic)
    data_periodic_slice = squeeze(mean(data_periodic.data[:, :, :], dims=2))
    PiCLES_data_periodic = get_fetch_variables(data_periodic_slice, ODEpars.r_g)

    PIC_per = get_non_dim_data(PiCLES_data_periodic, u10, data_periodic.x, data_periodic.time)
    DD_PIC_per[Case_str] = convert_picles_nondim_to_savedict(PIC_per, "PiCLES_per", attrs=Case_dict)

end


DDcollect["PIC_nonper"] = DD_PIC_nonper
DDcollect["PIC_per"]    = DD_PIC_per

# %
x_tilde = PIC_nonper.x_tilde
#JONSWAP
E_jon_tilde = FR.E_fetch_tilde(x_tilde) 
# PM 64
PM          = FR.PMlimits()


pE_tilde_pic = plot(x_tilde, PIC_nonper.E_tilde,
    label="PiCLES", xlabel="x_tilde", ylabel="E_tilde", 
    title="Non-dimensional Energy\nu10 = $(u10)",
    lw=2, lc=:blue)

plot!(pE_tilde_pic, x_tilde, FR.E_fetch_tilde(x_tilde), label="JONSWAP", lw=2, lc=:green)
plot!(pE_tilde_pic, x_tilde, x_tilde * 0 .+ PM.E_tilde, label="PW64",    lw=3, lc=:black)


t_tilde = PIC_per.t_tilde
x_tilde_tau = FR.X_tilde_from_tau.(t_tilde)

# JONSWAP time
E_jon_tilde_tau = FR.E_fetch_tilde(x_tilde_tau)

pE_tilde_pic_time = plot(t_tilde, PIC_per.E_tilde,
    label="PiCLES", xlabel="t_tilde", ylabel="E_tilde",
    title="Non-dimensional Energy",
    lw=2, lc=:blue)
plot!(pE_tilde_pic_time, t_tilde, FR.E_fetch_tilde(x_tilde_tau), label="JONSWAP", lw=2, lc=:green)
plot!(pE_tilde_pic_time, t_tilde, t_tilde * 0 .+ PM.E_tilde,     label="PW64",    lw=3, lc=:black)


#### F_p plot
pFp_tilde_pic = plot(x_tilde, x_tilde * 0 .+ PM.f_p_tilde, label="PW64", 
    lw=3, lc=:black,
    ylim=(0.1, 0.4), # xlims=(1e3, 1e5),
    xlabel="x_tilde", ylabel="fp_tilde", title="Non-dimensional peak frequecy")


plot!(pFp_tilde_pic, x_tilde, PIC_nonper.Fp_tilde,
    label="PiCLES",
    lw=2, lc=:blue) #, ylim = (0, 20))
#pFp_tilde_pic = plot(x_tilde, FR.fₘ_from_X_tilde.(u10, x_tilde), label="PiCLES")

cg_JON = FR.c_p_fetch.(x_tilde, u10) /2  # cg
fp_JON = get_fp_from_cg(cg_JON)
plot!(pFp_tilde_pic, x_tilde, FR.f_p_tilde(fp_JON, u10), label="JONSWAP", lw=2, lc=:green)

# F_p time plot
pFp_tilde_time_pic = plot(t_tilde, t_tilde * 0 .+ PM.f_p_tilde, label="PW64", 
    lw=3, lc=:black,
    ylim=(0.1, 0.4), 
    xlabel="t_tilde", ylabel="fp_tilde", title="Non-dimensional peak frequecy")

plot!(pFp_tilde_time_pic, t_tilde, PIC_per.Fp_tilde,
    label="PiCLES",
    lw=2, lc=:blue)

cg_JON = FR.c_p_fetch.(x_tilde_tau, u10) / 2  # cg
fp_JON = get_fp_from_cg(cg_JON)
plot!(pFp_tilde_time_pic, t_tilde, FR.f_p_tilde(fp_JON, u10), label="JONSWAP", lw=2, lc=:green)

# single particle
xx                         = OneDGridNotes(wave_model.grid).x[1]
WindSeamin                 = FetchRelations.get_initial_windsea(u10, 0,  DT)
uu                         = wave_model.winds
ODE_settings               = wave_model.ODEsettings
particle_defaults          = ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar"], 0.0)
ParticleState, particle_on = InitParticleValues(particle_defaults, xx, uu(xx, 0), DT)
PI4                        = InitParticleInstance(wave_model.ODEsystem, ParticleState, ODE_settings, 1, false, particle_on)


step!(PI4.ODEIntegrator, T, true)
PI = ParticleTools.FormatParticleData(PI4)

PI1D_x_tilde = FR.X_tilde(PI.x, u10)
PI1D_t_tilde = FR.t_tilde(PI.time, u10)
PI1D= Dict(   "x_tilde" => PI1D_x_tilde, 
        "t_tilde" => PI1D_t_tilde, 
        "E_tilde" => FR.E_tilde(PI.E, u10),
        "Fp_tilde" => FR.f_p_tilde(get_fp_from_cg(PI.cgx / ODEpars.r_g), u10), 
        "label" => "Single Particle 1D",
        "attrs" => nothing)


plot!(pE_tilde_pic, PI1D_x_tilde, PI1D["E_tilde"], linestyle=:dash, lw=3, label="Single Particle 1D")

plot!(pFp_tilde_pic, PI1D_x_tilde, PI1D["Fp_tilde"], linestyle=:dash, lw=3, label="Single Particle 1D")

# add to Energy - time plot
plot!(pE_tilde_pic_time, PI1D_t_tilde, PI1D["E_tilde"], linestyle=:dash, lw=3, label="Single Particle 1D")

# add to F_p - time plot
plot!(pFp_tilde_time_pic, PI1D_t_tilde, PI1D["Fp_tilde"], linestyle=:dash, lw=3, label="Single Particle 1D")


# % single particle 2D
u2(x, y, t) = x .* 0 + t * 0 + u10
v2(x, y, t) = x .* 0 + t * 0 + y *0 
WindSeamin                 = FetchRelations.get_initial_windsea(u10, 0,  DT)
particle_defaults          = ParticleDefaults_2D(log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0)
ParticleState, particle_on = InitParticleValues_2D(particle_defaults, (xx, xx), (u2(xx, xx, 0),0 ), DT)
ODESystem2D                = PW.particle_equations(u2, v2, γ=Const_ID.γ, q=Const_ID.q, IDConstants=Const_ID)
PI5                        = InitParticleInstance_2D(ODESystem2D, ParticleState, ODE_settings, (1, 1), false, particle_on)

step!(PI5.ODEIntegrator, T, true)
PI2 = ParticleTools.FormatParticleData(PI5)

PI2D_x_tilde = FR.X_tilde(PI2.x, u10)
PI2D_t_tilde = FR.t_tilde(PI2.time, u10)
PI2D= Dict(   "x_tilde" => PI2D_x_tilde, 
        "t_tilde" => PI2D_t_tilde, 
        "E_tilde" => FR.E_tilde(PI2.E, u10),
        "Fp_tilde" => FR.f_p_tilde(get_fp_from_cg(PI2.cgx / ODEpars.r_g), u10), 
        "label" => "Single Particle 2D",
        "attrs" => nothing)


plot!(pE_tilde_pic, PI2D_x_tilde, PI2D["E_tilde"], label="Single Particle 2D")

plot!(pFp_tilde_pic, PI2D_x_tilde, PI2D["Fp_tilde"], label="Single Particle 2D")

# add to Energy - time plot
plot!(pE_tilde_pic_time, PI2D_t_tilde, PI2D["E_tilde"], label="Single Particle 2D")

# add to F_p - time plot
plot!(pFp_tilde_time_pic, PI2D_t_tilde, PI2D["Fp_tilde"], label="Single Particle 2D")


plot(pE_tilde_pic, pFp_tilde_pic, pE_tilde_pic_time, pFp_tilde_time_pic, layout=(2, 2), size=(1000, 800))

# save figure
savefig(joinpath(plot_path_base, "B01_1D_PW_tuning_space_u$(u10).png"))

# %%

DDcollect["PI1"] = PI1D
DDcollect["PI2"] = PI2D

open(save_path *"PiCLES_v5_Fetch_parsets.json", "w") do f
    write(f, JSON.json(DDcollect))
end

# %%
