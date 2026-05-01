using StaticArrays

using PiCLES.ParticleSystems: particle_waves_v5 as PW
import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleValues, InitParticleInstance
using Oceananigans.Units

using BenchmarkTools
#using Revise

using Profile
# debugging:
using ProfileView

# %%
initParticleDefaults(s::ParticleDefaults) = [s.lne, s.c̄_x, s.c̄_y, s.x, s.y]

U10, V10 = +10.0, +10.0

# version 3
r_g0 = 0.85
# function to define constants for grouwth and dissipation
Const_ID = PW.get_I_D_constant()
#@set Const_ID.γ = 0.88
Const_Scg = PW.get_Scg_constants()



u(x::Float64, y::Float64, t::Float64) = (10 * (t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
v(x::Float64, y::Float64, t::Float64) = -(10 * (t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0

u_num(x::Number, y::Number, t::Number) = (10 * (t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
v_num(x::Number, y::Number, t::Number) = -(10 * (t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
DT = 10minutes

winds = (u=u, v=v)

# define V4 parameters absed on Const NamedTuple:
default_ODE_parameters = (
    r_g=r_g0,
    C_α=Const_Scg.C_alpha,
    C_φ=Const_ID.c_β,
    C_e=Const_ID.C_e,
    g=9.81)
WindSeamin = FetchRelations.get_initial_windsea(u(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), DT / 2)
z0 = initParticleDefaults(ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0))

z0_static = @SVector [-19.500304989027846, 0.00043962455576072634, -0.00043962455576072634, 0.0, 0.0]
z0_mut = @MVector [-19.500304989027846, 0.00043962455576072634, -0.00043962455576072634, 0.0, 0.0]
# %%
Revise.retry()
## non-static version 
@time @allocated particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q, static=false);
@time @allocated particle_system = PW.particle_equations(u_num, u_num, γ=Const_ID.γ, q=Const_ID.q, static=false);
# 01/14/2024: 0.000019 seconds (9 allocations: 272 bytes)

# %% ---------- benchmark speed
@benchmark dz = particle_system(z0_mut, z0_mut, default_ODE_parameters, 1.0)
@benchmark dz = particle_system(z0, z0, default_ODE_parameters, 1.0)
# 01/14/2024
# Memory estimate: 1.56 KiB, allocs estimate: 74.
# (median):     884.333 ns 

# 01/14/2024 No.2 
# Memory estimate: 0 bytes, allocs estimate: 0.
# Time  (median):     81.013 ns 

@profview_allocs particle_system3 = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q) # no result


@allocated z = @SVector [-19.500304989027846, 0.00043962455576072634, -0.00043962455576072634, 0.0, 0.0]
@allocated z0 = initParticleDefaults(ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0))
z0 = initParticleDefaults(ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0))

# costs of calliing PW5
Revise.retry()
@profview_allocs @time @allocated dz = particle_system(z0_mut, z0_mut, default_ODE_parameters, 0.1)
# org:  0.000036 seconds (74 allocations: 1.562 KiB)
# 1st:  0.000007 seconds (3 allocations: 240 bytes)
# 2ns:  0.000005 seconds


@benchmark dz = particle_system(z0, z0, default_ODE_parameters, 0.1)
#  Memory estimate: 0 bytes, allocs estimate: 0.

# %%
dz = particle_system(z0, z0, default_ODE_parameters, 1.0);

@profview_allocs for i in 1:10000
    #@info i
    dz = particle_system(dz, dz, default_ODE_parameters, 0.1)
end
# %%
Profile.clear()
@profile for i in 1:10000
    dz = particle_system(dz, dz, default_ODE_parameters, 0.1)
end

Profile.print()
ProfileView.view(expand_tasks=true, expand_threads=true)


# %% -----------------  individual profiling
x,y, t = 2.0, 3.0, 12.0
@time @allocated v(x, y, t)
@benchmark v(x, y, t)

u1 = u(x, y, t)
v1 = v(x, y, t)
c_gp_x, c_gp_y = 1.2, 4.2
p,q, n = PW.magic_fractions(-1/4.0)

@time @allocated PW.αₚ(u1, v1, c_gp_x, c_gp_y)
@benchmark PW.αₚ(u1, v1, c_gp_x, c_gp_y)

@time @allocated PW.H_β(PW.αₚ(u1, v1, c_gp_x, c_gp_y), p)
@benchmark PW.H_β(PW.αₚ(u1, v1, c_gp_x, c_gp_y), p)
@benchmark PW.Δ_β(PW.αₚ(u1, v1, c_gp_x, c_gp_y) )
Hₚ = PW.H_β(PW.αₚ(u1, v1, c_gp_x, c_gp_y), p)


# %% PW.S_dir
Revise.retry()
#@time @allocated PW.S_dir(u1, v1, c_gp_x, c_gp_y, default_ODE_parameters.C_φ, Hₚ) # 0.000054 seconds (13 allocations: 256 bytes)

@time @allocated PW.α_func(PW.speed(u1, v1), PW.speed(c_gp_x, c_gp_y))
@time @allocated PW.sin2_a_min_b(u1, v1, c_gp_x, c_gp_y)

@profview_allocs for i in 1:10000
    #@info i
    #PW.speed(u1, v1), PW.speed(c_gp_x, c_gp_y)
    #PW.α_func(PW.speed(u1, v1), PW.speed(c_gp_x, c_gp_y))
    PW.S_dir(u1, v1, c_gp_x, c_gp_y, default_ODE_parameters.C_φ, Hₚ) 
end

#@time @allocated PW.α_func(PW.speed(u1, v1), PW.speed(c_gp_x, c_gp_y))

@time @allocated PW.S_dir(u1, v1, c_gp_x, c_gp_y, default_ODE_parameters.C_φ, Hₚ) 
# org: 0.000054 seconds (13 allocations: 256 bytes)
# 1st: 0.000012 seconds (9 allocations: 192 bytes)
# 2nd: 0.000008 seconds (2 allocations: 32 bytes)

# %% c_g_conversions
Revise.retry()

c̄ = PW.speed(c_gp_x, c_gp_y)

# function c_g_conversions(c̄::Float64; r_g::Number=0.9) # this is a slow function
#     c̄ / r_g
# end

function c_g_conversions(c̄::Float64; r_g::Number=0.9) # this is a slow function
    c̄ / r_g
end

function c_g_conversions_vector(c̄::Number; g::Number=9.81, r_g::Number=0.9) # this is a slow function
    c_gp = c_g_conversions(c̄, r_g=r_g)
    kₚ = g / (4.0 * max( c_gp^2, 1e-2))
    ωₚ = g / (2.0 * max( abs(c_gp), 0.1))
    #@SVector [c_gp, kₚ, ωₚ]
    c_gp, kₚ, ωₚ
end

#c_vec = @SVector [c_gp_x, c_gp_y]
@time @allocated cx1 = c_g_conversions(c_gp_x, r_g=r_g0)

@time @allocated xx                = c_g_conversions_vector(abs(c̄), r_g=r_g0)
@time @allocated c_gp_speed, _, _  = c_g_conversions_vector(abs(c̄), r_g=r_g0)
@time @allocated c_gp_speed, kₚ, ωₚ = c_g_conversions_vector(abs(c̄), r_g=r_g0)

@profview_allocs for i in 1:10000
    #@info i
    c_gp_speed, kₚ, ωₚ = c_g_conversions_vector(abs(c̄), r_g=r_g0)
end


@time @allocated c_g_conversions(abs(c̄), r_g=r_g0)
# org: 0.000022 seconds (4 allocations: 128 bytes)
# 1st: 0.000012 seconds (4 allocations: 64 bytes)




# %% ------- Static version --------
Revise.retry()
@time @allocated particle_system_static = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q, static=true);

# %% -- benchmark speed
z0_static = @SVector [-19.500304989027846, 0.00043962455576072634, -0.00043962455576072634, 0.0, 0.0]
z0 = initParticleDefaults(ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0))
z0_mut = @MVector [-19.500304989027846, 0.00043962455576072634, -0.00043962455576072634, 0.0, 0.0]
#z0_mut = convert(MVector{5,Float16}, z0_mut)

@benchmark dz = particle_system_static(z0_mut, default_ODE_parameters, 1.0)
#  Memory estimate: 48 bytes, allocs estimate: 1.
#  Time  (median):     88.140 ns 
# v2: 
#  Memory estimate: 48 bytes, allocs estimate: 1.
#  Time  (median):     68.283 ns 
# %%

@profview_allocs for i in 1:10000
    #@info i
    dz1 = particle_system_static(z0_static, default_ODE_parameters, 0.1)
end
