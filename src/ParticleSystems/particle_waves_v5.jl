module particle_waves_v5

using DifferentialEquations, IfElse

using ...Architectures: AbstractODESettings, AbstractParticleSystem, IDConstantsInstance, ScgConstantsInstance

export particle_equations, ODESettings, ODEParameters
using LinearAlgebra
using StaticArrays

using Parameters
using DocStringExtensions
#export t, x, y, e, c̄, φ_p, dist, Gₙ, u_10

# #using Plots
# #using PyPlot
#
# using PyCall
# #pygui(gui) #:tk, :gtk3, :gtk, :qt5, :qt4, :qt, or :wx
# using PyPlot

# startupfile = joinpath(pwd(), "2022_particle_waves_startup.jl")
# isfile(startupfile) && include(startupfile)

######


"""
ODESettings
Structure to hold all information about the ODE system
# Fields  
$(DocStringExtensions.FIELDS)
"""
@with_kw mutable struct ODESettings <: AbstractODESettings
    "ODE parameters (Dict)"
    Parameters::NamedTuple
    "minimum allowed log energy on particle "
    log_energy_minimum::Float64
    "maximum allowed log energy on particle "
    log_energy_maximum::Float64 = log(17)

    "minimum needed squared wind velocity to seed particle"
    wind_min_squared::Float64 = 4.0
    "solver method for ODE system"
    #alternatives
    #Rosenbrock23(), AutoVern7(Rodas4()) ,AutoTsit5(Rosenbrock23()) , Tsit5()
    solver::Any = AutoTsit5(Rosenbrock23())
    "Internal saving timestep of the ODEs"
    saving_step::Float64
    "remeshing time step, i.e. timestep of the model"
    timestep::Float64

    "Absolute allowed error"
    abstol::Float64 = 1e-4
    "relative allowed error"
    reltol::Float64 = 1e-3
    "max iteration for ODE solver (1e4)"
    maxiters::Int = 1e4
    "Adaptive timestepping for ODE (true)"
    adaptive::Bool = true
    "timestep (if adaptive is false this is used), if adaptive is true this is the initial timestep"
    dt::Float64 = 60 * 6 # seconds
    "min timestep (if adaptive is true)"
    dtmin::Float64 = 60 * 5 # seconds
    "force min timestep (if adaptive is true)"
    force_dtmin::Bool = false

    "Total time of the ODE integration, should not be needed, this is problematic .. "
    total_time::Float64

    "Callback function for ODE solver"
    callbacks::Any = nothing
    "save_everystep (false)"
    save_everystep::Bool = false
end



# ------------------------------------------------------
# Paramter functions
# ------------------------------------------------------

"""
This function returns the universal exponent relations
[p, q, n] = magic_fractions(q::Float64=-1/4.0)
"""
function magic_fractions(q::Float64=-1 / 4.0)
    # returns universal exponent relations
    p = (-1 - 10 * q) / 2
    n = 2 * q / (p + 4 * q)
    [p, q, n]
end

"""
IDConstants(; c_D=2e-3, c_β=4e-2, c_e=1.3e-6, c_alpha= 11.8 , r_w = 2.35, q=-1/4)
this function returns a mutable struct with the constants for the growth and dissipation
inputs:
    c_D: drag coefficient
    c_β:    
    c_e:    
    c_alpha: 
    r_w:    
    q:
returns:
    c_D, c_β, c_e, c_alpha, r_w, C_e, γs, p, q, n    
"""
mutable struct IDConstants{PP} <: IDConstantsInstance #where {PP<:Number}
    c_D::PP
    c_β::PP
    c_e::PP
    c_alpha::PP
    r_w::PP
    C_e::PP
    γ::PP
    p::PP
    q::PP
    n::PP
end

function IDConstants(; r_g=0.85, c_D=2e-3, c_β=4e-2, c_e=1.3e-6, c_alpha=11.8, r_w=2.35, q=-1 / 4) #c_alpha changed from 11.8 to 14.5 #
    p = (-1 - 10 * q) / 2
    n = 2 * q / (p + 4 * q)

    C_e = r_w * c_β * c_D /r_g
    #γ = (p - q) * c_alpha^(-4) / (C_e * 2)
    γ = 1 - (p - q) / (c_alpha^4 * C_e * 2)
    return IDConstants(c_D, c_β, c_e, c_alpha, r_w, C_e, γ, p, q, n)
end


function Base.show(io::IO, ow::IDConstantsInstance)

    print(io, "IDConstants ", "\n",
        "├── c_D: ", ow.c_D, "\n",
        "├── c_β: ", ow.c_β, "\n",
        "├── c_e: ", ow.c_e, "\n",
        "├── c_alpha: ", ow.c_alpha, "\n",
        "├── r_w: ", ow.r_w, "\n",
        "├── C_e: ", ow.C_e, "\n",
        "├── γ: ", ow.γ, "\n",
        "├── p: ", ow.p, "\n",
        "├── q: ", ow.q, "\n",
        "└── n: ", ow.n, "\n")
end


"""
    ScgConstants(C_alpha=-1.41, C_varphi  =1.81e-5)

    this function returns a NamedTuple with constants for peak frequency shift.
        C_alpha: 1.41   # constant for peak frequency shift (?)
        C_varphi: 1.81e-5 # constant for peak frequency shift (?)
"""
mutable struct ScgConstants{PP} <: ScgConstantsInstance # where {PP<:Union{Float64,Float16,Int}}
    C_alpha::PP
    C_varphi::PP
end


function ScgConstants(; C_alpha=-1.41, C_varphi=1.81e-5)
    return ScgConstants(C_alpha, C_varphi)
end


function Base.show(io::IO, ow::ScgConstantsInstance)

    print(io, "ScgConstants ", "\n",
        "├── C_alpha: ", ow.C_alpha, "\n",
        "└── C_varphi: ", ow.C_varphi, "\n")
end


# need to be reomoved later
get_I_D_constant = IDConstants
get_Scg_constants = ScgConstants



"""
ODEParameters(; r_g=0.85,  q= -0.25)
    wrapper function to define constants and parameters for the ODE system

"""
function ODEParameters(; r_g=0.85, q=-0.25, g=9.81)
    Const_ID = IDConstants(;r_g=r_g, q = q)
    Const_Scg = ScgConstants()

    parset = (
        r_g=r_g,
        C_α=Const_Scg.C_alpha,
        C_φ=Const_Scg.C_varphi,
        C_e=Const_ID.C_e,
        g=g)

    return parset, Const_ID, Const_Scg
end


"""
        αₚ(α::Number, φ::Number, φ_w::Number)
        αₚ(α::Number, cφ_p::Number, sφ_p::Number, cφ_w::Number, sφ_w::Number)
        αₚ(u::NamedTuple, cg::NamedTuple)
        αₚ(u::NamedTuple,  cx::Number, cy::Number )

        returns angle between wave propagation direction and particle orientation.
        cg, cx, and cy are the peak wave directions (!), not the mean wave direction!
"""
# αₚ(α::Number, φ::Number, φ_w::Number) = cos.(φ .- φ_w) .* α
#αₚ(α::Number, cφ_p::Number, sφ_p::Number, cφ_w::Number, sφ_w::Number) = (cφ_p .* cφ_w + sφ_p .* sφ_w) .* α
αₚ(u::NamedTuple, cg::NamedTuple) = αₚ(u.u, u.v, cg.cx, cg.cy)
αₚ(u::NamedTuple, cx::Number, cy::Number) = αₚ(u.u, u.v, cx, cy)
αₚ(u::Number, v::Number, cx::Number, cy::Number) = @. (u .* cx + v .* cy) ./ (2 .* max(speed(cx, cy), 1e-4)^2)

#α_func(u_speed::Number, c_gp_speed::Number) = @. min( u_speed / (2.0 * c_gp_speed), 500)
function α_func(u_speed::Float64, c_gp_speed::Float64)
    a = @. u_speed / (2.0 * c_gp_speed)
    return IfElse.ifelse( a > 500, 500, a)
    #return IfElse.ifelse(u_speed / (2.0 * c_gp_speed) > 500, 500, u_speed / (2.0 * c_gp_speed))
    # if u_speed / (2.0 * c_gp_speed) > 500
    #     return 500.0
    # else
    #     return u_speed / (2.0 * c_gp_speed)
    # end
    # return min(u_speed / (2.0 * c_gp_speed), 500)::Float64
end

function α_func(u_speed, c_gp_speed)
    a = @. u_speed / (2.0 * c_gp_speed)
    return IfElse.ifelse(a > 500, 500, a)
    #return IfElse.ifelse(u_speed / (2.0 * c_gp_speed) > 500, 500, u_speed / (2.0 * c_gp_speed))
    # if u_speed / (2.0 * c_gp_speed) > 500
    #     return 500.0
    # else
    #     return u_speed / (2.0 * c_gp_speed)
    # end
    # return min(u_speed / (2.0 * c_gp_speed), 500)::Float64
end

#sin2_a_min_b(ca::Number, sa::Number, cb::Number, sb::Number) =  4 * sb * cb * (sa^2 -0.5) - 4 * sa * ca * (sb^2 -0.5)
# sign(cx) *
#sin2_a_min_b(ux::Number, uy::Number, cx::Number, cy::Number) = @. (2 / (speed(ux, uy) * speed(cx, cy))^2) * (ux * uy * (2 * cy^2 - speed(cx, cy)^2) - cx * cy * (2 * uy^2 - speed(ux, uy)^2))
function sin2_a_min_b(ux::Number, uy::Number, cx::Number, cy::Number)

    IfElse.ifelse(
        (speed(ux, uy) * speed(cx, cy)) == 0,
        0,
        @. (2 / (speed(ux, uy) * speed(cx, cy))^2) * (ux * uy * (2 * cy^2 - speed(cx, cy)^2) - cx * cy * (2 * uy^2 - speed(ux, uy)^2))
    )
end

function sin2_a_min_b(u::NamedTuple, cx::Number, cy::Number)
    sin2_a_min_b(u.u, u.v, cx, cy)
end
function sin2_a_min_b(u::NamedTuple, c::NamedTuple)
    sin2_a_min_b(u.u, u.v, c.cx, c.cy)
end

sin2_a_plus_b(ux::Number, uy::Number, cx::Number, cy::Number) = (2 / (speed(ux, uy) * speed(cx, cy))^2) * (cx * uy + cy * ux) * (cx * ux - cy * uy)
function sin2_a_plus_b(u::NamedTuple, cx::Number, cy::Number)
    sin2_a_plus_b(u.u, u.v, cx, cy)
end
function sin2_a_plus_b(u::NamedTuple, c::NamedTuple)
    sin2_a_plus_b(u.u, u.v, c.cx, c.cy)
end


# old version
#e_T_func(γ::Float64, p::Float64, q::Float64, n::Float64; C_e::Number=2.16e-4, c_e::Float64=1.3e-6, c_α::Float64=11.8) = sqrt(c_e * c_α^(-p / q) / (γ * C_e)^(1 / n))

# eq. A2.4 in Kudr. 2021 2D
e_T_func(γ::Float64, p::Float64, q::Float64, n::Float64; c_β::Number=2.16e-4, c_D::Number=2e-3, c_e::Float64=1.3e-6, c_α::Float64=11.8) = sqrt(c_e * c_α^(-p / q) / (γ * c_β * c_D)^(1 / n))


H_β(α::Number, p::Float64; α_thresh::Float64=0.85)  = @. 0.5 .* (1.0 + tanh.(p .* (α .- α_thresh)))
Δ_β(α::Number; α_thresh::Float64=0.85)              = @. (1.0 .- 1.25 .* sech.(10.0 .* (α .- α_thresh)) .^ 2)

"""
function c_g_conversions_vector(c̄::Number; g::Number=9.81, r_g::Number=0.9)
returns a vecotr with conversions between c̄, c_gp, kₚ, and ωₚ
"""
function c_g_conversions_vector(c̄::Number; g::Number=9.81, r_g::Number=0.85) # this is a slow function
    c_gp = c_g_conversions(c̄, r_g=r_g)
    kₚ = g / (4.0 * max(c_gp^2, 1e-2))  # < ---------- the power is slow 
    ωₚ = g / (2.0 * max(abs(c_gp), 0.1))
    #@SVector [c_gp, kₚ, ωₚ]
    c_gp, kₚ, ωₚ
end

function c_g_conversions(c̄::Float64; r_g::Number=0.9) # this is a slow function
    c̄ / r_g
end

function c_g_conversions(c̄; r_g::Number=0.9) # this is a slow function
    c̄ / r_g
end

speed(cx::Number, cy::Number) = @. sqrt(cx^2 + cy^2)

function speed_and_angles(cx::Number, cy::Number)
    #sqrt(cx.^2 + cy.^2), cx ./ sqrt(cx.^2 + cy.^2), cy ./sqrt(cx.^2 + cy.^2)
    c = sqrt.(cx .^ 2 .+ cy .^ 2)
    c, cx ./ c, cy ./ c
end

function speed_and_angles(cx, cy)
    #sqrt(cx.^2 + cy.^2), cx ./ sqrt(cx.^2 + cy.^2), cy ./sqrt(cx.^2 + cy.^2)
    c = sqrt.(cx .^ 2 .+ cy .^ 2)
    c, cx ./ c, cy ./ c
end


# -------------------------------
# Forcing functions
# -------------------------------

# wind input
function Ĩ_func(alpha::Number, Hₚ::Number, C_e::Number)
    # non-dimensional Wind energy input
    # eq sec. 1.3 in the manual
    C_e * Hₚ * alpha^2
end

# Dissipation
function D̃_func_e(e::Number, kₚ::Number, e_T::Number, n::Float64)
    # non-dimensional Wind energy input
    # eq sec. 1.3 in the manual
    e .^ n .* (kₚ ./ e_T) .^ (2 * n)
end

# disspation lne version 
function D̃_func_lne(lne::Number, kₚ::Number, e_T::Number, n::Float64)
    # non-dimensional Wind energy input
    # eq sec. 1.3 in the manual
    exp(n .* lne) .* (kₚ ./ e_T) .^ (2 * n)
end


# peak downshift
# C_α is negative in Kudravtec definition, here its a positive value
S_cg(lne::Number, Δₚ::Number, kₚ::Number, C_α::Number) = @. C_α * Δₚ * kₚ^4 * exp(2 * lne)
#S_cg(lne::Number, Δₚ::Number, kₚ::Number, C_α::Number) = @. C_α * kₚ^4 * exp(2 * lne)


# Peak direction shift
function S_dir(u::Number, v::Number, cx::Number, cy::Number, C_φ::Number, Hₚ::Number)
    return α_func(speed(u, v), speed(cx, cy))^2 * C_φ * Hₚ * sin2_a_min_b(u, v, cx, cy)#::Number
    #alpha^2 * C_φ * Hₚ * sin2_a_min_b(cx, cy, u.u, u.v)

    #alpha^2 * C_φ * sin2_a_min_b(u, cx, cy)
    #alpha^2 * C_φ * Hₚ * sin2_a_plus_b(u, cx, cy)
end


# function Gₙ( dphi_p::Number, dn::Number, dn_0::Number; dg_ratio::Float64 = 0.21 )
#         return (dphi_p / dn_0) * ( (dn /dn_0) / (  (dn/dn_0)^2  + (dg_ratio/2.0)^2 ) )
# end

# tΔφ_w_func(c_x::Number, c_y::Number, u_x::Number, v_y::Number) =   (c_x .* v_y - c_y .* u_x ) / ( c_x .* u_x + c_y .* v_y )
# Δφ_p_RHS(tΔφ_w::Number, tΔφ_p::Number , T_inv::Number)        =   ( tΔφ_w -  tΔφ_p ) * T_inv / (  1 + tΔφ_w * tΔφ_p )


"""
particle_equations(u , v ; γ::Number=0.88, q::Number=-1/4.0 )
Particle wave equations in 2D
inputs:
    u: zonal wind forcing field
    v: meridional wind forcing field
    γ: 0.88
    q: -1/4.0
    propagation: true
    input: true
    dissipation: true
    peak_shift: true
    direction: true
    debug_output: false

return an ODE system as function particle_system(dz, z, params, t) that provides 5 element state vector tendency of:
        z = [lne, c̄_x, c̄_y, x, y]
        params can be a named tuple with the parameters or a vector
        params = [r_g, C_α, g, C_e] 
"""
function particle_equations(u_wind, v_wind; γ::Number=0.88, q::Number=-1 / 4.0, IDConstants=IDConstants(),
    propagation=true,
    input=true,
    dissipation=true,
    peak_shift=true,
    direction=true,
    debug_output=false,
    static= false,
    )
    #t, x, y, c̄_x, c̄_y, lne, Δn, Δφ_p, r_g, C_α, C_φ, g, C_e = init_vars()

    p, q, n = magic_fractions(q)
    e_T = e_T_func(γ, p, q, n, c_β=IDConstants.c_β, c_D=IDConstants.c_D, c_e=IDConstants.c_e, c_α=IDConstants.c_alpha)

    if static
        
        function particle_system_static(z, params, t)

            
            # forcing fields
            #lne, c̄_x, c̄_y, x, y = z

            r_g, C_α, C_e, C_φ = params.r_g, params.C_α, params.C_e, params.C_φ
            # add projection matrix
            M = haskey(params, :M) ? params.M : [1 0 ; 0 1]
            x_lat = haskey(params, :x) ? params.x : x
            y_lat = haskey(params, :y) ? params.y : y
            PropagationCorrection = haskey(params, :PC) ? params.PC : x -> 0.0

            #u = (u=u, v=v)::NamedTuple{(:u, :v),Tuple{Number,Number}}
            u = u_wind(x_lat, y_lat, t)#::Number
            v = v_wind(x_lat, y_lat, t)#::Number
            # u = u_wind(z[4], z[5], t)#::Number
            # v = v_wind(z[4], z[5], t)#::Number

            c̄ = speed(z[2], z[3])
            u_speed = speed(u, v)

            # peak parameters
            c_gp_speed, kₚ, ωₚ = c_g_conversions_vector(abs(c̄), r_g=r_g)
            c_gp_x = c_g_conversions(z[2], r_g=r_g)
            c_gp_y = c_g_conversions(z[3], r_g=r_g)

            # direction equations
            α = α_func(u_speed, c_gp_speed)
            Hₚ = H_β(αₚ(u, v, c_gp_x, c_gp_y), p)
            Δₚ = Δ_β(αₚ(u, v, c_gp_x, c_gp_y))

            # Source terms
            Ĩ = input ? Ĩ_func(α, Hₚ, C_e) : 0.0
            D̃ = dissipation ? D̃_func_lne(z[1], kₚ, e_T, n) : 0.0
            S_cg_tilde = peak_shift ? S_cg(z[1], Δₚ, kₚ, C_α) : 0.0
            S_dir_tilde = direction ? S_dir(u, v, c_gp_x, c_gp_y, C_φ, Hₚ) : 0.0

            # apply great circle correction for spherical coorindates
            #S_dir_tilde = S_dir_tilde - PropagationCorrection(c̄_x)
            S_sphere_tilde = PropagationCorrection(c̄_x)

            # propagration projections
            c̄_projected = propagation ? M * [c̄_x, c̄_y] : [0.0, 0.0]

            z = @SVector [
                # energy
                +ωₚ .* r_g .* S_cg_tilde + ωₚ .* (Ĩ - D̃),

                # peak group velocity vector
                -z[2] .* ωₚ .* r_g .* S_cg_tilde + z[3] .* S_dir_tilde + z[3] .* S_sphere_tilde,
                -z[3] .* ωₚ .* r_g .* S_cg_tilde - z[2] .* S_dir_tilde + z[2] .* S_sphere_tilde,

                # D(z[2]) ~ -z[2] .* ωₚ .* r_g .* S_cg_tilde + (z[3] + 0.001) .* S_dir_tilde, #* (-1),
                # D(z[3]) ~ -z[3] .* ωₚ .* r_g .* S_cg_tilde - (z[2]  + 0.001) .* S_dir_tilde, #* (1),

                # propagation
                propagation ? c̄_projected[1] : 0.0,
                propagation ? c̄_projected[2] : 0.0
            ]

            if debug_output
                additional_output = @SVector [
                    Ĩ,
                    -D̃,
                    r_g * S_cg_tilde,
                    Hₚ,
                    S_dir_tilde,
                    Δₚ,
                    c_gp_y
                ]
                z = vcat(z, additional_output)
            end

            return z

        end
        return particle_system_static

    else

        function particle_system(dz, z, params, t)#::MVector{5, Number}

            # forcing fields
            #u = (u=u(x, y, t), v=v(x, y, t))::NamedTuple{(:u, :v),Tuple{Number,Number}}
            lne, c̄_x, c̄_y, x, y = z

            r_g, C_α, C_e, C_φ = params.r_g, params.C_α, params.C_e, params.C_φ

            # add projection matrix
            M     = haskey(params, :M) ? params.M : [1 0; 0 1] # Projection matrix dependent on GridType
            x_lat = haskey(params, :x) ? params.x : x  # latitude in degrees of the Particle instance
            y_lat = haskey(params, :y) ? params.y : y  # longitude in degrees of the Particle instance
            PropagationCorrection = haskey(params, :PC) ? params.PC : x -> 0.0 # Great circle correction for specific grid types  <--------- test this!

            #u = (u=u, v=v)::NamedTuple{(:u, :v),Tuple{Number,Number}}
            u                 = u_wind(x_lat, y_lat, t)#::Number
            v                 = v_wind(x_lat, y_lat, t)#::Number

            # @info "x_lat: $x_lat, y_lat: $y_lat"
            # @info "x_lat: $(round(x_lat, digits=2)), y_lat: $(round(y_lat, digits=2)), u: $(round(u, digits=2)), v: $(round(v, digits=2))"
            
            c̄                 = speed(c̄_x, c̄_y)
            u_speed           = speed(u, v)

            # peak parameters
            c_gp_speed, kₚ, ωₚ = c_g_conversions_vector(abs(c̄), r_g=r_g) 
            c_gp_x             = c_g_conversions(c̄_x, r_g=r_g)
            c_gp_y             = c_g_conversions(c̄_y, r_g=r_g)

            # direction equations
            α = α_func(u_speed, c_gp_speed)
            Hₚ = H_β(αₚ(u, v, c_gp_x, c_gp_y), p)
            Δₚ = Δ_β(αₚ(u, v, c_gp_x, c_gp_y)) ## <--- this one is slow!!

            # Source terms
            Ĩ           = input ? Ĩ_func(α, Hₚ, C_e) : 0.0
            D̃           = dissipation ? D̃_func_lne(lne, kₚ, e_T, n) : 0.0
            S_cg_tilde  = peak_shift ? S_cg(lne, Δₚ, kₚ, C_α) : 0.0
            S_dir_tilde = direction ? S_dir(u, v, c_gp_x, c_gp_y, C_φ, Hₚ) : 0.0

            # apply great circle correction for spherical coorindates
            #S_dir_tilde = S_dir_tilde - PropagationCorrection(c̄_x)
            S_sphere_tilde = PropagationCorrection(c̄_x)
            #@info "Propagtion direction test:", haskey(params, :PC), PropagationCorrection(c̄_x) # for testing projection kernal Correction
            
            #particle_equations::Vector{Number} = 
            # energy
            dz[1] = +ωₚ .* r_g .* S_cg_tilde + ωₚ .* (Ĩ - D̃) #- c̄ .* G_n,

            # peak group velocity vector
            dz[2] = -c̄_x .* ωₚ .* r_g .* S_cg_tilde + c̄_y .* S_dir_tilde + c̄_y .* S_sphere_tilde
            dz[3] = -c̄_y .* ωₚ .* r_g .* S_cg_tilde - c̄_x .* S_dir_tilde - c̄_x .* S_sphere_tilde

            # D(c̄_x) ~ -c̄_x .* ωₚ .* r_g .* S_cg_tilde + (c̄_y + 0.001) .* S_dir_tilde, #* (-1),
            # D(c̄_y) ~ -c̄_y .* ωₚ .* r_g .* S_cg_tilde - (c̄_x  + 0.001) .* S_dir_tilde, #* (1),

            # propagation
            dz[4:5] = propagation ? M * [c̄_x, c̄_y] : [0.0, 0.0]
            # dz[5] = propagation ? c̄_y : 0.0


            if debug_output
                additional_output = [
                    Ĩ,
                    -D̃,
                    r_g * S_cg_tilde,
                    #alpha_p ~ αₚ(u, c_gp_x, c_gp_y),
                    Hₚ,
                    #alpha ~ α,
                    S_dir_tilde,
                    Δₚ,
                    c_gp_y]
                append!(dz, additional_output)
            end

            return dz

        end
        
        return particle_system

    end


end

# ------------ 1D ------------
"""
particle_equations(u ; γ::Number=0.88, q::Number=-1/4.0 )
Particle wave equations in 2D
inputs:
    u: forcing field
    γ: 0.88
    q: -1/4.0
    propagation: true
    input: true
    dissipation: true
    peak_shift: true
    info: false

returns an ODE system as function particle_system(dz, z, params, t) that provides 3 element state vector tendency of:
        z = [lne, c̄_x, x]
        params can be a named tuple with the parameters or a vector
        params = [r_g, C_α, g, C_e]
"""
function particle_equations(u_wind; γ::Number=0.88, q::Number=-1 / 4.0, IDConstants=IDConstants(),
    propagation=true,
    input=true,
    dissipation=true,
    peak_shift=true, info=false)

    #t, x, c̄_x, lne, r_g, C_α, g, C_e = init_vars_1D()

    # Define basic constants for wave equation (invariant throughout the simulation)
    p, q, n = magic_fractions(q)
    e_T = e_T_func(γ, p, q, n, c_β=IDConstants.c_β, c_D=IDConstants.c_D, c_e=IDConstants.c_e, c_α=IDConstants.c_alpha)

    ###  ---------------- start function here
    function partice_system(dz, z, params, t) #<: Vector{Number}
        #unpack0
        lne, c̄_x, x      = z
        #lne, c̄_x, x      = z.lne, z.c̄_x, z.x
        r_g, C_α, C_e = params.r_g, params.C_α, params.C_e

        # forcing fields, need to be global scope? 
        u = u_wind(x, t)
        #u = (u=u2(x, y, t), v=v2(x, y, t))

        # trig-values # we only use scalers, not vectors
        c̄ = c̄_x
        u_speed = abs(u)

        # peak parameters
        c_gp_speed, kₚ, ωₚ = c_g_conversions_vector(abs(c̄), r_g=r_g)

        # direction equations
        α = α_func(u_speed, c_gp_speed)
        Hₚ = H_β(α, p)
        Δₚ = Δ_β(α)

        # Source terms
        Ĩ = input ? Ĩ_func(α, Hₚ, C_e) : 0.0
        D̃ = dissipation ? D̃_func_lne(lne, kₚ, e_T, n) : 0.0
        S_cg_tilde = peak_shift ? S_cg(lne, Δₚ, kₚ, C_α) : 0.0
        c_group_x = propagation ? c̄_x : 0.0
        # no directional changes  in 1D!
        if info
            println("alpha = ", α)
            println("Hₚ = ", Hₚ)
            println("Δₚ = ", Δₚ)
            println("I_tilde = ", Ĩ)
            println("D_tilde = ", D̃)
            println("S_cg_tilde = ", S_cg_tilde)
            println("c_tilde_x = ", c_group_x)
        end

        # energy
        dz[1] = +ωₚ .* r_g .* S_cg_tilde + ωₚ .* (Ĩ - D̃)  #- c̄ .* G_n,
        # dz[1] = +0.5 * 9.81 * c̄_x^(-1) .* r_g .* S_cg_tilde + ωₚ .* (Ĩ - D̃) #- e^3 *ξ / c̄ ,
        #e * ωₚ .* (Ĩ -  D̃)- e^3 *ξ / c̄ ,

        # peak group velocity vector
        dz[2] = - c̄_x .* ωₚ .* r_g .* S_cg_tilde
        # dz[2] = - 0.5 * 9.81 .* r_g .* S_cg_tilde
            
        # propagation
        dz[3] = c_group_x

        return dz

    end

    return partice_system
end


"""
particle_rays(u ; γ::Number=0.88, q::Number=-1/4.0 )
Particle wave equations in 2D

returns 3 element state vector of the particle system as vector:
        [lne, c̄_x, x], lne, c̄_x, are constants
"""
function particle_rays(info=false)

    # no directional changes  in 1D!
    if info
        println("c_x = ", c̄_x)
    end
    #D = Differential(t)
    function partice_system(dz, z, params, t) #<: Vector{Number}
        lne, c̄_x, x = z
        # energy
        dz[1] = 0
        # peak group velocity vector
        dz[2] = 0
        # propagation
        dz[3] = c̄_x
    end

    return partice_system
end


# end of module
end
