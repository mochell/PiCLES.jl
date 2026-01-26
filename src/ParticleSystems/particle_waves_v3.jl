module particle_waves_v3

# using ModelingToolkit, DifferentialEquations
using DifferentialEquations

using ...Architectures: AbstractODESettings

export D̃_eval, Ĩ_eval, e_T_eval, particle_equations, ODESettings
using LinearAlgebra

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


# register external forcing functions

# @register_symbolic u_10(x,y,t)
# @register_symbolic φ_w(x,y,t)
######




"""
ODESettings
Structure to hold all information about the ODE system
# Fields  
$(DocStringExtensions.FIELDS)
"""
@with_kw struct ODESettings <: AbstractODESettings
        "ODE parameters (Dict)"
        Parameters::Dict{Num,Float64}
        "minimum allowed log energy on particle "
        log_energy_minimum::Float64
        "maximum allowed log energy on particle "
        log_energy_maximum::Float64 = log(17)

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

        "Total time of the ODE integration, should not be needed, this is problematic .. "
        total_time::Float64

        "Callback function for ODE solver"
        callbacks::Any = nothing
        "save_everystep (false)"
        save_everystep::Bool = false
end



function magic_fractions(q::Number=-1 / 4.0)
        # returns universal exponent relations
        p = (-1 - 10 * q) / 2
        n = 2 * q / (p + 4 * q)
        [p, q, n]
end

αₚ(α::Number, φ::Number, φ_w::Number) = cos.(φ .- φ_w) .* α
αₚ(α::Number, cφ_p::Number, sφ_p::Number, cφ_w::Number, sφ_w::Number) = (cφ_p .* cφ_w - sφ_p .* sφ_w) .* α

H_β(α::Number, p::Number; α_thresh=0.85) = 0.5 .* (1.0 + tanh.(p .* (α .- α_thresh)))
Δ_β(α::Number; α_thresh=0.85) = (1.0 .- 1.25 .* sech.(10.0 .* (α .- α_thresh)) .^ 2)

sin2_a_min_b(ca::Number, sa::Number, cb::Number, sb::Number) = 4 * sb * cb * (sa^2 - 0.5) - 4 * sa * ca * (sb^2 - 0.5)
cos2_a_min_b(ca::Number, sa::Number, cb::Number, sb::Number) = (1 - 2 .* (sa .* cb - ca .* cb) .^ 2)
e_T_eval(γ::Float64, p::Float64, q::Float64, n::Float64; C_e::Number=2.16e-4) = sqrt(1.3e-6 * 11.8^(-p / q) / (γ * C_e)^(1 / n))


function D̃_eval_e(e::Number, kₚ::Number, e_T::Number, n::Float64)
        # non-dimensional Wind energy input
        # eq sec. 1.3 in the manual
        e .^ n .* (kₚ ./ e_T) .^ (2 * n)
end

function D̃_eval_lne(lne::Number, kₚ::Number, e_T::Number, n::Float64)
        # non-dimensional Wind energy input
        # eq sec. 1.3 in the manual
        exp(n .* lne) .* (kₚ ./ e_T) .^ (2 * n)
end

function Ĩ_eval(alpha::Number, Hₚ::Number, C_e::Number)
        # non-dimensional Wind energy input
        # eq sec. 1.3 in the manual
        C_e * Hₚ * alpha^2
end

function init_vars()
        @variables t
        @variables x(t), y(t), c̄_x(t), c̄_y(t), lne(t), Δn(t), Δφ_p(t)
        @parameters r_g C_α C_φ g C_e
        return t, x, y, c̄_x, c̄_y, lne, Δn, Δφ_p, r_g, C_α, C_φ, g, C_e
end

function init_vars_1D()
        @variables t
        @variables x(t), c̄_x(t), lne(t)
        @parameters r_g C_α g C_e
        return t, x, c̄_x, lne, r_g, C_α, g, C_e
end

#normal_vector(uui::Vector) = [- uui[2], uui[1]] / norm(uui)

function Gₙ(dphi_p::Number, dn::Number, dn_0::Number; dg_ratio::Float64=0.21)
        return (dphi_p / dn_0) * ((dn / dn_0) / ((dn / dn_0)^2 + (dg_ratio / 2.0)^2))
end

function c_g_conversions(c̄::Number; g::Number=9.81, r_g::Number=0.9)
        c_gp = c̄ ./ r_g
        kₚ = g ./ (4.0 .* c_gp^2)
        ωₚ = g ./ (2.0 .* c_gp)
        [c_gp, kₚ, ωₚ]
end

function speed_and_angles(cx::Number, cy::Number)
        #sqrt(cx.^2 + cy.^2), cx ./ sqrt(cx.^2 + cy.^2), cy ./sqrt(cx.^2 + cy.^2)
        c = sqrt(cx .^ 2 + cy .^ 2)
        c, cx ./ c, cy ./ c
end

speed(cx::Number, cy::Number) = sqrt(cx .^ 2 + cy .^ 2)
α_func(u_speed::Number, c_gp::Number) = u_speed ./ (2.0 * c_gp)

tΔφ_w_func(c_x::Number, c_y::Number, u_x::Number, v_y::Number) = (c_x .* v_y - c_y .* u_x) / (c_x .* u_x + c_y .* v_y)
Δφ_p_RHS(tΔφ_w::Number, tΔφ_p::Number, T_inv::Number) = (tΔφ_w - tΔφ_p) * T_inv / (1 + tΔφ_w * tΔφ_p)


function ξ_cross(C_φ::Number, α::Number, ωₚ::Number, Hₚ::Number, angles::Vector{<:Number})
        cφ_p, sφ_p, cφ_w, sφ_w = angles
        C_φ * α^2 * ωₚ * Hₚ * sin2_a_min_b(cφ_p, sφ_p, cφ_w, sφ_w)
end


# peak downshift
# C_α is negative in Kudravtec definition, here its a positive value, so we introduce a minus sign
ξ_shift(lne::Number, Δₚ::Number, kₚ::Number, C_α::Number, r_g::Number, g::Number) = 0.5 * r_g^2 * C_α * Δₚ * g * kₚ .^ 4 * exp(2 * lne)

"""
particle_equations(u , v, u_x, v_y ; γ::Number=0.88, q::Number=-1/4.0, dir_enhancement::Bool=true, delta_x_0::Number=10e3 )
Particle wave equations in 2D
"""
function particle_equations(u, v, u_x, v_y; γ::Number=0.88, q::Number=-1 / 4.0, dir_enhancement::Bool=true, delta_x_0::Number=10e3)
        t, x, y, c̄_x, c̄_y, lne, Δn, Δφ_p, r_g, C_α, C_φ, g, C_e = init_vars()

        p, q, n = magic_fractions()
        e_T = e_T_eval(γ, p, q, n, C_e=C_e)
        D = Differential(t)

        # forcing fields
        u = u(x, y, t)
        v = v(x, y, t)
        u_x = u_x(x, y, t)
        v_y = v_y(x, y, t)

        # trig-values # we only use scalers, not vectors
        c̄, cφ_p, sφ_p = speed_and_angles(c̄_x, c̄_y)
        u_speed, cφ_w, sφ_w = speed_and_angles(u, v)

        tΔφ_p = tan(Δφ_p)
        #tΔφ_w = (c̄_x .* v_y(x,y,t) - c̄_y .* u_x(x,y,t) ) / ( c̄_x .* u_x(x,y,t) + c̄_y .* v_y(x,y,t) )

        tΔφ_w = tΔφ_w_func(u, v, u + u_x, v + v_y)
        #tΔφ_w =  tΔφ_w_func(c̄_x, c̄_y, c̄_x +  u_x(x,y,t), c̄_y + v_y(x,y,t) )

        #tΔφ_w =  ( c̄_x .* u_x(x,y,t) + c̄_y .* v_y(x,y,t) ) /  ( c̄ .* u_speed  )
        #tΔφ_w =  u_x(x,y,t) + v_y(x,y,t) #( c̄_x .* u_x(x,y,t) + c̄_y .* v_y(x,y,t) ) /  ( c̄ .* u_speed  ) #

        #tΔφ_w =  ( sign(sφ_p) * sφ_p * u_x(x,y,t) + sign(cφ_p) * cφ_p * v_y(x,y,t) ) #/ u_speed # sqrt( v_y(x,y,t)^2 + u_x(x,y,t)^2 )   #  dot product with the normal vector of direction, not normalized


        #φ_w  = atan( v(x,y,t), u(x,y,t)  ) # y , x
        #Δφ_w =  pi./ 4
        #Δφ_w = atan( v_y(x,y, t) ./ u_x(x, y, t))
        #Δφ_w = Δφ_w_func( cos(φ_p) , sin(φ_p) , u_x(x,y,t), v_y(x,y,t) )

        # peak parameters
        c_gp, kₚ, ωₚ = c_g_conversions(c̄, r_g=r_g)

        # direction equations
        α = α_func(u_speed, c_gp)
        Hₚ = H_β(αₚ(α, cφ_p, sφ_p, cφ_w, sφ_w), p)
        Δₚ = Δ_β(αₚ(α, cφ_p, sφ_p, cφ_w, sφ_w))
        #AB      = C_φ * α^2 * ωₚ * Hₚ *  sin2_a_min_b(cφ_p, sφ_p, cφ_w, sφ_w)  #sin( 2.0 * (φ_p - φ_w) )

        # wave growth equation
        T_inv_angle = cos2_a_min_b(cφ_p, sφ_p, cφ_w, sφ_w)
        T_inv = 2 * C_φ * Hₚ * α .^ 2 * ωₚ * sign(T_inv_angle) * T_inv_angle # eq. 1.23
        D̃ = D̃_eval_lne(lne, kₚ, e_T, n)
        Ĩ = Ĩ_eval(α, Hₚ, C_e)

        # peak downshift
        # C_α is negative in Kudravtec definition, here its a positive value, so we introduce a minus sign

        # ray convergene
        G_n = Gₙ(Δφ_p, Δn, delta_x_0)
        #G_n    = Δφ_w /  delta_x  delta_x_0

        particle_equations = [
                # propagation
                D(x) ~ c̄_x,
                D(y) ~ c̄_y,

                # peak group velocity vector
                D(c̄_x) ~ +c̄_y * ξ_cross(C_φ, α, ωₚ, Hₚ, [cφ_p, sφ_p, cφ_w, sφ_w]) - cφ_w .* ξ_shift(lne, Δₚ, kₚ, C_α, r_g, g),
                D(c̄_y) ~ -c̄_x * ξ_cross(C_φ, α, ωₚ, Hₚ, [cφ_p, sφ_p, cφ_w, sφ_w]) - sφ_w .* ξ_shift(lne, Δₚ, kₚ, C_α, r_g, g),

                # energy
                D(lne) ~ -c̄ .* G_n + ξ_shift(lne, Δₚ, kₚ, C_α, r_g, g) / c̄ + ωₚ .* (Ĩ - D̃),
                #D(e)~  e * ωₚ .* (Ĩ -  D̃)- e^3 *ξ / c̄ ,

                # ray convergence
                D(Δn) ~ Δφ_p * c̄,
                D(Δφ_p) ~ 0#Δφ_p_RHS(tΔφ_w, tΔφ_p , T_inv)
        ]

        return particle_equations
end

"""
particle_equations(u , u_x, ; γ::Number=0.88, q::Number=-1/4.0)
Particle wave equations in 1D
"""
function particle_equations(u, u_x; γ::Number=0.88, q::Number=-1 / 4.0)
        t, x, c̄_x, lne, r_g, C_α, g, C_e = init_vars_1D()

        p, q, n = magic_fractions()
        e_T = e_T_eval(γ, p, q, n, C_e=C_e)
        D = Differential(t)

        # forcing fields
        u = u(x, t)
        u_x = u_x(x, t)

        # trig-values # we only use scalers, not vectors

        c̄ = c̄_x
        u_speed = u

        # peak parameters
        c_gp, kₚ, ωₚ = c_g_conversions(c̄, r_g=r_g)

        # direction equations
        α = α_func(u_speed, c_gp)
        Hₚ = H_β(α, p)
        Δₚ = Δ_β(α)

        # wave growth equation
        D̃ = D̃_eval_lne(lne, kₚ, e_T, n)
        Ĩ = Ĩ_eval(α, Hₚ, C_e)

        particle_equations = [
                # propagation
                D(x) ~ c̄_x,

                # peak group velocity vector
                D(c̄_x) ~ - ξ_shift(lne, Δₚ, kₚ, C_α, r_g, g),

                # energy
                D(lne) ~ ξ_shift(lne, Δₚ, kₚ, C_α, r_g, g) / c̄ + ωₚ .* (Ĩ - D̃),
                #D(e)~  e * ωₚ .* (Ĩ -  D̃)- e^3 *ξ / c̄ ,
        ]

        return particle_equations
end



end
