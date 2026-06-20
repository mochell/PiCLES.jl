module FetchRelations

using StaticArrays
using Parameters

# non-dimensionalizations
"""
    X_tilde(X, U10)

Calculate the dimensionless fetch distance.

# Arguments
- `X::Number`: Fetch distance in meters.
- `U10::Number`: U10 wind component at 10 meters above the sea surface in m/s.

# Returns
- `X_tilde::Number`: Dimensionless fetch distance.
"""
function X_tilde(X, U10)
    return 9.81 * X / U10^2
end


"""
    t_tilde(t, U10)

Calculate the dimensionless time.

# Arguments
- `t::Number`: Time in seconds.
- `U10::Number`: U10 wind component at 10 meters above the sea surface in m/s.

# Returns
- `t_tilde::Number`: Dimensionless time.
"""
function t_tilde(t, U10)
    return t * 9.81 / U10
end

"""
    E_tilde(E, U10)

Calculate the dimensionless energy.

# Arguments
- `E::Number`: Energy.
- `U10::Number`: U10 wind component at 10 meters above the sea surface in m/s.

# Returns
- `E_tilde::Number`: Dimensionless energy.
"""
function E_tilde(E, U10)
    return E * 9.81^2 / U10^4
end

"""
    f_p_tilde(f_p, U10)

Calculate the dimensionless peak frequency.

# Arguments
- `f_p::Number`: Peak frequency in Hz.
- `U10::Number`: U10 wind component at 10 meters above the sea surface in m/s.

# Returns
- `f_p_tilde::Number`: Dimensionless peak frequency.
"""
function f_p_tilde(f_p, U10)
    return f_p * U10 / 9.81
end



#### functions to transform non-dim time to non-dim fetch ####
# defining paramters from Dulov et al. See fetch_relations.ipynb for details
# xi_0 = 11.0
# R = 0.76
# q = 0.379
@with_kw struct Dulov_constants{T<:AbstractFloat} <: FieldVector{3,AbstractFloat}
    xi_0::T = 11.0
    R::T = 0.76
    q::T = 0.379
end


"""
function get_A(qx::Float64, R::Float64)
Helping function to get coefficients for the translation between fetch and tau
"""
function get_A(qx::Float64, R::Float64)
    return (4 * pi) / (R * (1 - qx))
end

function get_A(qx::Float64; R::Float64=Dulov_constants.R)
    return (4 * pi) / (R * (1 - qx))
end

"""
function get_q_x(q::Float64)
Helping function to get q for the translation between fetch and tau
"""
function get_q_x(; q::Float64 = Dulov_constants.q)
    return q / (q + 1)
end

# predefined because they don't really change
@with_kw struct Dulov_fetch_constants{T<:AbstractFloat} <: FieldVector{3,AbstractFloat}
    q_x::T = 0.2748
    A::T = 22.8013
    xi_0x::T = 2.4097
end

q_x = 0.2748
A = 22.8013
xi_0x = 2.4097
"""
# exact definitions
q_x = get_q_x(q)
A = get_A(q_x, R)
xi_0x = (xi_0 / A^q)^(1 / (1 + q))
"""

"""
function X_tilde_from_tau(tau::Float64)
    returns non-dimensional fetch given non-dim tau (m^2/s^2)
    The parameters q_x, A, xi_0X are are from Dulov et al. 2020
"""
function X_tilde_from_tau(tau::Float64; c=Dulov_fetch_constants() )
    return (tau / (c.A * c.xi_0x))^(1 / (1 - c.q_x)) 
end

"""
function tau_from_tilde_X(tau::Float64)
    returns non-dimensional tau given non-dimensional fetch (m^2/s^2)
    The parameters q_x, A, xi_0X are are from Dulov et al. 2020
"""
function tau_from_tilde_X(X::Float64; c=Dulov_fetch_constants() )#, R::Float64, q::Float64, xi_0::Float64)
    return  c.A * c.xi_0x * X^(1 - c.q_x)
end

# tau1 = tau_from_tilde_X(1e3)
# println(tau1)
# X1 = X_tilde_from_tau(tau1)
# println(X1)




# JONSWAP fetch relations
# fetch relations following JONSWAP
#fetch_grouth_parameter= 3.5#3.622#3.5

"""
function fₘ_from_X(U10::Float64, X::Float64)
    returns peak frequency using JONSWAP given U10 (m/s) and fetch (m)
"""
function fₘ_from_X(U10::Number, X::Number; g::Number=9.81, fgp::Number=3.5)
    fgp * (g / U10) * X_tilde(U10, X)^(-0.33)
end

"""
function fₘ_from_X_tilde(U10::Float64, ::Float64)
    returns peak frequency using JONSWAP given U10 (m/s) and non-dim fetch (m)
"""
function fₘ_from_X_tilde(U10::Number, X_tilde::Number; g::Number=9.81, fgp::Number=3.5)
    return fgp * (g / U10) * X_tilde^(-0.33)
end


"""
    alpha_j(U10, f_m, g = 9.81)

Calculate the spectral peak enhancement factor.

# Arguments
- `U10::Number`: Wind speed at 10 meters above the sea surface in m/s.
- `f_m::Number`: Spectral peak frequency in Hz.
- `g::Number`: Acceleration due to gravity in m/s². Default is 9.81 m/s².

# Returns
- `alpha_j::Number`: Spectral peak enhancement factor.

"""
function alpha_j(U10::Number, f_m::Number; g::Number = 9.81)
    return 0.033 * (f_m * U10 / g)^0.67
end

"""
    E_JONSWAP(f_m, alpha_j)

Calculate the JONSWAP wave energy spectrum.

# Arguments
- `f_m::Number`: Spectral peak frequency in Hz.
- `alpha_j::Number`: Spectral peak enhancement factor.

# Returns
- `E_JONSWAP::Number`: Wave energy spectrum.

"""
function E_JONSWAP(f_m::Number, alpha_j::Number)
    return 0.31 * 9.81^2 * alpha_j * (f_m * 2 * pi)^(-4)
end



## non-dimensional fetch relations

function c_p_fetch(X_tilde, U10, X_t_0=2.2 * 10^4)
    return U10 .* 1.2 .* min_fetch(X_tilde, X_t_0).^0.33
end

function H_s_fetch(X_tilde, U10, X_t_0=2.2 * 10^4)
    return 0.26 * U10^2 * min_fetch(X_tilde, X_t_0)^(0.5) / 9.81
end

function E_fetch(X_tilde, U10, X_t_0=2.2 * 10^4)
    return 4.23 * 10^(-3) * U10^4 * min_fetch(X_tilde, X_t_0) / 9.81^2
end

function E_fetch_tilde(X_tilde, X_t_0=2.2 * 10^4)
    return 4.23 * 10^(-3) * min_fetch(X_tilde, X_t_0)
end

function min_fetch(X_tilde, X_t_0=2.2 * 10^4)
    return [min(fi / X_t_0, 1) for fi in X_tilde]
end



## function that define default wind_sea

"""
    get_initial_windsea(U10, time_scale, type="JONSWAP")

Calculate the initial windsea parameters based on the given wind speed and time scale.

# Arguments
- `U10::Number`: Wind speed at 10 meters above the sea surface in m/s.
- `time_scale::Number`: Time scale in seconds.
- `type::String`: Type of wave spectrum. Default is "JONSWAP".

# Returns
- `initial_params::Dict`: Dictionary containing the initial windsea parameters:
    - `E::Number`: Windsea energy spectrum.
    - `cg_bar::Number`: Group velocity.
    - `Hs::Number`: Significant wave height.
    - `f_peak::Number`: Spectral peak frequency.
    - `T_bar::Number`: Spectral peak period.
    - `X_tilde::Number`: Dimensionless fetch distance.
    - `mom::Number`: Momentum.

"""
function get_initial_windsea(U10::Number, time_scale::Number, type::String="JONSWAP")
    time_scale      = abs(time_scale)
    tau             = 9.81 * time_scale / abs(U10)
    U10_sign        = sign(U10)

    # get non-linear growth parameters
    X_tilde_        = X_tilde_from_tau(tau)
    f_m_            = fₘ_from_X_tilde(abs(U10), X_tilde_)
    alpha_j_        = alpha_j(abs(U10), f_m_)

    # get initial windsea
    if type == "JONSWAP"
        E_          = E_JONSWAP(f_m_, alpha_j_)
        Hs_         = 4 * sqrt(E_)

        # from Boews 1998, eq. 4.2
        f_peak      = f_m_ * 9.81 / abs(U10)

    elseif type == "PM"
        f_peak      = 0.816 * 9.81 / (2 * pi * abs(U10))
        Hs_         = 0.0246 * abs(U10)^2
        E_          = (Hs_/4)^2
    end

    T_bar           = 0.9 * (1/f_peak)
    cg_bar_amp      = 9.81 * T_bar / (4 * pi)
    cg_bar          = U10_sign * cg_bar_amp

    # momentum
    mom             = U10_sign * E_ / (2 * cg_bar_amp)

    # return dictionary with E, cg_bar, Hs, f_peak, T_bar, X_tilde
    return Dict("E" => E_, "lne" => log(E_) , "cg_bar" => cg_bar, "Hs" => Hs_, "f_peak" => f_peak, "T_bar" => T_bar, "X_tilde" => X_tilde_, "m" => mom)
end

"""
    get_initial_windsea(U10, V10, time_scale, type="JONSWAP")

Calculate the initial windsea parameters based on the given U10 and V10 wind components, time scale, and type of wave spectrum.

# Arguments
- `U10::Number`: U10 wind component at 10 meters above the sea surface in m/s.
- `V10::Number`: V10 wind component at 10 meters above the sea surface in m/s.
- `time_scale::Number`: Time scale in seconds.
- `type::String`: Type of wave spectrum. Default is "JONSWAP".

# Returns
- `initial_params::Dict`: Dictionary containing the initial windsea parameters:
    - `E::Number`: Windsea energy spectrum.
    - `Hs::Number`: Significant wave height.
    - `cg_bar_x::Number`: X-component of group velocity.
    - `cg_bar_y::Number`: Y-component of group velocity.
    - `cg_bar::Number`: Magnitude of group velocity.
    - `f_peak::Number`: Spectral peak frequency.
    - `T_bar::Number`: Spectral peak period.
    - `X_tilde::Number`: Dimensionless fetch distance.
    - `m_x::Number`: X-component of momentum.
    - `m_y::Number`: Y-component of momentum.

"""
function get_initial_windsea(U10::Number, V10::Number, time_scale::Number; type::String="JONSWAP", particle_state::Bool=false)
    U_amp       = sqrt(U10^2 + V10^2)
    U_amp       = ifelse(U_amp < 0.1, 0.1, U_amp)

    time_scale  = abs(time_scale)
    tau         = 9.81 * time_scale / abs(U_amp)
    
    # get non-linear growth parameters
    X_tilde_    = X_tilde_from_tau(tau)
    f_m_        = fₘ_from_X_tilde(U_amp, X_tilde_)
    alpha_j_    = alpha_j(U_amp, f_m_)

    # get initial windsea
    if type == "JONSWAP"
        E_      = E_JONSWAP(f_m_, alpha_j_)
        Hs_     = 4 * sqrt(E_)

        # from Boews 1998, eq. 4.2
        f_peak  = f_m_ * 9.81 / U_amp

    elseif type == "PM"
        f_peak  = 0.816 * 9.81 / (2 * pi * U_amp)
        Hs_     = 0.0246 * U_amp^2
        E_      = (Hs_/4)^2
    end

    # mean period
    T_bar = 0.9 * (1 / f_peak)
    cg_bar_amp  = 9.81 * T_bar / (4 * pi)
    cg_bar_x    = cg_bar_amp * U10 / U_amp
    cg_bar_y    = cg_bar_amp * V10 / U_amp


    if particle_state
        return [log(E_), cg_bar_x, cg_bar_y, 0.0, 0.0]
    else
        # momentum
        mom_x = (U10 / U_amp) * E_ / (2 * cg_bar_amp)
        mom_y = (V10 / U_amp) * E_ / (2 * cg_bar_amp)

        # return dictionary with E, Hs, cg_bar_x, cg_bar_y, cg_bar, f_peak, T_bar, X_tilde, m_x, m_y
        return Dict("E" => E_, "lne" => log(E_), "Hs" => Hs_, "cg_bar_x" => cg_bar_x, "cg_bar_y" => cg_bar_y,
            "cg_bar" => cg_bar_amp, "f_peak" => f_peak, "T_bar" => T_bar, "X_tilde" => X_tilde_, "m_x" => mom_x, "m_y" => mom_y)
    end

end




u_min = 1.0
rand_sign() = sign.(rand(0:1) - 0.5)

"""
    MinimalWindsea(U10, timescale, type="JONSWAP")
Alias for default minimum Parameters takes wind just for the correct sign convention, then replaces with u_min values.
"""
function MinimalWindsea(U10::Number, time_scale::Number, type::String="JONSWAP")
    U10 = U10 == 0 ? rand_sign() : U10
    return get_initial_windsea(sign(U10) * u_min, time_scale, type)
end


"""
    MinimalWindsea(U10, V10, timescale, type="JONSWAP")
Alias for default minimum Parameters takes wind just for the correct sign convention, then replaces with u_min values.
"""
function MinimalWindsea(U10::Number, V10::Number, time_scale::Number, type::String="JONSWAP")
    U10 = U10 == 0 ? rand_sign() : U10 
    V10 = V10 == 0 ? rand_sign() : V10
    Uamp =  sqrt(U10^2 + V10^2)
    return get_initial_windsea(u_min * U10 / Uamp, u_min * V10 / Uamp, time_scale; type=type)
end

"""
    MinimalParticle(U10, timescale, type="JONSWAP")
Alias for default minimum Parameters. Takes wind just for the correct sign convention, then replaces with u_min values.
"""
function MinimalParticle(U10::Number, time_scale::Number, type::String="JONSWAP")
    WindSeamin=  MinimalWindsea(U10, time_scale, type)
    return [log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0, 0]
end

"""
    MinimalParticle(U10, V10, timescale, type="JONSWAP")
Alias for default minimum Parameters. Takes wind just for the correct sign convention, then replaces with u_min values.
"""
function MinimalParticle(U10::Number, V10::Number, time_scale::Number, type::String="JONSWAP")
    WindSeamin=  MinimalWindsea(U10, V10, time_scale, type)
    return [log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0, 0]
end

"""
    MinimalState(U10, V10, timescale, type="JONSWAP")
Alias for needed minimum State. Takes wind just for the correct sign convention, then replaces with u_min values.
    returns:
    [ minimal energy, momentum^2]
"""
function MinimalState(U10::Number, V10::Number, time_scale::Number, type::String="JONSWAP")
    WindSeamin=  MinimalWindsea(U10, V10, time_scale, type)
    return [WindSeamin["E"], WindSeamin["m_x"]^2 + WindSeamin["m_y"]^2 ]
end


## testing
# u10, v10 = -2, 0
# WS1d = get_initial_windsea(sqrt(u10^2 + v10^2), 60 * 60 * 12)
# WS2d = get_initial_windsea(u10      , v10     , 60 * 60 * 12)

# WS1d["E"], WS2d["E"]
# WS1d["cg_bar"], WS2d["cg_bar"], WS2d["cg_bar_x"], WS2d["cg_bar_y"]
# WS1d["m"],sqrt( WS2d["m_x"]^2 + WS2d["m_y"]^2 ), WS2d["m_x"], WS2d["m_y"]
# WS1d["T_bar"], WS2d["T_bar"]

#### combined static fetch relations ####

"""
function X_tilde_douple_limited(t::Number, U10::Number, X::Number)
    returns non-dimensional effective fetch of either a duration or fetch limited condition

# Arguments
- `t::Number`: Time in seconds.
- `U10::Number`: U10 wind component at 10 meters above the sea surface in m/s.
- `X::Number`: Fetch distance in meters.

# Returns
- `Xt::Number`: Double-limited dimensionless fetch distance.
"""
function X_tilde_time_and_fetch(t::Number, U10::Number, X::Number)
    Tt = t_tilde(t, U10)
    Xt = X_tilde(X, U10)
    
    if Tt < 10^5
        Xt = min( Xt  , X_tilde_from_tau(Tt)   )  # use Dulov et al 2020
    end
    return Xt
end

X_tilde_time_and_fetch(3 * 3600, 10, 5e3)

# %%
###### old functions ######

"""
function X̃_j_U_freq(X::Float64, U::Float64, g::Float=9.81)
    returns non-dimensional fetch using JONSWAP given peak frequency f_max (Hz), U (m/s) and gravity (m)
"""
function X̃_j_U_freq(U10::Number, f_max::Number, g::Number=9.81)
    fetch_grouth_parameter^3.0815 * g^3 /(U10^3 *f_max^3)
end

"""
function X_j_U_freq(X::Float64, U::Float64, g::Float=9.81)
    returns fetch (meters) using JONSWAP given peak frequency f_max (Hz), U (m/s) and gravity (m)
"""
function X_j_U_freq(U10::Number, f_max::Number,  g::Number=9.81)
    fetch_grouth_parameter^3.0815 * g^2 /(U10 *f_max^3)
end

"""
function X̃_j_U_freq(X::Float64, U::Float64, g::Float=9.81)
    returns fetch (meters) using JONSWAP given peak frequency f_max (Hz), U (m/s) and gravity (m)
"""
function X̃_j_U_tau(U10::Number, tau::Number,  g::Number=9.81)
    (tau * g / (14 * pi * U10) )^(3/2)
end

"""
function α_j(f_max::Float64, U::Float64,, g::Float=9.81 )
    returns fetch amplitude using JONSWAP given peak frequency f_max (Hz), U (m/s) and gravity (m)
"""
function α_j(U::Number,  f_max::Number, g::Number=9.81 )
    0.033*(f_max*U/g)^0.67
end

"""
function τ_j(U::Number,  X::Number, g::Number=9.81 )
    returns equivalent fetch time using JONSWAP given U (m/s) and fetch (meters)and gravity (m)
"""
function τ_j(U::Number,  X::Number, g::Number=9.81 )
    14 * pi * (U/g) * X_tilde(U, X)^(2/3.0)
end

# tau1 = τ_j(U, X)
#
# U = 15.0
# X = 500e3
# X_tilde =X_tilde(U, X)
#
# X_tilde
#
# f_max = fₘ(U, X)
# f_max
#
# X/X_j_U_freq(U, f_max)
#
# X_j_U_freq(U, f_max)/X
#
# X_tilde/X̃_j_U_freq( U, f_max)
#
# fₘ(U, X )/ fₘ_given_U_tau(U, tau1)
#
# X̃_j_U_tau(U, tau1)
#
#




"""
function fₘ_given_U_tau(U10::Float64, tau::Float64)
    returns peak frequency using JONSWAP given U10 (m/s) and time (sec)
"""
function fₘ_given_U_tau(U10::Float64, tau::Float64)
    X_tilde = (9.81 * tau/ ( 14 * pi * U10) )^(3.0/2.0)
    f_max   =  fetch_grouth_parameter * (9.81/ U10) * X_tilde^(-1/3.0)
    f_max * 1.035 # empirical adjustment factor
end

"""
function c_g_U_tau(U10::Float64, tau::Float64)
    returns peak frequency using JONSWAP given U10 (m/s) and time (sec)
"""
function c_g_U_tau(U10::Float64, tau::Float64)
    g= 9.81
    g / (4 * pi * fₘ_given_U_tau(U10, tau) )
end


"""
function Eⱼ(U10::Float64, tau::Float64)
    returns JONSWAP wave energy given U10 (m/s) and time (sec)
"""
function Eⱼ(U10::Float64, tau::Float64)
    f_max = fₘ_given_U_tau(U10::Float64, tau::Float64)
    alpha_j = α_j(U10,  f_max )
    0.31 * 9.81^2 * alpha_j * (f_max * 2 *pi)^(-4)
end



"""
JONSWAP_omega(U10::Float64, ωₚ , ω::Vector, g::Number=9.81)
    return the Pierson–Moskowitz spectrum parameters f_peak, Hs, Energy
    following (Bouws, 1998) given U10 (m/s) as the 10 meters wind speed.
"""
function JONSWAP_omega(U10::Float64, ωₚ , ω::Vector, g::Number=9.81)
    S = (2 * pi * α_j(U, ωₚ) * g^2 ) ./ ω.^5  .* exp.(-(5/4.0) .* (ωₚ./ω).^4 )
    sigma= (ω .> ωₚ ) .* 0.09 + (ω .<= ωₚ ) .* 0.07
    Gamma_j = exp.( - ( ω .- ωₚ).^2 ./ (2 .* sigma.^2 .* ωₚ^2) )
    S .*= 3.3.^Gamma_j
    S
end

# ω = collect(range(1/100, 1, step=1/100))
# ω.^3
# ωₚ = 2 * pi *f_max
#
# using Plots
# JONSWAP_omega(U, ωₚ , ω)
# plot( ω, JONSWAP_omega(U, ωₚ/2 , ω) )

"""
JONSWAP_frequency(U10::Float64, fₚ , freq::Vector, g::Number=9.81)
    return the Pierson–Moskowitz spectrum parameters f_peak, Hs, Energy
    following (Bouws, 1998) given U10 (m/s) as the 10 meters wind speed.
"""
function JONSWAP_frequency(U10::Float64, fₚ , freq::Vector, g::Number=9.81)
    JONSWAP_omega(U10, fₚ * 2 * pi , ω * 2 * pi)
end


#plot( ω/2 /pi, JONSWAP_frequency(U, ωₚ/2/pi/2 , ω/2/pi) )


"""
PMSpectrum(f::Vector, U10::Float64)
    returns Pierson Moskoviz spectum given f as the frequency in (Hz) and U10 (m/s) as the 10 meters wind speed.
    --- never tested!!
"""
function PMSpectrum(U10::Float64, f::Vector)
    """
    see Ocean Surface waves - S. R. Massel eq.3.79 and eq.3.80
    """
    g   = 9.81
    wp  = 0.879*g / U10
    w   = 2*np.pi*f
    sigma=0.04 *g / wp^2.0
    alpha=5.0* (wp^2.0 *sigma / g)^2.0
    alpha * w^(-5.0) * g^2.0 *np.exp(-5/4.0 * (w/wp)^-4)#
end

"""
PMParameters(f::Vector, U10::Float64)
    return the Pierson–Moskowitz spectrum parameters f_peak, Hs, Energy
    following (Bouws, 1998) given U10 (m/s) as the 10 meters wind speed.
    ---- never tested!!
"""
function PMParameters(U10::Float64)
    f_peak = 0.816  * 9.81 / (2 * pi * U10)
    Hs     = 0.0246 * U10^2
    E      = (Hs/4) ^4
    return (f_peak = f_peak , Hs = Hs, E=E)
end


function PMlimits()
    return (E_tilde= 0.00402, f_p_tilde=0.123)
end


"""
    windsea_momentum_PM(U; g=9.81)

Momentum amplitude |m| of a *fully developed* Pierson–Moskowitz wind sea for a 10 m wind
speed `U` (m/s). This is the closed form of the `"PM"` branch of `get_initial_windsea`:
with Hs = 0.0246·U², E = (Hs/4)², f_peak = 0.816·g/(2π·U), T̄ = 0.9/f_peak,
c̄g = g·T̄/(4π) and |m| = E/(2·c̄g), the U-dependence collapses to |m|_ws,PM(U) ∝ U³
(≈ 3.43e-5·U³). Allocation-free (no dict), so it can be evaluated per-node in `remesh!`.

Used to scale a local, wind-sea-aware `m_amp_minimum` reconstruction floor:
`threshold = max(m_amp_minimum, α · windsea_momentum_PM(U_local))`. Because PM is fully
developed, this is an *upper bound* on the local wind-sea momentum (a young/fetch-limited
sea carries less), so the tunable coefficient α is correspondingly small.
"""
@inline function windsea_momentum_PM(U::Number; g::Float64=9.81)
    Ua = abs(U)
    Ua == 0 && return 0.0
    Hs = 0.0246 * Ua^2
    E  = (Hs / 4)^2
    f_peak = 0.816 * g / (2 * pi * Ua)
    cg = g * (0.9 / f_peak) / (4 * pi)         # c̄g = g·T̄/(4π), T̄ = 0.9/f_peak
    return E / (2 * cg)
end


end
