module RK35Integrator

export ODEIntegrator, step!, solve!

using ...Architectures: AbstractODEIntegrator

# ------------------------------------------------------------
# Bogacki–Shampine RK3(2) ("RK3.5") adaptive integrator (in-place)
# RHS signature: model!(du, u, params, t)
# ------------------------------------------------------------

mutable struct ODEIntegrator{F,T,Z<:AbstractVector,P} <: AbstractODEIntegrator
    model!::F
    u::Z
    t::T
    params::P

    # adaptive control
    dt::T
    reltol::T
    abstol::T
    dtmin::T
    dtmax::T
    safety::T

    # work buffers
    k1::Z
    k2::Z
    k3::Z
    k4::Z
    tmp::Z
    unew::Z
    sc::Z
    err::Z

    # FSAL cache flag: if true, k1 already equals model!(u,t)
    has_fsal::Bool
end

"""
    ODEIntegrator(model!, u0, t0, params=nothing; kwargs...)

Create an RK3(2) Bogacki–Shampine integrator (adaptive internal dt), in-place RHS:

    model!(du, u, params, t)

`u0` must be an AbstractVector.
"""
function ODEIntegrator(model!::F, u0::AbstractVector, t0::Real, params=nothing;
    dt=1e-2, reltol=1e-6, abstol=1e-9,
    dtmin=1e-12, dtmax=1.0, safety=0.9) where {F}
    u = copy(u0)
    Tt = float(t0)

    k1 = similar(u)
    k2 = similar(u)
    k3 = similar(u)
    k4 = similar(u)
    tmp = similar(u)
    unew = similar(u)
    sc = similar(u)
    err = similar(u)

    return ODEIntegrator{F,typeof(Tt),typeof(u),typeof(params)}(
        model!, u, Tt, params,
        float(dt), float(reltol), float(abstol), float(dtmin), float(dtmax), float(safety),
        k1, k2, k3, k4, tmp, unew, sc, err,
        false
    )
end

@inline function _scaled_errnorm!(integ::ODEIntegrator)
    u = integ.u
    un = integ.unew
    err = integ.err
    sc = integ.sc
    atol = integ.abstol
    rtol = integ.reltol

    @inbounds @. sc = atol + rtol * max(abs(u), abs(un))

    s = zero(eltype(sc))
    @inbounds for i in eachindex(err)
        r = err[i] / sc[i]
        s += r * r
    end
    return sqrt(s / length(err))
end

# One RK3(2) attempt of size dt (does not commit state/time).
# Fills integ.unew and integ.err.
function _rk35_attempt!(integ::ODEIntegrator, dt::Real)
    model! = integ.model!
    u = integ.u
    t = integ.t
    p = integ.params

    k1 = integ.k1
    k2 = integ.k2
    k3 = integ.k3
    k4 = integ.k4
    tmp = integ.tmp
    un = integ.unew
    err = integ.err

    dtT = float(dt)

    # k1 = f(u,t) (FSAL reuse if available)
    if !integ.has_fsal
        model!(k1, u, p, t)
    end

    # tmp = u + dt/2 * k1
    @. tmp = u + (dtT / 2) * k1
    model!(k2, tmp, p, t + dtT / 2)

    # tmp = u + 3dt/4 * k2
    @. tmp = u + (3dtT / 4) * k2
    model!(k3, tmp, p, t + 3dtT / 4)

    # 3rd-order solution
    @. un = u + dtT * ((2 / 9) * k1 + (1 / 3) * k2 + (4 / 9) * k3)

    # FSAL evaluation at end of step
    model!(k4, un, p, t + dtT)

    # embedded 2nd-order solution error (u3 - u2)
    @. err = un - (u + dtT * ((7 / 24) * k1 + (1 / 4) * k2 + (1 / 3) * k3 + (1 / 8) * k4))

    return nothing
end

# Internal accept/reject loop used by solve! when we don't want to save
function _advance_to!(integ::ODEIntegrator, t_end::Real; maxiters::Int=10^7)
    t_end = float(t_end)
    t_end < integ.t && error("t_end=$(t_end) is behind current time t=$(integ.t)")

    expo = 1 / 3
    iters = 0

    while integ.t < t_end
        iters += 1
        iters > maxiters && error("_advance_to! exceeded maxiters=$maxiters at t=$(integ.t)")

        dt_try = min(integ.dt, t_end - integ.t)

        _rk35_attempt!(integ, dt_try)
        en = _scaled_errnorm!(integ)

        if en <= 1.0
            copyto!(integ.u, integ.unew)
            integ.t += dt_try

            copyto!(integ.k1, integ.k4)
            integ.has_fsal = true

            fac = integ.safety * en^(-expo)
            integ.dt = clamp(integ.dt * fac, integ.dtmin, integ.dtmax)
        else
            integ.has_fsal = false

            fac = integ.safety * en^(-expo)
            integ.dt = clamp(integ.dt * max(0.1, fac), integ.dtmin, integ.dtmax)

            if integ.dt == integ.dtmin
                error("Step size underflow (dtmin reached) at t=$(integ.t), errnorm=$en")
            end
        end
    end

    return integ
end

"""
    step!(integ, DT)

Advance `integ` from `t` to `t + DT` using adaptive internal steps.

Updates `integ.u`, `integ.t`, and keeps `integ.dt` as the suggested next internal step.
"""
function step!(integ::ODEIntegrator, DT::Real; maxiters::Int=10^4)
    return _advance_to!(integ, integ.t + float(DT); maxiters=maxiters)
end

# alias for old version

function step!(integ::ODEIntegrator, DT::Real, a::Bool; maxiters::Int=10^4)
    return _advance_to!(integ, integ.t + float(DT); maxiters=maxiters)
end


"""
    solve!(integ, t_end; saveat=nothing, save=true, maxiters=10^7)

Integrate forward until `t_end`.

Saving behavior (when `save=true`):
- `saveat === nothing`: saves every accepted internal step (variable spacing).
- `saveat::Real`: saves at uniform output interval `saveat` (outer stepping).
- `saveat::AbstractVector`: saves exactly at those times (in ascending order).

When `save=false`:
- No history is stored; only advances to `t_end` and returns `(t_final, u_final_copy)`.

Returns:
- if `save=true`: `(ts::Vector{Float64}, us::Vector{Vector})`
- if `save=false`: `(t_final::Float64, u_final::Vector)`
"""
function solve!(integ::ODEIntegrator, t_end::Real; saveat=nothing, save::Bool=true, maxiters::Int=10^7)
    t_end = float(t_end)
    t_end < integ.t && error("t_end=$(t_end) is behind current time t=$(integ.t)")

    if !save
        _advance_to!(integ, t_end; maxiters=maxiters)
        return float(integ.t), copy(integ.u)
    end

    ts = Float64[float(integ.t)]
    us = [copy(integ.u)]

    if saveat === nothing
        # Save every accepted internal step (variable spacing)
        expo = 1 / 3
        iters = 0
        while integ.t < t_end
            iters += 1
            iters > maxiters && error("solve! exceeded maxiters=$maxiters at t=$(integ.t)")

            dt_try = min(integ.dt, t_end - integ.t)
            _rk35_attempt!(integ, dt_try)
            en = _scaled_errnorm!(integ)

            if en <= 1.0
                copyto!(integ.u, integ.unew)
                integ.t += dt_try

                copyto!(integ.k1, integ.k4)
                integ.has_fsal = true

                push!(ts, float(integ.t))
                push!(us, copy(integ.u))

                fac = integ.safety * en^(-expo)
                integ.dt = clamp(integ.dt * fac, integ.dtmin, integ.dtmax)
            else
                integ.has_fsal = false

                fac = integ.safety * en^(-expo)
                integ.dt = clamp(integ.dt * max(0.1, fac), integ.dtmin, integ.dtmax)

                if integ.dt == integ.dtmin
                    error("Step size underflow (dtmin reached) at t=$(integ.t), errnorm=$en")
                end
            end
        end
        return ts, us
    end

    if saveat isa Real
        dt_out = float(saveat)
        dt_out <= 0 && error("saveat interval must be > 0")

        while integ.t < t_end
            dt = min(dt_out, t_end - integ.t)
            step!(integ, dt; maxiters=maxiters)
            push!(ts, float(integ.t))
            push!(us, copy(integ.u))
        end
        return ts, us
    end

    if saveat isa AbstractVector
        for tout in saveat
            toutf = float(tout)
            if toutf < integ.t
                continue
            end
            if toutf > t_end
                break
            end
            step!(integ, toutf - integ.t; maxiters=maxiters)
            push!(ts, float(integ.t))
            push!(us, copy(integ.u))
        end
        if integ.t < t_end
            step!(integ, t_end - integ.t; maxiters=maxiters)
            push!(ts, float(integ.t))
            push!(us, copy(integ.u))
        end
        return ts, us
    end

    error("Unsupported saveat type: $(typeof(saveat))")
end

end # module