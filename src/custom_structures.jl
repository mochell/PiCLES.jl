module custom_structures

export ParticleInstance1D, ParticleInstance2D, MarkedParticleInstance, AbstractParticleInstance, AbstractMarkedParticleInstance, wni, ForcingCollection
export AbstractForcingField, FunctionForcingField, ArrayForcingField

#using OrdinaryDiffEq: OrdinaryDiffEqCore.ODEIntegrator
using DocStringExtensions
using StaticArrays

using ..Architectures: AbstractParticleInstance, AbstractMarkedParticleInstance, AbstractODEIntegrator
using ..Architectures: AbstractBoundary

using Parameters

# ParticleInstance is the Stucture that carries each particle.
mutable struct ParticleInstance2D <: AbstractParticleInstance
        position_ij::Tuple{Int, Int}
        position_xy::Tuple{Float64, Float64}
        ODEIntegrator::Union{AbstractODEIntegrator,Nothing}
        boundary :: Bool
        on::Bool
end

function Base.show(io::IO, ow::ParticleInstance2D)
        sys_print = "ParticleEquations2D: u(x,y,t), v(x,y,t)"
        ODEint = ow.ODEIntegrator
        print(io, "ParticleInstance2D ", "\n",
                "├── position_ij: ", ow.position_ij, "\n",
                "├── position_xy: ", ow.position_xy, "\n",
                "├── boundary:    ", ow.boundary, "\n",
                "├── on:          ", ow.on, "\n",
                "├── ODEIntegrator    \n",
                "|        ├── model: ", ODEint.model!, "\n",
                "|        ├── u:     ", ODEint.u, "\n",
                "|        ├── t:     ", ODEint.t, "\n",
                "|        └── params: ", ODEint.params, "\n",
                "└── ODEIntegrator-solver    \n",
                "         ├── dt:       ", ODEint.dt, "\n",
                "         ├── has_fsal: ", ODEint.has_fsal, "\n",
                "         └── err:      ", ODEint.err, "\n")
end

mutable struct ParticleInstance1D <: AbstractParticleInstance
        position_ij::Int
        position_xy::Float64
        ODEIntegrator::AbstractODEIntegrator
        boundary::Bool
        on::Bool
end

# Debugging ParticleInstance
mutable struct MarkedParticleInstance <: AbstractMarkedParticleInstance
        Particle::AbstractParticleInstance
        time :: Float64
        state :: Vector{Any}
        errorReturnCode
end

Base.copy(s::ParticleInstance1D) = ParticleInstance1D(s.position_ij, s.position_xy, s.ODEIntegrator, s.boundary, s.on)
Base.copy(s::ParticleInstance2D) = ParticleInstance2D(s.position_ij, s.position_xy, s.ODEIntegrator, s.boundary, s.on)

# Regridding types:
"""
Weights & Index (wni) for a separable N-point-per-axis deposition stencil (N = spline order + 1).
`xi`/`yi` are the N integer node indices per axis, `xw`/`yw` the N B-spline weights per axis.
N is encoded in the type so `construct_loop` unrolls the N^2 stencil per order.
"""
struct wni{N,TI<:SVector{N,Int64},TF<:SVector{N,Float64}}
        xi::TI
        xw::TF
        yi::TI
        yw::TF
end

# Explicit outer constructor so N is inferred from the stencil width (avoids relying on
# static-parameter solving of N through the SVector{N,...} field constraints).
wni(xi::SVector{N,Int64}, xw::SVector{N,Float64}, yi::SVector{N,Int64}, yw::SVector{N,Float64}) where {N} =
        wni{N,SVector{N,Int64},SVector{N,Float64}}(xi, xw, yi, yw)

# Define Boundary Types:

struct N_Periodic{T} <: AbstractBoundary
        N::T
end

struct N_NonPeriodic{T} <: AbstractBoundary
        N::T
end

struct N_TripolarNorth{T} <: AbstractBoundary
        N::T
end

# struct N_TripolarSouth{T} <: AbstractBoundary
#         N::T
# end


abstract type AbstractForcingField end

"Function-backed forcing field f(x,y,t)."
struct FunctionForcingField{F<:Function} <: AbstractForcingField
        f::F
end

"Array-backed forcing field with optional x/y/t axes." 
struct ArrayForcingField{A<:AbstractArray,TX,TY,TT} <: AbstractForcingField
        data::A
        x::TX
        y::TY
        t::TT
end

ArrayForcingField(data::AbstractArray; x=nothing, y=nothing, t=nothing) = ArrayForcingField{typeof(data),typeof(x),typeof(y),typeof(t)}(data, x, y, t)

_axis_summary(axis) = axis === nothing ? "none" : "len=$(length(axis))"

function _show_forcing_field(x)
        if x === nothing
                return "nothing"
        elseif x isa Number
                return "$(typeof(x)) value=$(x)"
        elseif x isa FunctionForcingField
                return "FunctionForcingField($(typeof(x.f)))"
        elseif x isa ArrayForcingField
                return "ArrayForcingField(data=$(typeof(x.data)) size=$(size(x.data)), x=$(_axis_summary(x.x)), y=$(_axis_summary(x.y)), t=$(_axis_summary(x.t)))"
        elseif x isa AbstractArray
                return "$(typeof(x)) size=$(size(x))"
        elseif x isa Function
                return "Function($(typeof(x)))"
        else
                return string(typeof(x))
        end
end

@with_kw mutable struct ForcingCollection
        "Zonal wind component [m/s] - function(x,y,t), array, or interpolation object"
        u_wind::Union{Function,AbstractForcingField,AbstractArray,AbstractFloat,Nothing} = nothing
        "Meridional wind component [m/s] - function(x,y,t), array, or interpolation object"
        v_wind::Union{Function,AbstractForcingField,AbstractArray,AbstractFloat,Nothing} = nothing
        "Zonal current component [m/s] - function(x,y,t), array, or interpolation object"
        u_current::Union{Function,AbstractForcingField,AbstractArray,AbstractFloat,Nothing} = nothing
        "Meridional current component [m/s] - function(x,y,t), array, or interpolation object"
        v_current::Union{Function,AbstractForcingField,AbstractArray,AbstractFloat,Nothing} = nothing
        "Sea ice concentration [0-1] - function(x,y,t), array, or interpolation object"
        sea_ice_concentration::Union{Function,AbstractForcingField,AbstractArray,AbstractFloat,Nothing} = nothing
        "Sea ice thickness [m] - function(x,y,t), array, or interpolation object"
        sea_ice_thickness::Union{Function,AbstractForcingField,AbstractArray,AbstractFloat,Nothing} = nothing
end

function Base.show(io::IO, ::MIME"text/plain", fc::ForcingCollection)
        print(io, "ForcingCollection\n",
                "├── u_wind:                ", _show_forcing_field(fc.u_wind), "\n",
                "├── v_wind:                ", _show_forcing_field(fc.v_wind), "\n",
                "├── u_current:             ", _show_forcing_field(fc.u_current), "\n",
                "├── v_current:             ", _show_forcing_field(fc.v_current), "\n",
                "├── sea_ice_concentration: ", _show_forcing_field(fc.sea_ice_concentration), "\n",
                "└── sea_ice_thickness:     ", _show_forcing_field(fc.sea_ice_thickness))
end



end