module custom_structures

export ParticleInstance1D, ParticleInstance2D, MarkedParticleInstance, AbstractParticleInstance, AbstractMarkedParticleInstance, wni

#using OrdinaryDiffEq: OrdinaryDiffEqCore.ODEIntegrator
using DocStringExtensions
using StaticArrays

using ..Architectures: AbstractParticleInstance, AbstractMarkedParticleInstance, AbstractODEIntegrator
using ..Architectures: AbstractBoundary

# ParticleInstance is the Stucture that carries each particle.
mutable struct ParticleInstance2D <: AbstractParticleInstance
        position_ij::Tuple{Int, Int}
        position_xy::Tuple{Float64, Float64}
        ODEIntegrator::Union{AbstractODEIntegrator,Nothing}
        boundary :: Bool
        on::Bool
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
"""Weights & Index (wni) FieldVector """
struct wni{TI<:SVector,TF<:SVector} <: FieldVector{4,SVector}
        xi::TI
        xw::TF
        yi::TI
        yw::TF
end

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

end