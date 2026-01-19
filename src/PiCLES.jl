"""
Main module for 'PiCLES.jl'
"""
module PiCLES

# external modules
using HDF5
using DocStringExtensions
#using Printf

export 
    # run simulation
    Simulations,
    
    #Operators/Core
    Operators,

    # models
    Models,

    # Particle Systems
    ParticleSystems,

    # grids
    ParticleMesh,
    Grids,

    # utils
    Utils, Debugging, FetchRelations, ParticleTools, WindEmulator, visualization


    #externals


include("Architectures.jl")
using .Architectures

include("custom_structures.jl")

include("ParticleSystems/ParticleSystems.jl")
using .ParticleSystems

include("FetchRelations.jl")
using .FetchRelations

include("Grids/Grids.jl")
using .Grids

include("ParticleMesh.jl")
include("ParticleInCell.jl")
using .ParticleMesh
using .ParticleInCell

include("Operators/Operators.jl")
include("Simulations/Simulations.jl")

#include("Grids/Grids.jl")

#include("Particles/Particles.jl")

include("Utils/Utils.jl")
using .Utils


using .Simulations


include("Models/Models.jl")
using .Models

include("visualization/Plotting.jl")
using .Plotting

end