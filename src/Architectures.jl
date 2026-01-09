module Architectures

using SharedArrays
using StaticArrays


export AbstractGrid, AbstractODESettings, AbstractParticleInstance, AbstractMarkedParticleInstance, Abstract1DModel, Abstract2DModel, Abstract2DStochasticModel, AbstractModel, AbstractStore, AbstractParticleSystem, StateTypeL1, IDConstantsInstance, ScgConstantsInstance, CartesianGrid, CartesianGrid1D, CartesianGrid2D, TripolarGrid, Grid2D, MeshGrids, MeshGridStatistics

export StandardRegular1D_old, StandardRegular2D_old

export AbstractBoundary, BoundaryType

abstract type AbstractGrid end

abstract type AbstractGridStatistics <: AbstractGrid end

abstract type CartesianGrid <: AbstractGrid end
abstract type CartesianGrid1D <: CartesianGrid end
abstract type CartesianGrid2D <: CartesianGrid end
abstract type CartesianGridStatistics <: AbstractGridStatistics end

#Spherical Grids
abstract type SphericalGrid <: AbstractGrid end
abstract type SphericalGrid2D <: SphericalGrid end
abstract type SphericalGridStatistics <: AbstractGridStatistics end

abstract type TripolarGrid <: AbstractGrid end
# abstract type MOM6_2_3 <: TripolarGrid end
abstract type TripolarGridStatistics <: AbstractGridStatistics end

abstract type StandardRegular1D_old <: AbstractGrid end
abstract type StandardRegular2D_old <: AbstractGrid end

Grid2D = Union{CartesianGrid2D,SphericalGrid2D,StandardRegular2D_old}

# used for model intialization
MeshGrids = Union{CartesianGrid2D,TripolarGrid,SphericalGrid2D}
MeshGridStatistics = Union{CartesianGridStatistics,TripolarGridStatistics,SphericalGridStatistics}

abstract type AbstractODESettings end
abstract type AbstractParticleInstance end
abstract type AbstractMarkedParticleInstance end

abstract type IDConstantsInstance end
abstract type ScgConstantsInstance end


"""
    AbstractModel
Abstract supertype for models.
"""
abstract type AbstractModel{TS} end

abstract type Abstract1DModel <: AbstractModel{Nothing} end
abstract type Abstract2DModel <: AbstractModel{Nothing} end
abstract type Abstract2DStochasticModel <: Abstract2DModel end

#All posiible types of a single-layer StateVectors
StateTypeL1 = Union{SharedArray{Float64,3},MArray}

abstract type AbstractStore end

abstract type AbstractParticleSystem end


# %% abstract Boundary Integer types
abstract type AbstractBoundary <: Integer end
Base.show(io::IO, obj::AbstractBoundary) = print(io, "Int=", obj.N, " ", typeof(obj))

BoundaryType = Union{AbstractBoundary,Integer}


end