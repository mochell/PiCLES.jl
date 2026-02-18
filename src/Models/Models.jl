module Models

export WaveGrowthModels1D, WaveGrowthModels2D, GeometricalOpticsModels, reset_boundary!

include("WaveGrowthModels1D.jl")
using .WaveGrowthModels1D
include("WaveGrowthModels2D.jl")
using .WaveGrowthModels2D
include("GeometricalOpticsModels.jl")
using .GeometricalOpticsModels


end