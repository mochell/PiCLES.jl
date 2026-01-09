module Models

export WaveGrowthModels1D, WaveGrowthModels2D, GeometricalOpticsModels, reset_boundary!

include("WaveGrowthModels1D.jl")
include("WaveGrowthModels2D.jl")
include("GeometricalOpticsModels.jl")

using .WaveGrowthModels1D
using .WaveGrowthModels2D
using .GeometricalOpticsModels

end