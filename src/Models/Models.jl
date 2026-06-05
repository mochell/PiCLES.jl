module Models

export WaveGrowthModels1D, WaveGrowthModels2D, reset_boundary!

# WaveGrowthModels2D must be loaded first: WaveGrowthModels1D's `WaveGrowth1D` constructor
# returns a `WaveGrowth2D`.
include("WaveGrowthModels2D.jl")
include("WaveGrowthModels1D.jl")

using .WaveGrowthModels2D
using .WaveGrowthModels1D

end