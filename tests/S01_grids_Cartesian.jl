using Pkg
Pkg.activate("PiCLES/")

using PiCLES.Grids.CartesianGrid: TwoDCartesianGridStatistics, TwoDCartesianGridMesh, ProjectionKernel
#using PiCLES.Grids.TripolarGridMOM6: TripolarGridMOM6, ProjectionKernel
using Plots

using Revise
using BenchmarkTools

Revise.retry()


# %% testing
grid_stats = TwoDCartesianGridStatistics(0.0, 1.0, 40, 0.0, 1.0, 25)
grid_test = TwoDCartesianGridMesh(grid_stats)
# merge to mesh
TwoDCartesianGridMesh(grid_test, grid_stats)

# short hand for doing both at ones
G = TwoDCartesianGridMesh(0.0, 1.0, 40, 0.0, 1.0, 25)

# short hand
G = TwoDCartesianGridMesh(1, 3, 1, 2)

# add angle definition
G = TwoDCartesianGridMesh(0.0, 39, 40, 0.0, 24, 25; angle=0.5)
# test pojection
M = ProjectionKernel(G)

# add mask
mask = ones(Bool, size(G.data.x)) # 1 is ocean, 0 is land (?)
mask[10:15, 10:15] .=0

# reset mesh
grid_w_mask = TwoDCartesianGridMesh(grid_stats; mask=mask)
GM_mask = TwoDCartesianGridMesh(grid_w_mask, grid_stats)

# test
scatter(GM_mask.data.x[GM_mask.data.mask], GM_mask.data.y[GM_mask.data.mask], color="black", legend=false, markersize=1)

