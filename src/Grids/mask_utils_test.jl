using Pkg
Pkg.activate("PiCLES/")

using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, ProjectionKernel, TwoDCartesianGridStatistics
using PiCLES.Grids


# %%

grid = TwoDCartesianGridMesh(400e3, 21, 200e3, 11)

# % Make fake mask
mask = ones(Bool, size(grid.data.x)) # 1 is ocean, 0 is land (?)
mask[10:20, 5:10] .= 0
# mask  = .! mask # to make one active block

# reset mesh with amsk 
gridstats_mask = TwoDCartesianGridMesh(grid.stats; mask=mask)
grid = TwoDCartesianGridMesh(gridstats_mask, grid.stats, ProjectionKernel)


# %%
mask = grid.data.mask
total_mask = Grids.make_boundaries(mask)

blists = Grids.make_boundary_lists(total_mask, periodic_boundary=true)

length(blists.land_boundary) + length(blists.grid_boundary) + length(blists.ocean) + sum(total_mask .== 0) == length(mask)
length(blists.land_boundary) == 30
length(blists.grid_boundary) == 60

blists = Grids.make_boundary_lists(total_mask, periodic_boundary=false)

length(blists.land_boundary) + length(blists.ocean) + sum(total_mask .== 0) == length(mask)
length(blists.land_boundary) == 90
length(blists.grid_boundary) == 90

