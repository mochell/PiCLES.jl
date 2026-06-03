
using ...Architectures: BoundaryType
using ...custom_structures: N_NonPeriodic
using Statistics

"""
interior_boundary(mask::Array{Bool,2})
function that returns the interior boundary points of the mask
    mask == 1: ocean
    mask == 0: land

    return bmask: Array{Bool, 2} - mask with interior boundary points == 1, else 0

"""
function interior_boundary(mask::Union{Array{Bool,2}, BitMatrix})
    # interios boundary points
    bmask = zeros(Bool, size(mask))
    for dims in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        bmask += circshift(mask, dims) .&& .!mask
    end
    bmask = bmask .!= 0
    return bmask
end

"""
make_boundaries(mask::Array{Bool, 2})
function that returns total mask with
    - 0: land
    - 1: ocean
    - 2: land boundary
    - 3: grid boundary

    inputs:
    mask: Array{Bool, 2} - ocean mask of the grid (1 ocean , 0 land), no Int matrix allowed! 

    returns:
    total_mask: Array{Int, 2} - mask with ocean, land, land boundary and grid boundary
"""
function make_boundaries(mask::Union{Array{Bool,2},BitMatrix}, Nx::BoundaryType, Ny::BoundaryType)

    bmask = interior_boundary(mask)

    total_mask = mask + 2 * bmask

    if Nx isa N_NonPeriodic
        total_mask[1, :] .= 3
        total_mask[end, :] .= 3
    end

    if Ny isa N_NonPeriodic
        total_mask[:, 1] .= 3
        total_mask[:, end] .= 3
    end

    return total_mask
end


"""
make_boundary_lists(total_mask::Array{Int, 2})
function that returns lists of boundary nodes (tuples of indixes)

    inputs:
    total_mask: Array{Int, 2} - mask with ocean, land, land boundary and grid boundary

    returns:
    ocean: Vector{CartesianIndex} - ocean indicies
    land_boundary: Vector{CartesianIndex} - land boundary indicies
    grid_boundary: Vector{CartesianIndex} - grid boundaries indicies
    land: Vector{CartesianIndex} - land indicies
"""
function make_boundary_lists(total_mask)

    land_boundary = findall(total_mask .== 2)
    grid_boundary = findall(total_mask .== 3)
    ocean         = findall(total_mask .== 1)
    #land          = findall(total_mask .== 0)

    return (ocean     = ocean,
        land_boundary = land_boundary,
        grid_boundary = grid_boundary)
    #land = land)
end


# mutable struct BList
#     ocean::Vector{CartesianIndex}
#     land_boundary::Vector{CartesianIndex}
#     grid_boundary::Vector{CartesianIndex}
#     land::Vector{CartesianIndex}
# end




"""
function mask_circle!(mask, grid, pp_ij , radius)
    adds a circular masked area to the existing mask 

    mask    1 ocean, 0 is land
    grid    grid instance
    pp)ij   i,j position of the center
    radius  radius in degree

    returns mask with circular masked areas
"""
function mask_circle!(mask, grid, pp_ij, radius)

    #in_circle(xy, pp, r) = (xy[1] - pp[1])^2 + (xy[2] - pp[2])^2 < r^2

    mask_circle!(mask, grid.data.x, grid.data.y, pp_ij, radius)

    # pp = grid.data.x[pp_ij[1], pp_ij[2]], grid.data.y[pp_ij[1], pp_ij[2]]

    # for ij in CartesianIndices(mask)
    #     x, y = grid.data.x[ij[1], ij[2]], grid.data.y[ij[1], ij[2]]
    #     # @info x, y, in_circle( (x, y) , pp, 5)
    #     if in_circle((x, y), pp, radius)
    #         mask[ij[1], ij[2]] = false
    #     end
    # end
    # return mask
end


function mask_circle!(mask, xx, yy, pp_ij, radius)

    in_circle(xy, pp, r) = (xy[1] - pp[1])^2 + (xy[2] - pp[2])^2 < r^2

    pp = xx[pp_ij[1], pp_ij[2]], yy[pp_ij[1], pp_ij[2]]

    for ij in CartesianIndices(mask)
        x, y = xx[ij[1], ij[2]], yy[ij[1], ij[2]]
        # @info x, y, in_circle( (x, y) , pp, 5)
        if in_circle((x, y), pp, radius)
            mask[ij[1], ij[2]] = false
        end
    end
    # return mask
end