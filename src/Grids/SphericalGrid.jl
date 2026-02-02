module SphericalGrid

using ...Architectures: AbstractGrid, AbstractGridStatistics, CartesianGrid1D, SphericalGrid, CartesianGridStatistics, AbstractBoundary, BoundaryType, SphericalGridStatistics, SphericalGrid2D
using ...custom_structures: N_Periodic, N_NonPeriodic, N_TripolarNorth

#using LinearAlgebra
using StructArrays
using StaticArrays

include("mask_utils.jl")
include("spherical_grid_corrections.jl")

    # Generate Spherical mesh statistics on rectangle `dimx`x `dimy` with `nx` x `ny` points

    # - `nx` : indices are in [1:nx]
    # - `ny` : indices are in [1:ny]
    # - `dimx = xmax - xmin`
    # - `dimy = ymax - ymin`
    # - `x, y` : node positions
    # - `dx, dy` : step size


## Utilites

function cal_dx_degree(XX)
    dx = zeros(size(XX))
    dx[2:end-1, :] = (XX[3:end, :] - XX[1:end-2, :]) / 2
    dx[1, :] = (XX[2, :] - XX[1, :])
    dx[end, :] = (XX[end, :] - XX[end-1, :])
    return dx
end

function cal_dy_degree(YY)
    dy = zeros(size(YY))
    dy[:, 2:end-1] = (YY[:, 3:end] - YY[:, 1:end-2]) / 2
    dy[:, 1] = (YY[:, 2] - YY[:, 1])
    dy[:, end] = (YY[:, end] - YY[:, end-1])
    return dy
end


"""
    cal_dx_meters(XX, YY)

    This function calculates the distance in meters between two points on the sphere in the longitude direction.

    # Arguments
    - `XX::Array{Float64, 2}`: 2D array of x-coordinates in degrees.
    - `YY::Array{Float64, 2}`: 2D array of y-coordinates in degrees.

    # Returns
    - `Array{Float64, 2}`: 2D array of distances in meters between points in the longitude direction.
"""
function cal_dx_meters(XX, YY)

    R = 6371.0e3 #meters 
    R_meridian = R * cos.(YY * pi / 180) #meters
    return cal_dx_degree(XX) * pi / 180 .* R_meridian
end


"""    cal_dy_meters(YY)

    This function calculates the distance in meters between two points on the sphere in the latitude direction.

    # Arguments
    - `YY::Array{Float64, 2}`: 2D array of y-coordinates in degrees.

    # Returns
    - `Array{Float64, 2}`: 2D array of distances in meters between points in the latitude direction.
"""
function cal_dy_meters(YY)
    R = 6371.0e3 #meters 
    return cal_dy_degree(YY) * pi / 180 .* R
end



"""
    TwoDSphericalGridStatistics

A module for set the statistics on a 2D spherical grid. This module provides variables needed to define the spherical grid. 

    inputs
    - `xmin` : minimum longitude
    - `xmax` : maximum longitude
    - `Nx` : number of grid points in the x direction
    - `ymin` : minimum latitude
    - `ymax` : maximum latitude
    - `Ny` : number of grid points in the y direction
    - `angle` : angle of rotation of the grid in degrees
    - `periodic_boundary` : Tuple{Bool,Bool} = (false, false) - if the grid is periodic in the x and y directions

# Example
grid =TwoDSphericalGridStatistics(10, 20, 11, 10, 30, 41)


"""
struct TwoDSphericalGridStatistics <: SphericalGridStatistics

    Nx::BoundaryType
    Ny::BoundaryType
    Ndx::Int
    Ndy::Int

    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64

    dimx::Float64 # size in x
    dimy::Float64 # szie in y

    dx_deg::Float64 #  resolution in degrees in x
    dy_deg::Float64 #  resolution in degrees in y

    angle_dx::Float64

    mask_value::Int



    function TwoDSphericalGridStatistics(xmin, xmax, Nx::Int, ymin, ymax, Ny::Int; mask_value=1, angle=0.0, periodic_boundary::Tuple{Bool,Bool}=(false, false))
        dimx = xmax - xmin
        dimy = ymax - ymin

        Ndx = Nx - 1
        Ndy = Ny - 1

        Nx = periodic_boundary[1] ? N_Periodic(Nx) : N_NonPeriodic(Nx)
        Ny = periodic_boundary[2] ? N_Periodic(Ny) : N_NonPeriodic(Ny)

        dx_deg = dimx / Ndx
        dy_deg = dimy / Ndy

        return new(Nx, Ny, Ndx, Ndy, xmin, xmax, ymin, ymax, dimx, dimy, dx_deg, dy_deg, angle, mask_value)
    end
end



# %%

"""
    Module: Grids.SphericalGrid

    inputs:
    data: StructArray{<:Any}
    stats: TwoDSphericalGridStatistics
    ProjetionKernel: Function that returns the projection kernel for the grid


# Examples

"""
struct TwoDSphericalGridMesh <: SphericalGrid2D
    data::StructArray{<:Any}
    stats::TwoDSphericalGridStatistics
    ProjetionKernel::Function
    PropagationCorrection::Function
end

function TwoDSphericalGridMesh(grid::SphericalGridStatistics; mask=nothing, total_mask=nothing)

    x = collect(range(grid.xmin, stop=grid.xmax, length=grid.Nx.N))
    y = collect(range(grid.ymin, stop=grid.ymax, length=grid.Ny.N))

    XX = transpose(reshape(repeat(x, inner=length(y)), length(y), length(x)))
    YY = transpose(reshape(repeat(y, outer=length(x)), length(y), length(x)))

    dx = cal_dx_meters(XX, YY) # meters, centered around grid vertex
    dy = cal_dy_meters(YY) # meters, centered around grid vertex
    area = dx .* dy # meters^2

    if isnothing(mask)
        mask = ones(Bool, size(XX))#fill(1, size(XX))
    else
        mask = mask
    end

    if isnothing(total_mask)
        mask = make_boundaries(mask, grid.Nx::BoundaryType, grid.Ny::BoundaryType)
    else
        mask = total_mask
    end
    # mask = make_boundaries(mask)

    return StructArray(
        x=XX,
        y=YY,
        dx=dx,
        dy =dy,
        area=area,
        mask=mask
    )

end

# initalization
function TwoDSphericalGridMesh(xmin, xmax, Nx::Int, ymin, ymax, Ny::Int; mask=nothing, angle=0.0, periodic_boundary=(false, false))
    GS = TwoDSphericalGridStatistics(xmin, xmax, Nx, ymin, ymax, Ny;  angle=angle, periodic_boundary=periodic_boundary)
    GMesh = TwoDSphericalGridMesh(GS, mask=mask)
    return TwoDSphericalGridMesh(GMesh, GS, ProjetionKernel, SphericalPropagationCorrection)
end

## projection Kernel for this grid:
function ProjetionKernel(Gdata::StructArray)
    cos_lat = cos.(Gdata.dy * pi / 180)
    # R = 6371.0e3 #meters 

    # in normalized degree lon/latitude
    #@SArray
    # proj(cos_lat, dx, dy, mask, R) = mask == 1 ? SMatrix{2,2}([
    #     180/(R*cos_lat*pi*dx) 0;
    #     0 180/(R * pi * dy)
    #     ]) : nothing

    proj(cos_lat, dx, dy, mask, R) = mask == 1 ? SMatrix{2,2}([
        1/(cos_lat*dx) 0;
        0 1/(dy)
        ]) : nothing
    
    M = map(proj, cos_lat, Gdata.dx, Gdata.dy, Gdata.mask, R)
    return M
end


function ProjetionKernel(Gi::NamedTuple, stats::TwoDSphericalGridStatistics)
    
    cos_lat = cos.(Gi.dy * pi / 180)
    # R = 6371.0e3 #meters

    # in normalized degree lon/latitude
    return SMatrix{2,2}([ 
        1/(cos_lat * Gi.dx) 0;
        0              1/Gi.dy
                        ])
end


# # alias for GRid object
ProjetionKernel(G::TwoDSphericalGridMesh) = ProjetionKernel(G.stats)


end