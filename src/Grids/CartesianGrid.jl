module CartesianGrid

using ...Architectures: AbstractGrid, AbstractGridStatistics, CartesianGrid1D, CartesianGrid2D, CartesianGridStatistics, AbstractBoundary, BoundaryType
using ...custom_structures: N_Periodic, N_NonPeriodic, N_TripolarNorth

#using LinearAlgebra
using StructArrays
using StaticArrays

include("mask_utils.jl")
# include("spherical_grid_corrections.jl")00


function PropagationCorrection_dummy(ij_mesh::NamedTuple, gridstats::AbstractGridStatistics)
    PropagationCorrection_dummy = x -> 0.0
    return PropagationCorrection_dummy
end

"""
    CartesianGrid2D( dimx, nx, dimy, ny)

    Generate a cartesians mesh on rectangle `dimx`x `dimy` with `nx` x `ny` points

    - `nx` : indices are in [1:nx]
    - `ny` : indices are in [1:ny]
    - `dimx = xmax - xmin`
    - `dimy = ymax - ymin`
    - `x, y` : node positions
    - `dx, dy` : step size
"""
struct TwoDCartesianGridStatistics <: CartesianGridStatistics
    
    Nx::BoundaryType
    Ny::BoundaryType
    Ndx::Int
    Ndy::Int

    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64

    dimx::Float64
    dimy::Float64

    dx::Float64
    dy::Float64

    area::Float64
    angle_dx::Float64

    function TwoDCartesianGridStatistics(xmin, xmax, Nx::Int, ymin, ymax, Ny::Int; angle=0.0, periodic_boundary::Tuple{Bool, Bool}=(false, false))
        dimx = xmax - xmin
        dimy = ymax - ymin

        Ndx = Nx - 1
        Ndy = Ny - 1

        dx = dimx / Ndx
        dy = dimy / Ndy

        area = dx * dy

        Nx = periodic_boundary[1] ? N_Periodic(Nx) : N_NonPeriodic(Nx)
        Ny = periodic_boundary[2] ? N_Periodic(Ny) : N_NonPeriodic(Ny)

        return new(Nx, Ny, Ndx, Ndy, xmin, xmax, ymin, ymax, dimx, dimy, dx, dy, area, angle)
    end
end


struct TwoDCartesianGridMesh <: CartesianGrid2D
    data::StructArray{<:Any}
    stats::TwoDCartesianGridStatistics
    ProjectionKernel::Function
    PropagationCorrection::Function
end

function TwoDCartesianGridMesh(data::StructArray{<:Any}, stats::TwoDCartesianGridStatistics, ProjectionKernel::Function)
    return TwoDCartesianGridMesh(data, stats, ProjectionKernel, PropagationCorrection_dummy)
end


function TwoDCartesianGridMesh(grid::CartesianGridStatistics; mask=nothing, total_mask=nothing)

    x = collect(range(grid.xmin, stop=grid.xmax, step=grid.dx))
    y = collect(range(grid.ymin, stop=grid.ymax, step=grid.dy))

    XX = transpose(reshape(repeat(x, inner=length(y)), length(y), length(x)))
    YY = transpose(reshape(repeat(y, outer=length(x)), length(y), length(x)))

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
        mask=mask
        )

end

# initalization
function TwoDCartesianGridMesh(      xmin, xmax, Nx::Int, ymin, ymax, Ny::Int; mask=nothing, angle=0.0, periodic_boundary = (false, false))
    GS = TwoDCartesianGridStatistics(xmin, xmax, Nx, ymin, ymax, Ny                        ; angle = angle, periodic_boundary = periodic_boundary)
    GMesh = TwoDCartesianGridMesh(GS, mask= mask)
    return TwoDCartesianGridMesh(GMesh, GS, ProjectionKernel, PropagationCorrection_dummy)
end

# short hand for function above
TwoDCartesianGridMesh(dimx, nx::Int, dimy, ny::Int ; angle=0.0, periodic_boundary = (false, false)) = 
TwoDCartesianGridMesh( 0.0, dimx, nx, 0.0, dimy, ny; mask=nothing, angle=angle, periodic_boundary = periodic_boundary)


function ProjectionKernel(stats::CartesianGridStatistics)
    if stats.angle_dx == 0.0
        M = [
            1/stats.dx 0;
            0 1/stats.dy
        ]
    else
        cosa = cos(stats.angle_dx * pi / 180)
        sina = sin(stats.angle_dx * pi / 180)

        M = @SArray [
            cosa/stats.dx sina/stats.dy;
            sina/stats.dx cosa/stats.dy
        ]
    end
    return M
end

# alias for initialization call
ProjectionKernel(Gi::NamedTuple, stats::CartesianGridStatistics) = ProjectionKernel(stats)
# alias for GRid object
ProjectionKernel(G::TwoDCartesianGridMesh) = ProjectionKernel(G.stats)


# %% ADD 1D version here



end # module