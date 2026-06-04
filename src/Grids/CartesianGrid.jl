module CartesianGrid

using ...Architectures: AbstractGrid, AbstractGridStatistics, CartesianGrid1D, CartesianGrid2D, CartesianGridStatistics, AbstractBoundary, BoundaryType
using ...custom_structures: N_Periodic, N_NonPeriodic, N_TripolarNorth

#using LinearAlgebra
using StructArrays
using StaticArrays

include("mask_utils.jl")
include("spherical_grid_corrections.jl")


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
    ProjetionKernel::Function
    PropagationCorrection::Function
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
    return TwoDCartesianGridMesh(GMesh, GS, ProjetionKernel, SphericalPropagationCorrection_dummy)
end

# short hand for function above
TwoDCartesianGridMesh(dimx, nx::Int, dimy, ny::Int ; angle=0.0, periodic_boundary = (false, false)) = 
TwoDCartesianGridMesh( 0.0, dimx, nx, 0.0, dimy, ny; mask=nothing, angle=angle, periodic_boundary = periodic_boundary)


function ProjetionKernel(stats::CartesianGridStatistics)
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
ProjetionKernel(Gi::NamedTuple, stats::CartesianGridStatistics) = ProjetionKernel(stats)
# alias for GRid object
ProjetionKernel(G::TwoDCartesianGridMesh) = ProjetionKernel(G.stats)


# %% 1D version -------------------------------------------------------------
#
# A 1D run is realised as a degenerate 2D mesh with a thin, periodic y-extent
# (Ny ≥ 2 — the PIC interpolation and `dy = dimy/(Ny-1)` both need at least two
# y-nodes). Combined with a v-wind of 0 and c̄_y seeded to 0, the particle
# equations keep every y-row identical and c_y ≡ 0, so the whole 2D engine
# (core_2D / mapping_2D / WaveGrowth2D / time_step!) runs unchanged. `CartesianGridMesh1D`
# subtypes `CartesianGrid2D` so the engine accepts it transparently; output code
# dispatches on the concrete type to squeeze the singleton y-dimension.

struct CartesianGridMesh1D <: CartesianGrid2D
    data::StructArray{<:Any}
    stats::TwoDCartesianGridStatistics
    ProjetionKernel::Function
    PropagationCorrection::Function
end

"""
    CartesianGridMesh1D(xmin, xmax, Nx::Int; Ny=3, mask=nothing, angle=0.0, periodic_boundary=true)

Build a 1-spatial-dimension grid on `[xmin, xmax]` with `Nx` points, backed by a thin
`Ny`-row (default 3) periodic-in-y mesh so the 2D engine can run it directly. The y-axis is
always periodic; `periodic_boundary` controls the x-axis. `Ny` must be ≥ 2.
"""
function CartesianGridMesh1D(xmin, xmax, Nx::Int; Ny::Int=3, mask=nothing, angle=0.0, periodic_boundary::Bool=true)
    Ny >= 2 || throw(ArgumentError("CartesianGridMesh1D requires Ny ≥ 2 for PIC interpolation; got Ny=$Ny"))
    dx = (xmax - xmin) / (Nx - 1)
    ymin, ymax = 0.0, dx * (Ny - 1)   # square-ish cells; thin extent in y
    GS = TwoDCartesianGridStatistics(xmin, xmax, Nx, ymin, ymax, Ny; angle=angle, periodic_boundary=(periodic_boundary, true))
    GMesh = TwoDCartesianGridMesh(GS, mask=mask)
    return CartesianGridMesh1D(GMesh, GS, ProjetionKernel, SphericalPropagationCorrection_dummy)
end

# short hand: domain length + point count, x starting at 0
CartesianGridMesh1D(dimx, nx::Int; kwargs...) = CartesianGridMesh1D(0.0, dimx, nx; kwargs...)

ProjetionKernel(G::CartesianGridMesh1D) = ProjetionKernel(G.stats)

"""
    gridnotes_1d(grid::CartesianGridMesh1D)

Return the 1D vector of x node positions (one entry per `Nx`), squeezing the thin y-extent.
"""
gridnotes_1d(grid::CartesianGridMesh1D) = vec(grid.data.x[:, 1])


end # module