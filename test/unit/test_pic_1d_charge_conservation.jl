"""
Quantitative 1D PIC charge-conservation checks, swept over B-spline deposition order P=1,2,3.

This mirrors the non-divergent manual cases from test/manual/T01_test_PIC_1D.jl, but deposits
through the **production** particle->grid path used by the live engine
(`ParticleInCell.compute_weights_and_index_mininal` + the `wni` `push_to_grid!`) on a thin
periodic 2D mesh, rather than the legacy standalone 1D `push_to_grid!` (which no simulation uses;
see issues #59, #60 and #38). Charge is laid out along x at a single y-row; higher orders may
spread a little in y, but partition of unity keeps the total exactly conserved, so the same ±2%
window holds for every P. Readback collapses the y-dimension to recover the per-x-node charge.
"""

using Test
using SharedArrays
using StaticArrays

using PiCLES.Grids.CartesianGrid: CartesianGridMesh1D, gridnotes_1d
import PiCLES.ParticleInCell
using PiCLES.custom_structures: N_Periodic

const NY_DEPOSIT = 5      # thin-y mesh wide enough that the cubic (P=3) y-stencil never wraps
const JROW = 3            # center row for the 1D charge line

function grid1d_axes(grid1d)
    stats = grid1d.stats
    return (Nx=stats.Nx.N, xmin=stats.xmin, xmax=stats.xmax, dx=stats.dx, x=gridnotes_1d(grid1d))
end

function integrate_charge_history(grid1d, charges_1d, xp; N::Int, cg::Float64, P::Int)
    ax = grid1d_axes(grid1d)
    Nx = ax.Nx
    q = copy(charges_1d)
    x = copy(xp)

    Bx = N_Periodic(Nx)
    By = N_Periodic(NY_DEPOSIT)
    spline = Val(P)
    S = SharedArray{Float64,3}((Nx, NY_DEPOSIT, 3))

    charge_hist = Vector{Float64}(undef, N + 1)
    charge_hist[1] = Float64(sum(q))

    for ti in 1:N
        S .= 0.0
        for i in 1:Nx
            # absolute normalized x position, offset relative to the parcel's home node i (1-based)
            znorm = (x[i] - ax.xmin) / ax.dx
            zrel = znorm - (i - 1)
            w = ParticleInCell.compute_weights_and_index_mininal((i, JROW), zrel, 0.0, spline)
            ParticleInCell.push_to_grid!(S, SVector{3,Float64}(q[i], 0.0, 0.0), w, Bx, By)
        end

        # recover per-x-node charge by collapsing the (small) y-spread, then self-advect
        q = vec(sum(Array(S)[:, :, 1], dims=2))
        x = ax.x .+ cg .* q

        charge_hist[ti + 1] = Float64(sum(q))
    end

    return charge_hist
end

function init_box_case(axes; base::Float64)
    xp = axes.x .+ axes.dx * 1.5
    charges_1d = fill(base, axes.Nx)
    charges_1d[40:Int(ceil(axes.Nx * 2 / 3))] .= 1.0
    return xp, charges_1d
end

function init_sin_left_case(axes)
    xp = axes.x .+ axes.dx * 0.5
    charges_1d = sin.(3 * 2 * pi * xp ./ axes.xmax) * 0.2 .+ 0.2
    return xp, charges_1d
end

function max_relative_charge_drift(charge_hist::Vector{Float64})
    q0 = charge_hist[1]
    return maximum(abs.(charge_hist .- q0)) / abs(q0)
end

@testset "PIC 1D total charge conservation (non-divergent cases)" begin
    grid1d = CartesianGridMesh1D(0.0, 20.0, 101; Ny=3, periodic_boundary=true)
    axes = grid1d_axes(grid1d)

    # N is 4x the manual script values: 50 -> 200, 100 -> 400
    cases = [
        (name="box right, zero base", N=200, cg=0.2, init=axes -> init_box_case(axes; base=0.0)),
        (name="box left, zero base", N=200, cg=-0.2, init=axes -> init_box_case(axes; base=0.0)),
        (name="sin left, nonlinear advection", N=200, cg=-0.3, init=init_sin_left_case),
    ]

    @testset "spline_order=$(P)" for P in (1, 2, 3)
        for case in cases
            xp, charges_1d = case.init(axes)
            charge_hist = integrate_charge_history(grid1d, charges_1d, xp; N=case.N, cg=case.cg, P=P)
            drift = max_relative_charge_drift(charge_hist)
            @test drift <= 0.02
        end
    end
end
