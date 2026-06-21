"""
Benchmark the particle->grid B-spline deposition across orders P = 1, 2, 3 (issues #59, #60).

Run from the repo root:
    julia --project=. test/benchmark/bench_bspline_deposition.jl

Measures per-deposit wall time and allocations through the production path
(`compute_weights_and_index_mininal` + the `wni` `push_to_grid!`) and writes a scaling plot to
`plots/bspline_deposition_scaling.png`.

Expectations:
- cost scales roughly with the number of stencil points (P+1)^2 = 4 / 9 / 16 for P = 1 / 2 / 3,
- allocations scale *linearly with the number of stencil points* and stay constant per point
  across orders. That per-point cost is the pre-existing `grid[i,j,:] += w*charge` array-slice
  assignment (unchanged by this work); the linear (not super-linear) scaling and the absence of
  any extra per-order overhead is what confirms the Val{P} deposit specializes correctly with no
  dynamic dispatch on the order.

Not included in runtests.jl.
"""

using BenchmarkTools
using SharedArrays
using StaticArrays
using Plots
using Plots.PlotMeasures: mm
import PiCLES.ParticleInCell as PIC
using PiCLES.custom_structures: N_Periodic

gr()

const Nx, Ny = 64, 64
const Bx, By = N_Periodic(Nx), N_Periodic(Ny)
const CHARGE = SVector{3,Float64}(2.5, 0.3, -0.7)
const IJ = (32, 32)
const XP, YP = 0.37, 0.62
const ORDERS = (1, 2, 3)

# one deposit through the full production path (weights + push), order P
function deposit!(S, spline::Val)
    w = PIC.compute_weights_and_index_mininal(IJ, XP, YP, spline)
    PIC.push_to_grid!(S, CHARGE, w, Bx, By)
    return nothing
end

function main()
    S = SharedArray{Float64,3}((Nx, Ny, 3))
    pts = Int[]; tmin = Float64[]; tmed = Float64[]; allocs = Int[]

    for P in ORDERS
        spline = Val(P)
        S .= 0.0
        deposit!(S, spline)                       # warm up / compile
        b = @benchmark deposit!($S, $spline) samples = 10000 evals = 100
        a = @allocated deposit!(S, spline)
        push!(pts, (P + 1)^2)
        push!(tmin, minimum(b).time)
        push!(tmed, median(b).time)
        push!(allocs, a)
        println("P=$P  (stencil $(P+1)^2 = $((P+1)^2) pts):  ",
            "min ", minimum(b).time, " ns,  median ", median(b).time, " ns,  ",
            "allocs/deposit = ", a)
    end

    # ---- scaling plot: time and allocations vs number of stencil points -------
    # ideal references anchored at the P=1 point: cost proportional to stencil points.
    ideal_t = tmed[1] .* pts ./ pts[1]
    ideal_a = allocs[1] .* pts ./ pts[1]
    labels = ["P=$P" for P in ORDERS]

    p1 = plot(pts, tmed; marker=:circle, ms=6, lw=2, label="median",
        xlabel="stencil points (P+1)²", ylabel="time per deposit [ns]",
        title="deposit time", legend=:topleft,
        left_margin=12mm, bottom_margin=10mm, top_margin=5mm, right_margin=5mm)
    plot!(p1, pts, tmin; marker=:diamond, ms=5, lw=1, ls=:dash, label="min")
    plot!(p1, pts, ideal_t; ls=:dot, lw=2, color=:gray, label="∝ stencil pts (ideal)")
    annotate!(p1, [(pts[i], tmed[i], text("  " * labels[i], 8, :left, :bottom)) for i in eachindex(pts)])

    p2 = plot(pts, allocs; marker=:circle, ms=6, lw=2, label="measured",
        xlabel="stencil points (P+1)²", ylabel="allocations per deposit [bytes]",
        title="deposit allocations", legend=:topleft,
        left_margin=14mm, bottom_margin=10mm, top_margin=5mm, right_margin=5mm)
    plot!(p2, pts, ideal_a; ls=:dot, lw=2, color=:gray, label="∝ stencil pts (ideal)")
    annotate!(p2, [(pts[i], allocs[i], text("  " * labels[i], 8, :left, :bottom)) for i in eachindex(pts)])

    fig = plot(p1, p2; layout=(1, 2), size=(1100, 480),
        plot_title="B-spline PIC deposition scaling (P=1,2,3)")

    outdir = joinpath(@__DIR__, "..", "..", "plots")
    mkpath(outdir)
    outpath = joinpath(outdir, "bspline_deposition_scaling.png")
    savefig(fig, outpath)
    println("wrote ", abspath(outpath))
end

main()
