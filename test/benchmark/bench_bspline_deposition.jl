"""
Benchmark the particle->grid B-spline deposition across orders P = 1, 2, 3 (issues #59, #60).

Run from the repo root:
    julia --project=. test/benchmark/bench_bspline_deposition.jl

Reports per-deposit wall time and allocations. Expectations:
- cost scales roughly as (P+1)^2 (4 / 9 / 16 stencil points for P = 1 / 2 / 3),
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
import PiCLES.ParticleInCell as PIC
using PiCLES.custom_structures: N_Periodic

const Nx, Ny = 64, 64
const Bx, By = N_Periodic(Nx), N_Periodic(Ny)
const CHARGE = SVector{3,Float64}(2.5, 0.3, -0.7)
const IJ = (32, 32)
const XP, YP = 0.37, 0.62

# one deposit through the full production path (weights + push), order P
function deposit!(S, spline::Val)
    w = PIC.compute_weights_and_index_mininal(IJ, XP, YP, spline)
    PIC.push_to_grid!(S, CHARGE, w, Bx, By)
    return nothing
end

function main()
    S = SharedArray{Float64,3}((Nx, Ny, 3))
    for P in (1, 2, 3)
        spline = Val(P)
        S .= 0.0
        deposit!(S, spline)                       # warm up / compile
        b = @benchmark deposit!($S, $spline) samples = 10000 evals = 100
        allocs = @allocated deposit!(S, spline)
        println("P=$P  (stencil $(P+1)^2 = $((P+1)^2) pts):  ",
            "min ", minimum(b).time, " ns,  median ", median(b).time, " ns,  ",
            "allocs/deposit = ", allocs)
    end
end

main()
