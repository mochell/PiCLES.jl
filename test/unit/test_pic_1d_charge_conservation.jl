"""
Quantitative 1D PIC conservation checks.

This test mirrors the non-divergent manual cases from
test/manual/T01_test_PIC_1D.jl and verifies total charge remains within 2%
of the initial value over an integration that is 4x longer than in the manual
diagnostic script.
"""

using Test
using SharedArrays

using PiCLES.Grids.CartesianGrid: CartesianGridMesh1D, gridnotes_1d
import PiCLES.ParticleInCell

function grid1d_axes(grid1d)
    stats = grid1d.stats
    return (Nx=stats.Nx.N, xmin=stats.xmin, xmax=stats.xmax, dx=stats.dx, x=gridnotes_1d(grid1d))
end

function compute_weights_and_index_1d(grid1d, xp::Vector{Float64})
    axes = grid1d_axes(grid1d)
    index_list, weight_list = [], []
    for xi in xp
        xp_normed = (xi - axes.xmin) / axes.dx
        idx, wtx = ParticleInCell.get_absolute_i_and_w(xp_normed)
        push!(index_list, idx)
        push!(weight_list, wtx)
    end
    return index_list, weight_list
end

function integrate_charge_history(grid1d, charges_1d, xp; N::Int, cg::Float64)
    axes = grid1d_axes(grid1d)
    q = copy(charges_1d)
    x = copy(xp)

    state = SharedMatrix{Float64}(axes.Nx, 1)
    charge_hist = Vector{Float64}(undef, N + 1)
    charge_hist[1] = Float64(sum(q))

    for ti in 1:N
        state .= 0
        index_positions, weights = compute_weights_and_index_1d(grid1d, x)
        ParticleInCell.push_to_grid!(state, q, index_positions, weights, axes.Nx, true)

        q = dropdims(Array{Float64}(state), dims=2)
        x = axes.x .+ cg .* q

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
        (name="box left, zero base", N=400, cg=-0.2, init=axes -> init_box_case(axes; base=0.0)),
        (name="sin left, nonlinear advection", N=200, cg=-0.3, init=init_sin_left_case),
    ]

    for case in cases
        xp, charges_1d = case.init(axes)
        charge_hist = integrate_charge_history(grid1d, charges_1d, xp; N=case.N, cg=case.cg)
        drift = max_relative_charge_drift(charge_hist)
        @test drift <= 0.02
    end
end
