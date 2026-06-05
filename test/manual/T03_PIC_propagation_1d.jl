"""
Manual diagnostic: 1D particle-in-cell test scenario.

Interactive script for visual checks of PIC mapping behavior in 1D.
"""

using Statistics
using Plots

using PiCLES.Grids.CartesianGrid: CartesianGridMesh1D, gridnotes_1d
import PiCLES.ParticleInCell

import PiCLES

function check_sum(charges_1d)
    Float64(sum(charges_1d)) #/grid1d.nx
end

using SharedArrays

# %% define some functions 
plot_path_base = "plots/tests/T01_PIC_1D/with_merge_rule/"
#mkdir(plot_path_base)

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

function PIC_loop(grid1d, charges_1d, xp; N= 100, cg= 0.2, verbose=false)
    axes = grid1d_axes(grid1d)

    x_collect = []
    p_collect = []
    push!(x_collect, xp)
    push!(p_collect, charges_1d)

    #xp         = grid1d.x .+ cg0 #.* (1 .+ rand())
    i_charges = charges_1d

    State = SharedMatrix{Float64}(axes.Nx, 1)
    State .= 0
    for ti in 1:1:N

        index_positions, weights = compute_weights_and_index_1d(grid1d, xp)
        #index_positions          = wrap_indexpositons(index_positions, grid1d)

        ParticleInCell.push_to_grid!(State, i_charges, index_positions, weights, axes.Nx, true)

        if verbose
            @show ip_charges_sum - Float64(sum(State))
        end

        # update charges
        i_charges = dropdims(Array{Float64}(State), dims=2)#Array{Float64}(State) #State

        # propagate
        xp = axes.x .+ cg .* i_charges  #.* (0.9 .+ 0.05 *(1 .+  rand(axes.Nx)))
        #xp                       = grid1d.x .+ cg0 .* (0.9 .+ 0.05 *(1 .+  rand(grid1d.Nx)))
        #xp                       = grid1d.x .+  0.1 * (.-0.5 .+  rand(grid1d.Nx))
        #xp                       = xp .+  0.1 * (.-0.2 .+  rand(grid1d.Nx))

        push!(x_collect, xp)
        push!(p_collect, copy(State))
        State .= 0
    end
    return x_collect, p_collect
end


function animate(x_collect, p_collect, grid1d; path, name)

    anim = @animate for i in 1:1:length(p_collect)

        scatter(x_collect[i], p_collect[i], label="new")
        if i > 1
            scatter!(x_collect[i-1], p_collect[i-1], color="gray", alpha=0.4, label="old")
        end
        title!("sum = $( round(sum(p_collect[i]) )) ")

        ylims!(-0.5, 1.9)
        xlims!(grid1d.stats.xmin, grid1d.stats.xmax,)
    end
    gif(anim, joinpath(path, name * ".gif"), fps=10)
    #round(sum(p_collect[1]) )
end

function convert_store_to_tuple(p_collect, grid1d)
    store_data = cat(p_collect..., dims=2)
    store_data = permutedims(store_data, (2, 1))
    x = gridnotes_1d(grid1d)
    steps = collect(range(1, size(store_data)[1], step=1))
    return (data=store_data, x=x, steps=steps)
end

# %%
n_particles = 101
eta_min, eta_max =0, 20
grid1d = CartesianGridMesh1D(eta_min, eta_max, n_particles; Ny=3, periodic_boundary=true)
grid1dnotes = gridnotes_1d(grid1d)
grid1daxes = grid1d_axes(grid1d)

#grid1d = OneGridPars(grid)
#xp = rand(n_particles) * eta_max

# make State vector:
State = SharedMatrix{Float64}(grid1daxes.Nx, 1)
#State = grid1dnotes.x * 0

# initial charge position
xp = grid1dnotes .+ grid1daxes.dx *1.5 #+ rand(n_particles)/5


charges_1d = xp * 0 .- 0.2
charges_1d[40:Int(ceil(grid1daxes.Nx*2/3))] .=1

ip_charges_sum  = check_sum(charges_1d)
@info "charge sum $(check_sum(charges_1d))"

# manual step
index_positions, weights = compute_weights_and_index_1d(grid1d, xp)

#index_positions          = wrap_indexpositons(index_positions, grid1d) # is part of push_charged_to_gridpoints!
ParticleInCell.push_to_grid!(State, charges_1d, index_positions, weights, grid1daxes.Nx)
@info "charge sum $(check_sum(State))"


plot()
scatter(xp, charges_1d, label ="advected", alpha= 1, s =3 )
scatter!(grid1dnotes, State, color="black", alpha=0.5, s=1, label="new")


# %% Box moving to the right, zero base
xp = grid1dnotes .+ grid1daxes.dx * 1.5 #+ rand(n_particles)/5
charges_1d = xp * 0 
charges_1d[40:Int(ceil(grid1daxes.Nx * 2 / 3))] .= 1

case_title = "Box right, zero base"
case_name = "box_r0"

x_collect, p_collect = PIC_loop(grid1d, charges_1d, xp; N=50, cg=0.2, verbose=true);

data = convert_store_to_tuple(p_collect, grid1d)
plot()
heatmap(data.x, data.steps, data.data)
title!(case_title)
#save figure
savefig(joinpath(plot_path_base, case_name * ".png"))

animate(x_collect, p_collect, grid1d, path=plot_path_base, name=case_name)



# %% Box moving to the left, non-zero base, i.e. convergence and divergence
xp = grid1dnotes .+ grid1daxes.dx * 1.5 #+ rand(n_particles)/5
charges_1d = xp * 0 .- 0.5
charges_1d[40:Int(ceil(grid1daxes.Nx * 2 / 3))] .= 1

case_title = "Box right, non-zero base"
case_name = "box_rb"

x_collect, p_collect = PIC_loop(grid1d, charges_1d, xp; N=100, cg = 0.2, verbose=true);

data = convert_store_to_tuple(p_collect, grid1d)
plot()
heatmap(data.x, data.steps, data.data)
title!(case_title)
#save figure
savefig(joinpath(plot_path_base, case_name * ".png"))

animate(x_collect, p_collect, grid1d, path=plot_path_base, name=case_name)


# %% box to the left, zero base
xp = grid1dnotes .+ grid1daxes.dx * 1.5 #+ rand(n_particles)/5
charges_1d = xp * 0 
charges_1d[40:Int(ceil(grid1daxes.Nx * 2 / 3))] .= 1

case_title = "Box left, zero base"
case_name = "box_l0"

x_collect, p_collect = PIC_loop(grid1d, charges_1d, xp; cg=-0.2, N=100, verbose=true);

data = convert_store_to_tuple(p_collect, grid1d)
plot()
heatmap(data.x, data.steps, data.data)
title!(case_title)
savefig(joinpath(plot_path_base, case_name * ".png"))

animate(x_collect, p_collect, grid1d, path=plot_path_base, name=case_name)



# %% box to the left, non-zero base, i.e. convergence and divergence
xp = grid1dnotes .+ grid1daxes.dx * 1.5 #+ rand(n_particles)/5
charges_1d = xp * 0 .- 0.5
charges_1d[40:Int(ceil(grid1daxes.Nx * 2 / 3))] .= 1

case_title = "Box left, non-zero base"
case_name = "box_lb"

x_collect, p_collect = PIC_loop(grid1d, charges_1d, xp; cg = -0.2, N=100, verbose=true);

data = convert_store_to_tuple(p_collect, grid1d)
plot()
heatmap(data.x, data.steps, data.data)
title!(case_title)
savefig(joinpath(plot_path_base, case_name * ".png"))

animate(x_collect, p_collect, grid1d, path=plot_path_base, name=case_name)



# %% sin wave moving to the left, Speed dependent on charge, i.e. non-linear advection

xp = grid1dnotes .+ grid1daxes.dx * 0.5 #  #+ rand(n_particles)/5

charges_1d = sin.(3 * 2 * pi * xp ./ grid1daxes.xmax) * 0.2 .+ 0.2

case_title = "Sin left, nonlinear advection"
case_name = "sin_l"

x_collect, p_collect = PIC_loop(grid1d, charges_1d, xp; cg=-0.3, N=50, verbose=true);
animate(x_collect, p_collect, grid1d, path=plot_path_base, name=case_name)

data = convert_store_to_tuple(p_collect, grid1d)
plot()
heatmap(data.x, data.steps, data.data)
title!(case_title)
savefig(joinpath(plot_path_base, case_name * ".png"))
animate(x_collect, p_collect, grid1d, path=plot_path_base, name=case_name)



# %% sin wave moving to the left, Speed dependent on charge, i.e. non-linear advection

xp = grid1dnotes .+ grid1daxes.dx * 0.5 #  #+ rand(n_particles)/5

charges_1d = sin.(3* 2 *pi * xp ./ grid1daxes.xmax) * 0.2 .+ 0.2

case_title = "Sin left, nonlinear advection"
case_name = "sin_l"

x_collect, p_collect = PIC_loop(grid1d, charges_1d, xp; cg =-0.3, N= 50, verbose=true);
animate(x_collect, p_collect, grid1d, path=plot_path_base, name=case_name)

data = convert_store_to_tuple(p_collect, grid1d)
plot()
heatmap(data.x, data.steps, data.data)
title!(case_title)
savefig(joinpath(plot_path_base, case_name * ".png"))
animate(x_collect, p_collect, grid1d, path=plot_path_base, name=case_name)



# %% diverging sin, moving to left and right until cancelation
xp = grid1dnotes .+ grid1daxes.dx * 1.5 #+ rand(n_particles)/5
#charges_1d = rand(n_particles)* 0 .+ 1
charges_1d = sin.(3* 2 *pi * xp ./ grid1daxes.xmax) *0.4 .+0.0

case_title = "Diverging sin, canceling"
case_name = "sin_div"

x_collect, p_collect = PIC_loop(grid1d, charges_1d, xp; cg =-0.3, N= 30, verbose=true);
animate(x_collect, p_collect, grid1d, path=plot_path_base, name=case_name)

data = convert_store_to_tuple(p_collect, grid1d)
plot()
heatmap(data.x, data.steps, data.data)
title!(case_title)
savefig(joinpath(plot_path_base, case_name * ".png"))

animate(x_collect, p_collect, grid1d, path=plot_path_base, name=case_name)