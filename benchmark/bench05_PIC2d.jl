using Statistics
using Plots

using PiCLES.ParticleMesh: OneDGrid, OneDGridNotes
import PiCLES.ParticleInCell

import PiCLES

using PiCLES.Utils: Init_Standard
using PiCLES.Operators.core_2D: InitParticleInstance
using PiCLES.ParticleSystems: particle_waves_v5 as PW
using PiCLES.Architectures: AbstractParticleInstance

function check_sum(charges_1d)
    Float64(sum(charges_1d)) #/grid1d.nx
end

using Oceananigans.Units
using SharedArrays
using StaticArrays

using BenchmarkTools
#using Revise
using Profile
using ProfileView

using PiCLES.Operators.core_2D: GetVariablesAtVertex, Get_u_FromShared
using PiCLES.custom_structures: wni

# % define some functions 
#plot_path_base = "plots/tests/T01_PIC_1D/with_merge_rule/"
#mkdir(plot_path_base)


# %%

u(x::Number, y::Number, t::Number) = 10.0 
v(x::Number, y::Number, t::Number) = -10.0

ParticleState0, default_ODE_parameters, WindSeamin, Const_ID = Init_Standard(2.0, 2.0, 10minutes)

particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q)

ODE_settings = PW.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=log(WindSeamin["E"]),
    #maximum energy threshold
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=2minutes,
    timestep=10minutes,
    total_time=T = 6days,
    save_everystep=false,
    maxiters=1e4,
    adaptive=true,
    dt=10,#60*10, 
    dtmin=1,#60*5, 
    force_dtmin=true,)


PI = InitParticleInstance(particle_system, copy(ParticleState0), ODE_settings, (1, 1), false, true)

@time @allocated grid2d = PiCLES.ParticleMesh.TwoDGrid(100e3, 21, 100e3, 21)
grid2dnotes = PiCLES.ParticleMesh.TwoDGridNotes(grid2d)

#State = SharedArray{Float64,4}(grid2d.Nx, grid2d.Ny, 5, 1);
cg = [0.3, 0.2]
dt = 10minutes
S = SharedArray{Float64,3}(grid2d.Nx, grid2d.Ny, 3);
S = @MArray zeros(grid2d.Nx, grid2d.Ny, 3);

function propagate!(PI::AbstractParticleInstance, cg::Vector{Float64})
    PI.ODEIntegrator.u[4] = PI.position_xy[1] .+ cg[1] #.* i_charges  #.* (0.9 .+ 0.05 *(1 .+  rand(grid1d.Nx)))
    PI.ODEIntegrator.u[5] = PI.position_xy[2] .+ cg[2] #.* i_charges  #.* (0.9 .+ 0.05 *(1 .+  rand(grid1d.Nx)))
    nothing
end

### loop start



# %%
Revise.retry()
time = 0

function loop_particle(PI, S, G, cg, dt)
    # fake advance: propgate
    propagate!(PI, cg)
    time = PI.ODEIntegrator.t + dt
    #end of advance!
    PiCLES.Operators.mapping_2D.ParticleToNode!(PI, S, grid2d, false)

    # remesh!
    u_state = Get_u_FromShared(PI, S)
    ui = GetVariablesAtVertex(u_state, PI.position_xy[1], PI.position_xy[2])
    #@info exp(ui[1]), ui[2], ui[4]/1e3, ui[5]/1e3              
    PiCLES.Operators.mapping_2D.reset_PI_ut!(PI; ui=ui, ti=time)

end

loop_particle(PI, S, grid2d, cg, dt)

@benchmark begin 
    loop_particle(PI, S, grid2d, cg, dt)
    #@info S[1:5, 1:5, 1]
    # reset state
    if isa(S, SharedArray)
        S[:, :, :] .= 0.0
    else
        S .= 0.0
    end

    # S[:,:,:] .= 0.0   #  <-------- this is the fastest version
end
# org: Memory estimate: 5.42 KiB, allocs estimate: 91.  29.583 μs 
# 1st: Memory estimate: 2.47 KiB, allocs estimate: 41. 1.575 μs 
# 2nd: Memory estimate: 1.41 KiB, allocs estimate: 32.  1.200 μs  # MArray
# 3rd Memory estimate: 1.06 KiB, allocs estimate: 27. 1.033 μs  # Marray
# 4th:  Memory estimate: 752 bytes, allocs estimate: 24.  965.000 ns  #MArray with ODEIntegrator.u as MVector
# 5th:  Memory estimate: 1.23 KiB, allocs estimate: 32. 1.100 μs  #MArray with ODEIntegrator.u as normal vector


# 2nd: Memory estimate: 1.38 KiB, allocs estimate: 31. 1.142 μs  #SArray
# 4th:  Memory estimate: 976 bytes, allocs estimate: 25.   1.183 μs 

PI.ODEIntegrator.u

@benchmark S .= 0.0
# org:  Memory estimate: 3.03 KiB, allocs estimate: 52. 28250 ns or 28.250 μs  # SArray
# MArray  Memory estimate: 0 bytes, allocs estimate: 0.    204.375 ns

# better
@benchmark S[:,:,:] .= 0.0
# Memory estimate: 80 bytes, allocs estimate: 2. 312.685 ns # SArray 
# Memory estimate: 32 bytes, allocs estimate: 1.  278.906 ns  # MArray


# %% --- single step
@benchmark loop_particle(PI, S, grid2d, cg, dt)
# org: Memory estimate: 2.39 KiB, allocs estimate: 39.  1.196 μs 
# Memory estimate: 1.38 KiB, allocs estimate: 31. 858.560 ns 
# Memory estimate: 1.06 KiB, allocs estimate: 27. 786.555 ns 
# Memory estimate: 752 bytes, allocs estimate: 24. 760.642ns # MArray
# Memory estimate: 896 bytes, allocs estimate: 23. 794.643 ns @SArray
# %%

@profview_allocs for t in 1:20000
    loop_particle(PI, S, grid2d, cg, dt)
    #@info S[1:5, 1:5, 1]
    # reset state
    if isa(S, SharedArray)
        S[:, :, :] .= 0.0
    else
        S .= 0.0
    end
end

# %%
Profile.clear()
@profile for t in 1:20000
    loop_particle(PI, S, grid2d, cg, dt)
    #@info S[1:5, 1:5, 1]
    # reset state
    S[:, :, :] .= 0.0
end

ProfileView.view(expand_tasks=true, expand_threads=true)
Profile.print(mincount=20, groupby=:thread)


# %% 
PiCLES.Operators.mapping_2D.ParticleToNode!(PI, S, grid2d, false)
@benchmark PiCLES.Operators.mapping_2D.ParticleToNode!(PI, S, grid2d, false)
# org:  Memory estimate: 1.64 KiB, allocs estimate: 22.   1.579 μs
# 3rd: Memory estimate: 272 bytes, allocs estimate: 8.  149.981 ns  




# %% ---------- old
using StaticArrays

index_pos = (3, 5)
weights = (0.5, 0.5)
charge = [ 1.0, 0.1, 0.01 ]
@benchmark S[index_pos[1], index_pos[2], :] +=  weights[1] * weights[2] * charge
# org:  Memory estimate: 336 bytes, allocs estimate: 7. 3.241 μs 

@benchmark  ParticleInCell.push_to_grid!(S, charge, index_pos, weights, grid2d.Nx, grid2d.Ny, false)
# Memory estimate: 464 bytes, allocs estimate: 5.  386.062 ns 

function push_to_grid3!(S2, charge, index_pos, weights)
    S2[index_pos[1], index_pos[2], :] += weights[1] * weights[2] * charge
end

push_to_grid3!(S, charge, index_pos, weights)
@benchmark push_to_grid3!(S, charge, index_pos, weights)
#  Memory estimate: 240 bytes, allocs estimate: 3. 268.334 ns 

@time @allocated S[5, 6, :] += @. [ 2.0, 0.2, 0.02]
# 5 acclocations
#@time @allocated S[5, 6, :] = S[5, 6, :] .+  [2.0, 0.2, 0.02]

@time @allocated S[5, 6, 3] += 3.0
# 2 allocation

#S2 = MMatrix{grid2d.Nx, grid2d.Ny, 3}
#S2[:,:,:] .= 0.0

#MArray{Float64,3}(grid2d.Nx, grid2d.Ny, 3)
S2 = @MArray zeros(grid2d.Nx, grid2d.Ny,3) ;
@time @allocated S2[5, 6, 3] += 3.0
# 2 allocations
@time @allocated S2[5, 6, :] += @. [2.0, 0.2, 0.02]
# 3 allocations

@time @allocated S2[index_pos[1], index_pos[2], :] += weights[1] * weights[2] * charge
# 5 accolations

@time @allocated S2[index_pos[1], index_pos[2], :] += @. weights[1] .* weights[2] .* charge
# 7 allocations 

#{Float64,3}(grid2d.Nx, grid2d.Ny, 3);


# %%
#PI.position_ij
Revise.retry()

S2_type = typeof(S2)

S2 = @MArray zeros(grid2d.Nx, grid2d.Ny, 3);
index_pos = @SVector [3, 5]
weights = @SVector Float16[0.5, 0.2]
charge = @MVector [1.0, 0.1, 0.01]

@time @allocated SVector{2, Int64}(1, 9)

#@benchmark S2[index_pos[1], index_pos[2], :] += weights[1] * weights[2] * charge
# org Memory estimate: 240 bytes, allocs estimate: 7.  513.531 ns
# 1st: Memory estimate: 128 bytes, allocs estimate: 5. 310.914 ns  

ParticleInCell.push_to_grid!(S2, charge, index_pos, weights, grid2d.Nx, grid2d.Ny, false)
@benchmark ParticleInCell.push_to_grid!(S2, charge, index_pos, weights, grid2d.Nx, grid2d.Ny, false)
# org: Memory estimate: 256 bytes, allocs estimate: 3. 261.701 ns
# 1st:  Memory estimate: 256 bytes, allocs estimate: 3. 80.263 ns  < --- MArray
#       Memory estimate: 368 bytes, allocs estimate: 5.      < --------- SharedArray

@profview_allocs for t in 1:20000
    ParticleInCell.push_to_grid!(S2, charge, index_pos, weights, grid2d.Nx, grid2d.Ny, false)
end

grid2d.Nx * grid2d.Ny *3


function push_to_grid2!(S2, charge, index_pos, weights)
    S2[index_pos[1], index_pos[2], :] += weights[1] * weights[2] * charge
end
@benchmark push_to_grid2!(S2, charge, index_pos, weights)
# 2nd: Memory estimate: 32 bytes, allocs estimate: 1.  21.355 ns < --- best


# S2[index_pos[1], index_pos[2], :] += weights[1] * weights[2] * charge
# weights[1] * weights[2] * charge
# @time @allocated aa = weights * charge'
# @time @allocated aa[1, :] .* aa[2, :]


# %% compute_weights_and_index
using PiCLES.ParticleMesh

function get_i_and_w3(zp_normed::Float64)

    # floor points x
    ip_base = floor(zp_normed) # ceil position
    #ip_floor = Int(ip_base)
    # if sign(ip_base) == -1.0
    #     ip_floor = Int(ip_base)
    # else
    #     ip_floor = Int(ip_base+ 1)
    # end
    ip_floor = Int(ip_base + 1)

    dxp_ceil = zp_normed - ip_base   # floor weight
    dxp_floor = 1.0 - dxp_ceil #ip_base+1 - xp_normed # ceil weight

    # ii = SVector{2, Int64}( [ip_floor, ip_floor + 1])
    # ww = SVector{2, Float64}( [dxp_floor, dxp_ceil])
    
    # nothing
    # #return ii, ww
    return SVector{2, Int64}( [ip_floor, ip_floor + 1]) , SVector{2, Float64}( [dxp_floor, dxp_ceil]) 
end

function norm_distance(xp::T, xmin::T, dx::T) where {T <: AbstractFloat}
    return (xp - xmin) / dx
end


function compute_weights_and_index3(g_pars::TwoDGrid, xp::Float64, yp::Float64)
    """
    2d wrapper for 1d function
    """

    
    # xp_normed = (xp - g_pars.xmin) / g_pars.dx # multiples of grid spacing
    # yp_normed = (yp - g_pars.ymin) / g_pars.dy # multiples of grid spacing

    xi, xw = get_i_and_w3(norm_distance(xp, g_pars.xmin, g_pars.dx) )
    yi, yw = get_i_and_w3(norm_distance(yp, g_pars.ymin, g_pars.dy) )

    idx = [(xi[1], yi[1]), (xi[2], yi[1]), (xi[1], yi[2]), (xi[2], yi[2])]
    wtx = [(xw[1], yw[1]), (xw[2], yw[1]), (xw[1], yw[2]), (xw[2], yw[2])]

    return idx, wtx
end

struct wni7{TI <: SVector , TF <: SVector} <: FieldVector{4,SVector}
    xi::TI
    xw::TF
    yi::TI
    yw::TF
end

function compute_weights_and_index4(g_pars::TwoDGrid, xp::Float64, yp::Float64)
    """
    2d wrapper for 1d function
    """


    # xp_normed = (xp - g_pars.xmin) / g_pars.dx # multiples of grid spacing
    # yp_normed = (yp - g_pars.ymin) / g_pars.dy # multiples of grid spacing

    xp_normed = norm_distance(xp, g_pars.xmin, g_pars.dx)
    yp_normed = norm_distance(yp, g_pars.ymin, g_pars.dy)

    xi, xw = get_i_and_w3(xp_normed)
    yi, yw = get_i_and_w3(yp_normed)

    # idx = [(xi[1], yi[1]), (xi[2], yi[1]), (xi[1], yi[2]), (xi[2], yi[2])]
    # wtx = [(xw[1], yw[1]), (xw[2], yw[1]), (xw[1], yw[2]), (xw[2], yw[2])]

    return wni7(xi, xw, yi, yw)
    #return @SVector [xi, xw, yi, yw]
end


# standard version
@benchmark ParticleInCell.compute_weights_and_index(grid2d, PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5])
# org: Memory estimate: 640 bytes, allocs estimate: 9. 628.886 ns 


@benchmark compute_weights_and_index3(grid2d, PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5])
# Memory estimate: 608 bytes, allocs estimate: 8. 568.838 ns

u45S = @MVector [PI.ODEIntegrator.u[4] + 0.345, PI.ODEIntegrator.u[5]]

@benchmark compute_weights_and_index3(grid2d, u45S[1], u45S[2]) # < ------ this version is the fastest the particle state being a MVector
# 1st: Memory estimate: 608 bytes, allocs estimate: 8. 379.934ns 
# 2nd: Memory estimate: 608 bytes, allocs estimate: 8. 130.276 ns  

wni_i = compute_weights_and_index4(grid2d, u45S[1], u45S[2])
@benchmark compute_weights_and_index4(grid2d, u45S[1], u45S[2]) # < ------ this version is the fastest the particle state being a MVector
# Memory estimate: 352 bytes, allocs estimate: 6.  291.559 ns
# this is faster. rewrite ParticleInCell.push_to_grid! L 281 such that it can use this output
# Memory estimate: 352 bytes, allocs estimate: 6.  98.958 ns 
#  Memory estimate: 352 bytes, allocs estimate: 6. 274.276 ns

wni_i

@time @allocated ParticleInCell.construct_loop(wni_i)

wni = wni_i 
@time @allocated 

function construct_loop(wni::FieldVector{4,SVector})
    idx = SVector{4,Tuple{Int, Int}}( (wni.xi[1], wni.yi[1]), (wni.xi[2], wni.yi[1]), (wni.xi[1], wni.yi[2]), (wni.xi[2], wni.yi[2]) )
    wtx = SVector{4,Tuple{AbstractFloat, AbstractFloat } }( (wni.xw[1], wni.yw[1]), (wni.xw[2], wni.yw[1]), (wni.xw[1], wni.yw[2]), (wni.xw[2], wni.yw[2]))
    return zip(idx, wtx)
end

@time @allocated construct_loop(wni_i)

@time @allocated xx = SVector{4,Tuple}((wni.xi[1], wni.yi[1]), (wni.xi[2], wni.yi[1]), (wni.xi[1], wni.yi[2]), (wni.xi[2], wni.yi[2]))

xx[4]

xi2 = @SVector 
xw



wni_i = wni7(xi, xw, yi, yw)

xi

wni_i.xi[1]
function construct_loop(wni::FieldVector{4,SVector})
    idx = [(wni.xi[1], wni.yi[1]), (wni.xi[2], wni.yi[1]), (wni.xi[1], wni.yi[2]), (wni.xi[2], wni.yi[2])]
    wtx = [(wni.xw[1], wni.yw[1]), (wni.xw[2], wni.yw[1]), (wni.xw[1], wni.yw[2]), (wni.xw[2], wni.yw[2])]
    return zip(idx, wtx)
end

@benchmark construct_loop(wni_i)

typeof(xx)

compute_weights_and_index4(grid2d, u45S[1], u45S[2])

@benchmark @SVector [xi, xw]
@benchmark @SVector [xi, xw, yi, yw]
@benchmark [xi, xw, yi, yw]
@benchmark wni7(xi, xw, yi, yw)


# %%
# using PiCLES.custom_structures: ParticleInstance2D

# mutable struct ParticleInstance2D_new <: AbstractParticleInstance
#     position_ij::SVector{2,Int64}
#     position_xy::SVector{2,AbstractFloat}
#     ODEIntegrator::ODEIntegrator
#     boundary::Bool
#     on::Bool
# end


# @benchmark PIold = ParticleInstance2D(PI.position_ij, PI.position_xy, PI.ODEIntegrator, PI.boundary, PI.on)
# # org:  Memory estimate: 128 bytes, allocs estimate: 3. 250.903 ns  
# #  Memory estimate: 128 bytes, allocs estimate: 3.

# @benchmark PIn = ParticleInstance2D_new(PI.position_ij, PI.position_xy, PI.ODEIntegrator, PI.boundary, PI.on)
# # org: Memory estimate: 144 bytes, allocs estimate: 4. 255.952 ns 



typeof(S) == SharedArray{Float64,3}







