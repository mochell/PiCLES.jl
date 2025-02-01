module ParticleInCell

using ..Architectures: StateTypeL1, AbstractBoundary, StateTypeL1_1D
using Statistics
using LinearAlgebra

using SharedArrays
using StaticArrays

using ..custom_structures: wni, N_Periodic, N_NonPeriodic, N_TripolarNorth

# %%
using ..ParticleMesh
# Tolerance for comparison of real numbers: set it here!

# Set parameters
# eta_min = 0.0
# eta_max = 20.0
#
# num_cells = 22
# delta_t = 0.1
#
# grid2d = TwoDGrid(eta_max,num_cells, eta_max ,num_cells)
# gridnotes = TwoDGridNotes(grid2d)

export push_to_grid!
# %%

"""
get_absolute_i_and_w(zp_normed::Float64)
returns the floor and ceil index and weights
Indexes is the absolute floor and ceil index
"""
function get_absolute_i_and_w(zp_normed::Float64)

    # floor points x
    ip_base = floor(zp_normed) # ceil position
    #ip_floor = Int(ip_base)
    # if sign(ip_base) == -1.0
    #     ip_floor = Int(ip_base)
    # else
    #     ip_floor = Int(ip_base+ 1)
    # end
    ip_floor = Int(ip_base+ 1)

    dxp_ceil = zp_normed - ip_base   # floor weight
    dxp_floor = 1.0 - dxp_ceil #ip_base+1 - xp_normed # ceil weight
    return SVector{2, Int64}(ip_floor , ip_floor+1) , SVector{2, Float64}(dxp_floor, dxp_ceil)
end

"""
get_absolute_i_and_w(zp_normed::Float64, i_node::Int64)
returns the floor and ceil index and weights
inputs:
    zp_normed: Float64 normalized position relative to particle node position 
    i_node: Int64 particle node position
Indexes is the absolute floor and ceil index
"""
function get_absolute_i_and_w(zp_normed::Float64, i_node::Int64)

    # floor points x
    ip_base = floor(zp_normed)

    # this is relative the particle node
    ip_floor = Int(ip_base)

    dxp_ceil = round(zp_normed - ip_base, digits=6)   # floor weight
    dxp_floor = 1.0 - dxp_ceil #ip_base+1 - xp_normed # ceil weight

    # add particle node position here
    return SVector{2,Int64}(ip_floor + i_node, ip_floor + i_node + 1), SVector{2,Float64}(dxp_floor, dxp_ceil)
end

"""
get_relative_i_and_w(zp_normed::Float64)
returns the floor and ceil index and weights
Indexes is the absolute floor and ceil index
"""
function get_relative_i_and_w(zp_normed::Float64)
    # floor points x
    ip_base = floor(zp_normed)

    # this is relative the particle node
    ip_floor = Int(ip_base)

    dxp_ceil = round(zp_normed - ip_base, digits=6)   # floor weight
    dxp_floor = 1.0 - dxp_ceil #ip_base+1 - xp_normed # ceil weight
    return SVector{2,Int64}(ip_floor, ip_floor + 1), SVector{2,Float64}(dxp_floor, dxp_ceil)
end



"""
norm_distance(xp::T, xmin::T, dx::T) 
returns normalized distance
"""
function norm_distance(xp::T, xmin::T, dx::T) where {T<:AbstractFloat}
    return (xp - xmin) / dx
end


"""
compute_weights_and_index(g_pars::TwoDGrid, xp::Float64, yp:: Float64 )
returns indexes and weights for in 2D for single x,y point
"""
function compute_weights_and_index(g_pars::TwoDGrid, xp::Float64, yp:: Float64 )
    """
    2d wrapper for 1d function
    """

    xp_normed = norm_distance(xp, g_pars.xmin, g_pars.dx)
    yp_normed = norm_distance(yp, g_pars.ymin, g_pars.dy)

    xi, xw = get_absolute_i_and_w(xp_normed)
    yi, yw = get_absolute_i_and_w(yp_normed)

    idx  = [ (xi[1], yi[1]) , (xi[2], yi[1]) , (xi[1], yi[2]), (xi[2], yi[2]) ]
    wtx  = [ (xw[1], yw[1]) , (xw[2], yw[1]) , (xw[1], yw[2]), (xw[2], yw[2]) ]

    return idx, wtx
end


"""
compute_weights_and_index_mininal(g_pars::TwoDGrid, xp::Float64, yp:: Float64 )
returns indexes and weights as FieldVector for in 2D for single x,y point
"""
function compute_weights_and_index_mininal(g_pars::TwoDGrid, xp::Float64, yp::Float64)
    """
    2d wrapper for 1d function
    """

    xp_normed = norm_distance(xp, g_pars.xmin, g_pars.dx)  # multiples of grid spacing
    yp_normed = norm_distance(yp, g_pars.ymin, g_pars.dy)  # multiples of grid spacing

    xi, xw = get_absolute_i_and_w(xp_normed)
    yi, yw = get_absolute_i_and_w(yp_normed)

    return wni(xi, xw, yi, yw)
end

"""
compute_weights_and_index_mininal(xp::Float64, yp:: Float64 )
returns indexes and weights as FieldVector for in 2D for single x,y point
    returned indexes are in absolute coordindates to the particle node, calculated from :
    inputs:
    ij : Tuple{Int,Int} absolute position of the particle node
    xp,yp : Float64 particle position relative to the particle node (normalized units)
"""
function compute_weights_and_index_mininal(ij::II, xp::Float64, yp::Float64) where {II<:Union{Tuple{Int,Int},CartesianIndex}}
    """
    2d wrapper for 1d function
    """
    xi, xw = get_absolute_i_and_w(xp, ij[1])
    yi, yw = get_absolute_i_and_w(yp, ij[2])

    return wni(xi, xw, yi, yw)
end

"""
compute_weights_and_index(g_pars::OneDGrid, xp::Float64 )
returns indexes and weights for in 2D for single x point
"""
function compute_weights_and_index(g_pars::OneDGrid, xp::Float64 )

    xp_normed = (xp - g_pars.xmin) / g_pars.dx # multiples of grid spacing
    xi, xw = get_absolute_i_and_w(xp_normed)

    idx  = [ (xi[1]) , (xi[2]) ]
    wtx  = [ (xw[1]) , (xw[2]) ]

    return idx, wtx
end

#indexes, weights = compute_weights_and_index(grid2d, 10.0, 9.9 )

"""
compute_weights_and_index(g_pars::TwoDGrid, xp::Float64, yp:: Float64 )
returns indexes and weights for in 2D for vectors
"""
function compute_weights_and_index(grid::TwoDGrid, xp::Vector{Float64}, yp:: Vector{Float64} )
    """
    returns:
    weight list        2 N X 1 vector of tuples with indices and weights
    """
    #return reduce(vcat,[compute_weights_and_index_1d(g_pars, xi) for xi in xp])
    index_list, weight_list  = [], []
    for (xi,yi) in zip(xp, yp)
        dd, ff = compute_weights_and_index(grid, xi, yi)
        push!(index_list, dd)
        push!(weight_list,ff)
    end

    #return reduce(vcat, index_list ), reduce(vcat, weight_list)
    return index_list, weight_list
end




"""
compute_weights_and_index(g_pars::TwoDGrid, xp::Float64 )
returns indexes and weights for in 1D for vectors
"""
function compute_weights_and_index(grid::OneDGrid, xp::Vector{Float64})
    """
    returns:
    weight list        2 N X 1 vector of tuples with indices and weights
    """
    #return reduce(vcat,[compute_weights_and_index_1d(g_pars, xi) for xi in xp])
    index_list, weight_list  = [], []
    for xi in xp
        dd, ff = compute_weights_and_index(grid, xi)
        push!(index_list, dd)
        push!(weight_list,ff)
    end

    #return reduce(vcat, index_list ), reduce(vcat, weight_list)
    return index_list, weight_list
end


#index_positions, weights = compute_weights_and_index(grid2d, [0.1, 11.1, 0.1, 5.234, -4.3], [0.1, 0.5, 2.9, 9.99, -3.2])


#### merging rules: ####

# V1: angle and wave age
function merge!(grid_point::Vector{Float64}, charge::Vector{Float64}; verbose=false)

    verbose ? (@info "grid_point = $grid_point, charge = $charge") : nothing

    ΔE = grid_point[1] - charge[1]
    # ΔE > 0 means that the grid point has more energy than the charge
    if norm(grid_point[2:3]) == 0
        # state vector is empty
        cosθ = 1
    else
        # calculate angle between both
        cosθ = grid_point[2] * charge[2] + grid_point[3] * grid_point[3] / (norm(grid_point[2:3]) * norm(charge[2:3]))
    end

    if (cosθ >= 0.5)
        verbose ? (@info "within 60 degrees, or grid point is 0: add") : nothing
        grid_point += charge
    elseif (cosθ < 0.5) & (ΔE > 0)
        verbose ? (@info "θ > 60 deg and E_grid > E_charge: forget charge") : nothing
    elseif (cosθ > 0.5) & (ΔE <= 0)
        verbose ? (@info "θ > 60 deg and E_grid < E_charge: replace") : nothing
        grid_point = charge
    end
    return grid_point

end

# V0 : Who is larger
# function merge!(grid_point::Vector{Float64}, charge::Vector{Float64}; verbose=false)

#     verbose ? (@info "grid_point = $grid_point, charge = $charge") : nothing

#     m_grid = grid_point[2]
#     m_charge = charge[2]
#     if (sign(m_grid) == sign(m_charge)) | (m_grid == 0)
#         verbose ? (@info "add, same sign, or grid is zero") : nothing
#         grid_point += charge
#     elseif (sign(m_grid) != sign(m_charge)) & (sign(m_grid) * m_grid > sign(m_charge) * m_charge)
#         verbose ? (@info "forget charge, State is larger") : nothing

#     elseif (sign(m_grid) != sign(m_charge)) & (sign(m_grid) * m_grid <= sign(m_charge) * m_charge)
#         verbose ? (@info "overwrite, charge is >=, but sign is different") : nothing
#         grid_point = charge
#     end
#     return grid_point
# end

function merge!_1d(grid_point::Vector{Float64}, charge::Vector{Float64}; verbose=false)

    verbose ? (@info "grid_point = $grid_point, charge = $charge") : nothing

    m_grid = grid_point[2]
    m_charge = charge[2]
    if (sign(m_grid) == sign(m_charge)) | (m_grid == 0)
        verbose ? (@info "add, same sign, or grid is zero") : nothing
        grid_point += charge
    elseif (sign(m_grid) != sign(m_charge)) & (sign(m_grid) * m_grid > sign(m_charge) * m_charge)
        verbose ? (@info "forget charge, State is larger") : nothing

    elseif (sign(m_grid) != sign(m_charge)) & (sign(m_grid) * m_grid <= sign(m_charge) * m_charge)
        verbose ? (@info "overwrite, charge is >=, but sign is different") : nothing
        grid_point = charge
    end
    return grid_point

end

# V0: 1D version --- depreciated 
function merge!(grid_point::Float64, charge::Float64; verbose=false)

    verbose ? (@info "grid_point = $grid_point, charge = $charge") : nothing

    m_grid = grid_point
    m_charge = charge
    if (sign(m_grid) == sign(m_charge)) | (m_grid == 0)
        verbose ? (@info "add, same sign, or grid is zero") : nothing
        m_grid += m_charge
    elseif (sign(m_grid) != sign(m_charge)) & (sign(m_grid) * m_grid > sign(m_charge) * m_charge)
        verbose ? (@info "forget charge, State is larger") : nothing

    elseif (sign(m_grid) != sign(m_charge)) & (sign(m_grid) * m_grid <= sign(m_charge) * m_charge)
        verbose ? (@info "overwrite, charge is >=") : nothing
        m_grid = m_charge
    end
    return m_grid
end



#define merging operator
(⊓)(g::Vector{Float64}, c::Vector{Float64}) = merge!(g, c, verbose=false)
(⊓)(g::Float64, c::Float64) = merge!(g, c, verbose=true)


# ---------------------- Push to Grid 2D ------------------


## very general verion 2D
function push_to_grid!(grid::Matrix{Float64},
                            charge::Float64,
                            index_pos::Tuple{Int, Int},
                            weights::Tuple{Float64, Float64},
                            Nx::Int,  Ny::Int )

        grid[ wrap_index!(index_pos[1], Nx) , wrap_index!(index_pos[2], Ny) ] += weights[1] * weights[2] * charge
        #grid[ index_pos[1] , index_pos[2] ] += weights[1] * weights[2] * charge

end

## most recent version 2D - not with Boundary types 
function push_to_grid!(grid::StateTypeL1,
                            charge::CC, 
                            index_pos::II,
                            weights::WW,
                            Nx::Int, Ny::Int,
                            periodic::Bool=true) where {CC<:Union{Vector{Float64},SVector{3,Float64},MVector{3,AbstractFloat}},
                                                            II<:Union{Tuple{Int,Int},SVector{2,Int64}},
                                                            WW<:Union{Tuple{Float64,Float64},SVector{2,Float16}}}
    if periodic
        grid[ wrap_index!(index_pos[1], Nx) , wrap_index!(index_pos[2], Ny), : ] += weights[1] * weights[2] * charge
    else
        if sum( test_domain(index_pos, Nx, Ny) ) != 2
            # position outside of domain
            return
        else
            # position is inside the domain
            grid[ index_pos[1] , index_pos[2], : ] += weights[1] * weights[2] * charge
        end
    end

end

# Abstract Boundary Version
function push_to_grid!(grid::StateTypeL1,
                            charge::CC,
                            index_pos::II,
                            weights::WW,
                            Nx::AbstractBoundary, 
                            Ny::AbstractBoundary) where {CC<:Union{Vector{Float64},SVector{3,Float64},MVector{3,AbstractFloat}},
                                                            II<:Union{Tuple{Int,Int},SVector{2,Int64}},
                                                            WW<:Union{Tuple{Float64,Float64},SVector{2,Float16}}}

    # conditions where nothing should be returned
    if  (Nx isa N_NonPeriodic) & ~test_domain(index_pos[1], Nx.N) | # non-periodic in x and y-position is out of domain
        (Ny isa N_NonPeriodic) & ~test_domain(index_pos[2], Ny.N) | # non-periodic in y and y-position is out of domain
        (Ny isa N_TripolarNorth) & (index_pos[2] < 1) # tripolar north and y-position is below south pole
        # @info index_pos, " particle is not in domain, or at TripolarGrid SouthPole"
        return

    elseif (Ny isa N_TripolarNorth) & (index_pos[2] > Ny.N)  # Tripolar North boundary

        # @info index_pos, " particle is in domain, TripolarGrid make boundary condition"
        try
            index_pos, charge = TripolarNorthBoundary(index_pos, charge, Nx, Ny)
        catch e
            @error e, index_pos, charge, Nx, Ny
            return
        end
        grid[index_pos[1], index_pos[2], :] += weights[1] * weights[2] * charge

    else # all other boundaries

        # @info index_pos, " particle is in domain, wrap if needed"
        #@info wrap_index!(PI.position_ij[1], G.stats.Nx), wrap_index!(PI.position_ij[2], G.stats.Ny)
        grid[wrap_index!(index_pos[1], Nx), wrap_index!(index_pos[2], Ny), :] += weights[1] * weights[2] * charge

    end
    nothing
end


# old version
# ## MVector and SharedVector version
# function push_to_grid!(grid::AA,
#     charge::MVector{3,TzF},
#     index_pos::SVector{2, Int64},
#     weights::SVector{2,Float16},
#     Nx::Tx, Ny::Ty,
#     periodic::Bool=true) where {AA<:StateTypeL1,Tx<:Int,Ty<:Int,TzF<:AbstractFloat}
    
#     if periodic
    
#         grid[wrap_index!(index_pos[1], Nx), wrap_index!(index_pos[2], Ny), :] += weights[1] * weights[2] * charge
#         #set_to_grid!(grid, charge, [wrap_index!(index_pos[1], Nx), wrap_index!(index_pos[2], Ny)], weights)

#     else

#         if sum(test_domain(index_pos, Nx, Ny)) != 2
#             return
#             #position outside of domain
#         else
#             # position is inside the domain
#             grid[index_pos[1], index_pos[2], :] += weights[1] * weights[2] * charge
#             #set_to_grid!(grid, charge, index_pos, weights)
#         end
#     end

# end

# -------- grid specifica ------------

function TripolarNorthBoundary(index_pos::Tuple{Int,Int}, charge, Nx::N_Periodic, Ny::N_TripolarNorth,)

    # flip x position
    if index_pos[1] < 0
        xi_new  =  Nx.N + index_pos[1] % Int(Nx.N)
        xi_new = Nx.N - xi_new
    else
        xi_new = Nx.N - index_pos[1] % Int(Nx.N)
    end

    if Ny.N >= index_pos[2]
        error("Particle does not exceed the north boundary, this function should not be called.")
    else
        yi_new = 2 * Ny.N - index_pos[2] + 1
    end

    # flip y velocity
    charge_new = SVector{3,Float64}(charge[1], charge[2], charge[3])
    return (xi_new, yi_new), charge_new
end

# -------- helpers ------------

function wrap_index!(pos::Int, N::Int)
    pos = pos % Int(N)

    if pos < 0
        pos = pos + Int(N)
        #pos =  N
    elseif pos == 0
        pos = pos + Int(N)
    end
    return pos
end

function wrap_index!(pos::Int, N::Union{N_Periodic,N_NonPeriodic,N_TripolarNorth})
    pos = pos % Int(N.N)

    if pos < 0
        pos = pos + Int(N.N)
        #pos =  N
    elseif pos == 0
        pos = pos + Int(N.N)
    end
    return pos
end

"""
test_domain(index_pos, Nx, Ny)
test if the index position is within the domain
"""
function test_domain(index_pos, Nx, Ny)
    return (index_pos[1] > 0 && index_pos[1] <= Nx), (index_pos[2] > 0 && index_pos[2] <= Ny)
end

function test_domain(index_pos::Int, Nx::Int)
    return (index_pos > 0 && index_pos <= Nx)
end



# not used -- depricate
function set_to_grid!(S, charge, index_pos, weights)
    S[index_pos[1], index_pos[2], :] += weights[1] * weights[2] * charge
end

#push_to_grid!(charges_grid,1.0 , index_positions[1][1], weights[1][1], grid2d.Nx , grid2d.Ny )

# ----------------------- wrappers -----------------------
# wrapping over vectors of charges, index positions and weights
function push_to_grid!(grid::StateTypeL1,
                            charge::Vector{Float64},
                            index_pos::Vector{Tuple{Int, Int}},
                            weights::Vector{Tuple{Float64, Float64}},
                            Nx::Int, Ny::Int,
                            periodic::Bool=true)
    #@info "this is version D"
    for (i, w) in zip(index_pos, weights)
        push_to_grid!(grid, charge , i, w , Nx, Ny, periodic)
    end
end

#push_to_grid!(charges_grid, 1.0 , index_positions[3], weights[3] , grid2d.Nx , grid2d.Ny )
## allocation optimized:

"""
function construct_loop(wni::FieldVector{4,SVector})
    constructs loop over index positions and weights
"""
# function construct_loop(wni::FieldVector{4,SVector})
#     idx = [(wni.xi[1], wni.yi[1]), (wni.xi[2], wni.yi[1]), (wni.xi[1], wni.yi[2]), (wni.xi[2], wni.yi[2])]
#     wtx = [(wni.xw[1], wni.yw[1]), (wni.xw[2], wni.yw[1]), (wni.xw[1], wni.yw[2]), (wni.xw[2], wni.yw[2])]
#     return zip(idx, wtx)
# end

function construct_loop(wni::FieldVector{4,SVector})
    idx = SVector{4,Tuple{Int,Int}}(                    (wni.xi[1], wni.yi[1]), (wni.xi[2], wni.yi[1]), (wni.xi[1], wni.yi[2]), (wni.xi[2], wni.yi[2]))
    wtx = SVector{4,Tuple{AbstractFloat,AbstractFloat}}((wni.xw[1], wni.yw[1]), (wni.xw[2], wni.yw[1]), (wni.xw[1], wni.yw[2]), (wni.xw[2], wni.yw[2]))
    return zip(idx, wtx)
end


"""
wrapper over FieldVector weight&index (wni), 
"""
function push_to_grid!(grid::StateTypeL1,
    charge::CC,
    wni::FieldVector,
    Nx::Int, Ny::Int,
    periodic::Bool=true) where CC <: Union{Vector{Float64}, SVector{3, Float64}}
    #@info "this is version D"
    for (i, w) in construct_loop(wni)
        push_to_grid!(grid, charge, i, w, Nx, Ny, periodic)
    end
end


"""
wrapper over FieldVector weight&index (wni), 
# version wihtout boundary flag for BoundaryType
"""
function push_to_grid!(grid::StateTypeL1,
    charge::CC,
    wni::FieldVector,
    Nx::AbstractBoundary, Ny::AbstractBoundary) where {CC<:Union{Vector{Float64},SVector{3,Float64}}}
    #@info "this is version D"
    for (i, w) in construct_loop(wni)
        push_to_grid!(grid, charge, i, w, Nx, Ny)
    end
end



###### 1D versions ####

# wrapper over 1D Vecors of chanegs and (nested) index positions and weights
function push_to_grid!(grid::SharedMatrix{Float64},
    charge::CC,
    index_pos::Vector{Any},
    weights::Vector{Any},
    Nx::Int,
    periodic::Bool=true) where CC <: Union{Vector{Float64}, SVector{Float64}}
    #@info "this is version C"
    for (im, wm, c) in zip(index_pos, weights, charge)
        for (i, w) in zip(im, wm)
            push_to_grid!(grid, c, i, w, Nx, periodic)
        end
    end
end



# multiple index positions and 1 charge
function push_to_grid!(grid::StateTypeL1_1D,
                            charge::CC,
                            index_pos::Vector{Int},
                            weights::Vector{Float64},
                            Nx::Int,
                            periodic::Bool=true) where {CC<:Union{Vector{Float64},SVector{Float64}}}
    #@info "this is version B"
    if periodic
        for (im, wm) in zip(index_pos, weights)
            #grid[ wrap_index!(im, Nx), :] ⊓ wm * charge
            # old version
            #grid[wrap_index!(im, Nx), :] += wm * charge
            # merge rule version
            grid[wrap_index!(im, Nx), :] = merge!_1d(grid[wrap_index!(im, Nx), :], wm * charge)
        end
    else
        for (im, wm) in zip(index_pos, weights)
            if (im <= Nx) & (im > 0)
                # old version
                #grid[im, : ] += wm * charge

                # merge rule version
                grid[im, :] = merge!_1d(grid[im, :], wm * charge)
                #grid[im, :] = grid[im, :] ⊓ wm * charge
            end
        end
    end
end


# only 1 index position and 1 charge
function push_to_grid!(grid::MM,
                            charge::Float64,
                            index_pos::Int,
                            weights::Float64,
                            Nx::Int,
                            periodic::Bool=true) where {MM<:Union{SharedArray{Float64},Matrix{Float64}}}
        #@info "this is version A"
        if periodic
            #grid[ wrap_index!(index_pos[1], Nx) ] += weights * charge
            #grid[ index_pos[1] , index_pos[2] ] += weights[1] * weights[2] * charge

            #grid[wrap_index!(index_pos[1], Nx)] = grid[wrap_index!(index_pos[1], Nx)] ⊓ weights * charge
            grid[wrap_index!(index_pos[1], Nx)] = merge!(grid[wrap_index!(index_pos[1], Nx)], weights * charge)
        else
            if (index_pos <= Nx) & (index_pos > 0)
                #grid[index_pos] ⊓ weights * charge
                #grid[index_pos] += weights * charge
                grid[index_pos] = merge!(grid[index_pos], weights * charge)
            end
        end
end


# function push_to_grid!(grid::MM,
#                             charge::Float64,
#                             index_pos::Int,
#                             weights::Float64,
#                             Nx::Int,
#                             periodic::Bool = true) where MM <: Union{SharedMatrix{Float64}, Matrix{Float64}}
#         if periodic
#             grid[ wrap_index!(index_pos[1], Nx) ] += weights * charge
#             #grid[ index_pos[1] , index_pos[2] ] += weights[1] * weights[2] * charge
#         else
#             if (im <= Nx) & (im > 0)
#                 grid[im, : ] += wm * charge
#             end
#         end
# end




# end of module
end

# %%
# charges_grid = zeros(grid2d.Nx, grid2d.Ny)
# x_list = [5.0, 1.234, 0.5]#[0.1, 0.1, 0.5]#, 5.234, -4.3]
# y_list = [6.0, 2.34, 0.1]#[0.5, 0.5, 0.5]#, 9.99, -3.2]
# gridnotes.dx
#
#
# x_list = gridnotes.x .+ gridnotes.dx * 0.5
# y_list = gridnotes.y .+ gridnotes.dy* 0.9999999 ##[0.1, 0.1, 0.5]#, 5.234, -4.3]
# #y_list = [6.0, 2.34, 0.1]#[0.5, 0.5, 0.5]#, 9.99, -3.2]
# charges = gridnotes.y *0 .+1.0#, 1.0]#, 1.0, 1.0, 1.0]
#
# index_positions, weights = compute_weights_and_index(grid2d,x_list, y_list )
# push_to_grid!(charges_grid, charges , index_positions, weights  , grid2d.Nx , grid2d.Ny )

# %%
# n_particles = 100
# grid2d = mesh.TwoDGrid(eta_max,n_particles, eta_max ,50)
# gridnotes = mesh.TwoDGridNotes(grid2d)
#
# xmesh = [ x for x in gridnotes.x, y in gridnotes.y] # .+ 0.3
# ymesh = [ y for x in gridnotes.x, y in gridnotes.y] # .+0.5
#
# xp = xmesh .+ grid2d.dx* 0.1
# yp = ymesh .+ grid2d.dy* 0.1
#
# charge = xmesh .* 0
# charge[48:75, 5:30] .= sin.(gridnotes.x[48:75])
# #charge[20:40, 5:20] .= -1.0
#
# heatmap(gridnotes.x, gridnotes.y, charge')

# %%

# charges_grid = xmesh .* 0
#
# x_list = reshape(xp , (1, grid2d.Nx * grid2d.Ny) )
# y_list = reshape(yp , (1, grid2d.Nx * grid2d.Ny) )
# charge_list = reshape(charge , (1, grid2d.Nx * grid2d.Ny) )
#
# x_list      = x_list[charge_list .!= 0]
# y_list      = y_list[charge_list .!= 0]
# charge_list = charge_list[charge_list .!= 0]
#
# index_positions, weights = compute_weights_and_index(grid2d, x_list, y_list)
# #index_positions, weights = compute_weights_and_index(grid2d,dropdims(x_list, dims= 1),dropdims(y_list, dims= 1))
#
# push_to_grid!(charges_grid, charge_list , index_positions, weights  , grid2d.Nx , grid2d.Ny )
#
# #charge = charges_grid
#
# #contour(gridnotes.x, gridnotes.y, charges_grid)
# heatmap(gridnotes.x, gridnotes.y, charges_grid')



# %%
#dropdims(reshape(xp , (1, grid2d.Nx * grid2d.Ny) ), dims=1 )
# %%

# p_collect=[]
#
#
# cg0 = 0.12332 *1
#
#
# i_charge =copy(charge)
# charge_sum = sum(charge)
#
# i_charge
# for ti in 1:1:200
#
#     # xp = xmesh .+ grid2d.dx* cg0 .+ randn(100, 50) *0.01 + i_charge
#     # yp = ymesh .- grid2d.dy* cg0 .+ randn(100, 50) *0.01
#     xp = xmesh .+ i_charge*0.2
#     yp = ymesh .+ i_charge*0.05
#
#     #xp                       = grid1d.x .+ cg0 .*ip_charges  #.* (0.9 .+ 0.05 *(1 .+  rand(grid1d.Nx)))
#     #xp                       = grid1d.x .+ cg0 .* (0.9 .+ 0.05 *(1 .+  rand(grid1d.Nx)))
#     #xp                       = grid1d.x .+  0.1 * (.-0.5 .+  rand(grid1d.Nx))
#     #xp                       = xp .+  0.1 * (.-0.2 .+  rand(grid1d.Nx))
#
#     x_list = dropdims(reshape(xp , (1, grid2d.Nx * grid2d.Ny) ), dims=1 )
#     y_list = dropdims(reshape(yp , (1, grid2d.Nx * grid2d.Ny) ) , dims= 1)
#     charge_list = dropdims(reshape(i_charge , (1, grid2d.Nx * grid2d.Ny) ), dims = 1)
#
#     x_list      = x_list[charge_list .!= 0]
#     y_list      = y_list[charge_list .!= 0]
#     charge_list = charge_list[charge_list .!= 0]
#
#     index_positions, weights = compute_weights_and_index(grid2d, x_list, y_list)
#
#     charges_grid = xmesh .* 0
#     push_to_grid!(charges_grid, charge_list , index_positions, weights  , grid2d.Nx , grid2d.Ny )
#
#     @show charge_sum - Float64( sum(charges_grid) )
#
#     # propagate
#     i_charge = charges_grid
#     push!(p_collect, charges_grid)
# end

# %%
# anim = @animate for i in 1:1:length(p_collect)
#
#     #scatter(x_collect[i], p_collect[i], label ="new" )
#     heatmap(gridnotes.x, gridnotes.y, p_collect[i]', clim=(-1.0, 1.0))
#     # if i > 1
#     #     scatter!(x_collect[i-1], p_collect[i-1], color= "gray", alpha = 0.4, label="old")
#     # end
#     title!( "sum = $( round(sum(p_collect[i]) )) ")
#
#     #ylims!( 0, 1.1)
#     #xlims!(eta_min, eta_max,)
# end
# #round(sum(p_collect[1]) )
#
#
# gif(anim, "PIC_2d_test_fps30.gif", fps = 10)
