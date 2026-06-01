module mapping_2D

using SharedArrays
using StaticArrays
#using OrdinaryDiffEq
using ...Solvers.RK35Integrator: step! #ODEIntegrator, 

using Printf

using ...ParticleMesh: TwoDGrid, TwoDGridNotes
import ...ParticleInCell as PIC

using ...FetchRelations

using ...custom_structures: ParticleInstance1D, ParticleInstance2D, MarkedParticleInstance, ForcingCollection
using ...ParticleSystems: ForcingData

using ..core_2D: GetParticleEnergyMomentum, GetVariablesAtVertex, Get_u_FromShared, ResetParticleValues, ParticleDefaults

using ...Architectures: AbstractParticleInstance, AbstractMarkedParticleInstance, AbstractODESettings, StateTypeL1
using ...Architectures: Grid2D, CartesianGrid, CartesianGridStatistics, CartesianGrid2D, CartesianGrid1D, AbstractGridStatistics, AbstractGrid, StandardRegular2D_old, MeshGrids, MeshGridStatistics


# using ...ParticleMesh: TwoDGrid, TwoDGridNotes, TwoDGridMesh
# using ...Grids.CartesianGrid: TwoDCartesianGridMesh, TwoDCartesianGridStatistics
###### remeshing routines ############


speed(x::Float64, y::Float64) = sqrt(x^2 + y^2)
speed_square(x::Float64, y::Float64) = x^2 + y^2

_wind_uv(forcing::ForcingCollection, x, y, t) = (forcing.u_wind(x, y, t), forcing.v_wind(x, y, t))
_wind_uv(forcing::ForcingData, x, y, t) = (forcing.u_wind, forcing.v_wind)
_wind_uv(winds::NamedTuple{(:u, :v)}, x, y, t) = (winds.u(x, y, t), winds.v(x, y, t))

"""
        ParticleToNode!(PI::AbstractParticleInstance, S::SharedMatrix, G::TwoDGrid)
Pushes particle values to the neighboring nodes following the ParticleInCell rules.
1.) get weights and indexes of the neighboring notes,
2.) convert the particle state to nodestate
3.) push the calculated state to the shared arrays

inputs:

PI      Particle instance
S       Shared array where particles are stored
G       (TwoDGrid) Grid that defines the nodepositions
"""

function ParticleToNode!(PI::AbstractParticleInstance, S::StateTypeL1, G::TwoDGrid, periodic_boundary::Bool)

        #u[4], u[5] are the x and y positions of the particle
        #index_positions, weights = PIC.compute_weights_and_index(G, PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5])
        weights_and_index = PIC.compute_weights_and_index_mininal(G, PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5])

        #ui[1:2] .= PI.position_xy
        #@show index_positions
        u_state = GetParticleEnergyMomentum(PI.ODEIntegrator.u)
        #@show u_state

        #PIC.push_to_grid!(S, u_state , index_positions,  weights, G.Nx, G.Ny , periodic_boundary)
        PIC.push_to_grid!(S, u_state , weights_and_index, G.Nx, G.Ny , periodic_boundary)
        nothing
end

function ParticleToNode!(PI::AbstractParticleInstance, S::StateTypeL1, G::MeshGrids, periodic_boundary::Bool)
        
        #u[4], u[5] are the x and y positions of the particle. For the CartesianGrid2D these are cooridnates relative to the particle node
        weights_and_index = PIC.compute_weights_and_index_mininal(PI.position_ij, PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5])
        # @info PI.position_ij, weights_and_index

        #ui[1:2] .= PI.position_xy

        u_state = GetParticleEnergyMomentum(PI.ODEIntegrator.u)
        #@show u_state

        #PIC.push_to_grid!(S, u_state , index_positions,  weights, G.Nx, G.Ny , periodic_boundary)
        PIC.push_to_grid!(S, u_state, weights_and_index, G.stats.Nx, G.stats.Ny)
        nothing
end

"""
        ParticleToNode!(PI::AbstractParticleInstance, S::SharedMatrix, u_state::Vector{Float64})
Pushes values u_state to the node in S of the particle origin of PI.
"""
function ParticleToNode!(PI::AbstractParticleInstance, S::SharedMatrix, u_state::Vector{Float64})

        S[PI.position_ij[1], PI.position_ij[2], :] = u_state
        nothing
end

# function set_u_and_t!(integrator, u_new::CC, t_new::Number) where CC <:Union{Vector{Float64},MVector}
#         integrator.u = u_new
#         integrator.t = t_new
# end


function reset_PI_u!(PI::AbstractParticleInstance; ui::CC) where {N,CC<:Union{Vector{Float64},MVector{N,Float64}}}
        # this method keeps the correct time for time varying forcing (~may 2023)
        # set_u!(PI.ODEIntegrator, ui)
        # u_modified!(PI.ODEIntegrator, true)
        # auto_dt_reset!(PI.ODEIntegrator)
        PI.ODEIntegrator.u = ui
        PI.ODEIntegrator.has_fsal = false
end


"""
    reset_PI_ut!(integrator, u_init, t_init; dt_init=nothing)

Reset an existing ODE integrator state in-place.

This helper updates:
- `PI.ODEIntegrator.u` to `u_init`
- `PI.ODEIntegrator.t` to `t_init`
- optionally `PI.ODEIntegrator.dt` to `dt_init` (if provided)

It is a lightweight state reset and does **not** rebuild the integrator,
recreate solver caches, or clear stored solution history.
"""
function reset_PI_ut!(PI::AbstractParticleInstance, u_init::CC, t_init; dt_init=nothing) where {N,CC<:Union{Vector{Float64},MVector{N,Float64}}}
    PI.ODEIntegrator.u = u_init
    PI.ODEIntegrator.t = t_init
    if dt_init !== nothing
        PI.ODEIntegrator.dt = dt_init
    end
        PI.ODEIntegrator.has_fsal = false
end

# old version
# function reset_PI_ut!(PI::AbstractParticleInstance; ui::CC, ti::Number) where CC <:Union{Vector{Float64},MVector}
#         # this method keeps the correct time for time varying forcing (~may 2023)
#         set_u_and_t!(PI.ODEIntegrator, ui, ti)
#         u_modified!(PI.ODEIntegrator, true)
#         auto_dt_reset!(PI.ODEIntegrator)
# end

"""
    reset_PI_t!(PI::AbstractParticleInstance; ti::Number)
Reset the time of an existing ODE integrator in-place.

This helper updates:
- `PI.ODEIntegrator.t` to `ti`

It is a lightweight state reset and does **not** rebuild the integrator,
recreate solver caches, or clear stored solution history.

"""
function reset_PI_t!(PI::AbstractParticleInstance; ti::Number)
        # this method keeps the correct time for time varying forcing (~may 2023)
        PI.ODEIntegrator.t = ti
        # set_t!(PI.ODEIntegrator, ti)
        # u_modified!(PI.ODEIntegrator, true)
        # auto_dt_reset!(PI.ODEIntegrator)
end

"""
    reinit_PI!(integrator, u_init, t_init, ODE_settings)
Reinitialize an ODE integrator with new initial conditions and settings.
This helper creates a new ODEIntegrator instance with the provided initial state `u_init`, time `t_init`, and ODE parameters from `ODE_settings`. It effectively resets the integrator's state and solver configuration, including any internal caches, to reflect the new initial conditions.
"""
function reinit_PI!(PI::AbstractParticleInstance, u::CC, t, ODE_settings) where {N,CC<:Union{Vector{Float64},MVector{N,Float64}}}
    return ODEIntegrator(PI.ODEIntegrator.model!, u, t, ODE_settings.Parameters;
                dt      =ODE_settings.dt,
                reltol  =ODE_settings.reltol,
                abstol  =ODE_settings.abstol,
                dtmin   =ODE_settings.dtmin,
                dtmax   =ODE_settings.dtmax)
end


######### Core routines for advancing and remeshing

"""
        advance!(PI::AbstractParticleInstance, S::SharedMatrix{Float64}, G::TwoDGrid, DT::Float64)
"""
function advance!(PI::AbstractParticleInstance,
                        S::StateTypeL1,
                        Failed::Vector{AbstractMarkedParticleInstance},
                        Grid::Union{Grid2D,MeshGrids},
                        forcing::FF,
                        DT::Float64, 
                        log_energy_maximum::Float64,
                        wind_min_squared::Float64,
                        periodic_boundary::Bool, 
                        default_particle::PP,
                        ) where {PP<:Union{ParticleDefaults,Nothing},FF<:Union{ForcingCollection,ForcingData,NamedTuple{(:u, :v)}}}
        return advance!(
                PI,
                S,
                Failed,
                Grid,
                forcing,
                nothing,
                DT,
                log_energy_maximum,
                wind_min_squared,
                periodic_boundary,
                default_particle,
        )
end

function advance!(PI::AbstractParticleInstance,
                        S::StateTypeL1,
                        Failed::Vector{AbstractMarkedParticleInstance},
                        Grid::Union{Grid2D,MeshGrids},
                        winds_i::Tuple{Float64,Float64},
                        DT::Float64,
                        log_energy_maximum::Float64,
                        wind_min_squared::Float64,
                        periodic_boundary::Bool,
                        default_particle::PP,
                        ) where {PP<:Union{ParticleDefaults,Nothing}}
        return advance!(
                PI,
                S,
                Failed,
                Grid,
                nothing,
                winds_i,
                DT,
                log_energy_maximum,
                wind_min_squared,
                periodic_boundary,
                default_particle,
        )
end

function advance!(PI::AbstractParticleInstance,
                        S::StateTypeL1,
                        Failed::Vector{AbstractMarkedParticleInstance},
                        Grid::Union{Grid2D,MeshGrids},
                        forcing::Union{FF,Nothing},
                        winds_i::Union{Nothing,Tuple{Float64,Float64}},
                        DT::Float64, 
                        log_energy_maximum::Float64,
                        wind_min_squared::Float64,
                        periodic_boundary::Bool, 
                        default_particle::PP,
                        ) where {PP<:Union{ParticleDefaults,Nothing},FF<:Union{ForcingCollection,ForcingData,NamedTuple{(:u, :v)}}}
        #@show PI.position_ij

        #if ~PI.boundary # if point is not a 
        t_start  =  copy(PI.ODEIntegrator.t)
        # add_saveat!(PI.ODEIntegrator, PI.ODEIntegrator.t )
        # savevalues!(PI.ODEIntegrator)
        
        # set the position in particle state vector either to the node position or to the relative position in the CartesianGrid
        if typeof(Grid) <: MeshGrids
                xy = (0.0,0.0)
                # @info "advance: CartesianGrid"
        elseif typeof(Grid) <: StandardRegular2D_old
                xy = PI.position_xy[1], PI.position_xy[2]
                # @info "advance: StandardRegular2D_old"
        else
                @info "advance: no grid detected"
        end


        # advance particle
        if PI.on #& ~PI.boundary # if Particle is on and not boundary
        
                try
                        step!(PI.ODEIntegrator, DT, true)
                catch e
                        @printf "error on advancing ODE:\n"
                        print("- time after fail $(PI.ODEIntegrator.t)\n ")
                        print("- error message: $(e)\n")
                        print("- push to failed\n")
                        print("- state of particle: $(PI.ODEIntegrator.u)\n")
                        uw, vw = isnothing(winds_i) ?
                                _wind_uv(forcing, PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5], PI.ODEIntegrator.t) :
                                winds_i
                        print("- winds are: $(uw)\n")
                        print("- winds are: $(vw)\n")
                        push!(Failed,
                                MarkedParticleInstance(
                                        copy(PI),
                                        copy(PI.ODEIntegrator.t),
                                        copy(PI.ODEIntegrator.u),
                                        0
                                ))
                        return

                end
        
        elseif ~PI.on #& ~PI.boundary # particle is off, test if there was windsea

                t_end = t_start + DT
                wind_end = isnothing(winds_i) ?
                        convert(Tuple{Float64,Float64}, _wind_uv(forcing, PI.position_xy[1], PI.position_xy[2], t_end))::Tuple{Float64,Float64} :
                        winds_i

                # test if winds where strong enough
                if speed_square(wind_end[1], wind_end[2]) >= wind_min_squared
                        # winds are large eneough, reinit
                        ui = ResetParticleValues(default_particle, xy, wind_end, DT)
                        reset_PI_u!(PI, ui =ui)
                        PI.on = true
                end

        else    #particle is on and boundary
                
                #@info "particle is on and boundary"
                # particle stays off or is bounaday. do not advance
                PI.on=false
                return
        end

        # # check if integration reached limits or is nan, or what ever. if so, reset
        if sum(isnan.(PI.ODEIntegrator.u[1:3])) > 0
                @info "position or Energy is nan, reset"
                @info PI.position_ij
                @show PI
                
                t_end = t_start + DT
                winds_start = isnothing(winds_i) ?
                        convert(Tuple{Float64,Float64}, _wind_uv(forcing, PI.position_xy[1], PI.position_xy[2], t_end))::Tuple{Float64,Float64} :
                        winds_i
                @show winds_start

                ui = ResetParticleValues(default_particle, xy, winds_start, DT)
                @show PI.ODEIntegrator.u
                reset_PI_u!(PI, ui=ui)

        elseif  sum(isinf.(PI.ODEIntegrator.u[1:3])) > 0
                @info "position or Energy is inf"
                @show PI

                winds_start = isnothing(winds_i) ?
                        convert(Tuple{Float64,Float64}, _wind_uv(forcing, PI.position_xy[1], PI.position_xy[2], t_start))::Tuple{Float64,Float64} :
                        winds_i

                ui = ResetParticleValues(default_particle, xy, winds_start, DT)
                reset_PI_u!(PI, ui=ui)

        elseif PI.ODEIntegrator.u[1] > log_energy_maximum
                @info "e_max_log is reached"
                #@show PI

                # winds_start = convert(Tuple{Float64,Float64},
                #                         (winds.u(PI.position_xy[1], PI.position_xy[2], t_start),
                #                         winds.v(PI.position_xy[1], PI.position_xy[2], t_start)))::Tuple{Float64,Float64}

                ui = PI.ODEIntegrator.u
                ui[1] = log_energy_maximum
                # ui = ResetParticleValues(default_particle, xy, winds_start, DT)
                reset_PI_u!(PI, ui=ui)

        end

        #if PI.ODEIntegrator.u[1] > -13.0 #ODEs.log_energy_minimum # the minimum enerçy is distributed to 4 neighbouring particles
        if PI.on 
                ParticleToNode!(PI, S, Grid, periodic_boundary)
        end

        return PI
end

"""
        remesh!(PI::ParticleInstance2D, S::SharedMatrix{Float64, 3})
        Wrapper function that does everything necessary to remesh the particles.
        - pushes the Node State to particle instance
"""
function remesh!(PI::ParticleInstance2D, S::StateTypeL1,
                forcing::FF,
                ti::Number, 
                ODEs::AbstractODESettings, DT::Float64,  #
                grid_stats::AbstractGridStatistics,
                minimal_state::Vector{Float64},
                default_particle::PP) where {PP<:Union{ParticleDefaults,Nothing},FF<:Union{ForcingCollection,ForcingData,NamedTuple{(:u, :v)}}}        
        winds_i_local::Tuple{Float64,Float64} = _wind_uv(forcing, PI.position_xy[1], PI.position_xy[2], ti)

        PI = NodeToParticle!(PI, S,
                        winds_i_local,
                        grid_stats,
                        minimal_state,
                        ODEs.wind_min_squared,
                        default_particle,
                        ODEs.log_energy_minimum,
                        ODEs.dt,
                        DT)
        return PI
end

function remesh!(PI::ParticleInstance2D, S::StateTypeL1,
                winds_i::Tuple{Float64,Float64},
                ti::Number,
                ODEs::AbstractODESettings, DT::Float64,
                grid_stats::AbstractGridStatistics,
                minimal_state::Vector{Float64},
                default_particle::PP) where {PP<:Union{ParticleDefaults,Nothing}}
        PI = NodeToParticle!(PI, S,
                        winds_i,
                        grid_stats,
                        minimal_state,
                        ODEs.wind_min_squared,
                        default_particle,
                        ODEs.log_energy_minimum,
                        ODEs.dt,
                        DT)
        return PI
end

function remesh!(PI::ParticleInstance2D, S::StateTypeL1,
                forcing::Union{FF,Nothing},
                winds_i::Union{Nothing,Tuple{Float64,Float64}},
                ti::Number, 
                ODEs::AbstractODESettings, DT::Float64,  #
                grid_stats::AbstractGridStatistics,
                minimal_state::Vector{Float64},
                default_particle::PP) where {PP<:Union{ParticleDefaults,Nothing},FF<:Union{ForcingCollection,ForcingData,NamedTuple{(:u, :v)}}}        
                
        winds_i_local::Tuple{Float64,Float64} = isnothing(winds_i) ? _wind_uv(forcing, PI.position_xy[1], PI.position_xy[2], ti) : winds_i
        
        PI = NodeToParticle!(PI, S, 
                        winds_i_local, 
                        grid_stats,
                        minimal_state,
                        ODEs.wind_min_squared,
                        default_particle, 
                        ODEs.log_energy_minimum, 
                        ODEs.dt,
                        DT)
        return PI
end


"""
        NodeToParticle!(PI::AbstractParticleInstance, S::SharedMatrix)
        Pushes node value to particle:
        - If Node value is smaller than a minimal value, the particle is renintialized
        - If Node value is okey, it is converted to state variable and pushed to particle.
        - The particle position is set to the node positions
"""
function NodeToParticle!(PI::AbstractParticleInstance, S::StateTypeL1,
        wind_tuple::Tuple{Float64,Float64}, 
        grid_stats::MeshGridStatistics,
        minimal_state::Vector{Float64},
        wind_min_squared::Float64, 
        default_particle::PP, 
        e_min_log::Number,
        dt_init::Float64, 
        DT::Float64,) where {PP<:Union{ParticleDefaults,Nothing}}

        # load data from shared array
        u_state = Get_u_FromShared(PI, S)

        if typeof(grid_stats) <: MeshGridStatistics
                xy = ( 0.0, 0.0 )
                # if (PI.position_ij == (10, 10)) | (PI.position_ij == (40, 10))
                #         @info "NodeToParticle!: CartesianGridStatistics"
                # end
        else
                xy = PI.position_xy
                # if (PI.position_ij == (10, 10)) | (PI.position_ij == (40, 10))
                #         @info "NodeToParticle!: Standard grid"
                # end
        end

        # on_before = PI.on
        last_t = PI.ODEIntegrator.t
        # minimal_state[1] is the minimal Energy  
        # minimal_state[2] is the minimal momentum squared  
        if ~PI.boundary & (u_state[1] >= minimal_state[1]) & (speed_square(u_state[2], u_state[3]) >= minimal_state[2]) # all integrior nodes: convert note to particle values and push to ODEIntegrator

                #@show "u_state", u_state
                ui = GetVariablesAtVertex(u_state, xy[1], xy[2])
                #@info exp(ui[1]), ui[2], ui[4]/1e3, ui[5]/1e3              
                reset_PI_ut!(PI, ui, last_t; dt_init=dt_init)
                PI.on = true

                #@show PI.ODEIntegrator.t, PI.ODEIntegrator.u
                
        elseif ~PI.boundary & (speed_square(wind_tuple[1], wind_tuple[2]) >= wind_min_squared) #minimal windsea is too small but local winds are strong enough  #(u_state[1] < exp(e_min_log)) | PI.boundary
                # test if particle is below energy threshold, or
                #      if particle is at the boundary

                ui = ResetParticleValues(default_particle, xy, wind_tuple, DT) # returns winds sea given DT and winds

                # OrdinaryDiffEq reinit method, creates new integrator instance, resets cache, and erases solution history 
                # reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)#, reinit_callbacks=true)
                # reset_PI_t!(PI, ti=last_t)

                # new stuff
                reset_PI_ut!(PI, ui, last_t; dt_init=dt_init)

                PI.on = true

        elseif PI.boundary & (speed_square(wind_tuple[1], wind_tuple[2]) >= wind_min_squared) # at the boundary, reset particle if winds are strong enough

                ui = ResetParticleValues(default_particle, xy, wind_tuple, DT) # returns winds sea given DT and winds

                # OrdinaryDiffEq reinit method, creates new integrator instance, resets cache, and erases solution history 
                # reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)#, reinit_callbacks=true)
                # reset_PI_t!(PI, ti=last_t)

                reset_PI_ut!(PI, ui, last_t; dt_init=dt_init)
                # @info default_particle, ui
                PI.on = true
        elseif (u_state[1] >= minimal_state[1])
                ui = GetVariablesAtVertex(u_state, xy[1], xy[2])
                #@info exp(ui[1]), ui[2], ui[4]/1e3, ui[5]/1e3              
                reset_PI_ut!(PI, ui, last_t; dt_init=dt_init)
                PI.on = true

        else # particle is below energy threshold & on boundary
                # PI.ODEIntegrator.u = ResetParticleValues(minimal_particle, xy, wind_tuple, DT)
                # if ~PI.boundary
                #         @info u_state
                # end
                PI.on = false

        end
        
        # if PI.position_ij[2] == 6
        #         @info "end of NodeToParticle!: $(PI.position_ij) particle on change :$(on_before) to $(PI.on)"
        # end
        return PI

end




""" shows total energy of the all particles """
function ShowTotalEnergyChange(ParticleCollection, u_sum_m1)
        u_sum = zeros(Nstate)
        for a_particle in ParticleCollection
                u_sum +=  GetParticleEnergyMomentum(a_particle.ODEIntegrator.u)
        end
        @show u_sum_m1 - u_sum
        return u_sum
end

end
