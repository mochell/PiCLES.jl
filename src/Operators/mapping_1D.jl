module mapping_1D

using SharedArrays
#using OrdinaryDiffEq
using Printf

using ...ParticleMesh: OneDGrid, OneDGridNotes

using ...ParticleInCell

#include("../Utils/FetchRelations.jl")
import ...FetchRelations

using ...custom_structures: ParticleInstance1D, MarkedParticleInstance

using ..core_1D: GetParticleEnergyMomentum, GetVariablesAtVertex, Get_u_FromShared, ResetParticleValues, ParticleDefaults

using ...Architectures: AbstractParticleInstance, AbstractMarkedParticleInstance, AbstractODESettings


speed(x::Float64, y::Float64) = sqrt(x^2 + y^2)
speed_square(x::Float64, y::Float64) = x^2 + y^2


###### remeshing routines ############

"""
        ParticleToNode!(PI::AbstractParticleInstance, S::SharedMatrix, G::OneDGrid)
Pushes particle values to the neighboring nodes following the ParticleInCell rules.
1.) get weights and indexes of the neighboring notes,
2.) convert the particle state to nodestate
3.) push the calculated state to the shared arrays

inputs:

PI      Particle instance
S       Shared array where particles are stored
G       (TwoDGrid) Grid that defines the nodepositions
"""
function ParticleToNode!(PI::AbstractParticleInstance, S::SharedMatrix, G::OneDGrid, periodic_boundary::Bool)

        #u[3] is the x position of the particle
        index_positions, weights = ParticleInCell.compute_weights_and_index(G, PI.ODEIntegrator.u[3])
        #ui[1:2] .= PI.position_xy
        #@show index_positions
        u_state = GetParticleEnergyMomentum(PI.ODEIntegrator.u)
        #@show u_state, index_positions, weights
        ParticleInCell.push_to_grid!(S, u_state, index_positions, weights, G.Nx, periodic_boundary)
        nothing
end

"""
        ParticleToNode!(PI::AbstractParticleInstance, S::SharedMatrix, u_state::Vector{Float64})
Pushes values u_state to the node in S of the particle origin of PI.
"""
function ParticleToNode!(PI::AbstractParticleInstance, S::SharedMatrix, u_state::Vector{Float64})

        S[PI.position_ij[1], :] = u_state
        nothing
end



function set_u_and_t!(integrator, u_new, t_new)
        integrator.u = u_new
        integrator.t = t_new
end

function reset_PI!(PI::AbstractParticleInstance, ui::Vector{Float64})
        set_u!(PI.ODEIntegrator, ui)
        u_modified!(PI.ODEIntegrator, true)
        auto_dt_reset!(PI.ODEIntegrator)
end



######### Core routines for advancing and remeshing

"""
        advance!(PI::AbstractParticleInstance, S::SharedMatrix{Float64}, G::OneDGrid, DT::Float64)
"""
function advance!(PI::AbstractParticleInstance,
                        S::SharedMatrix{Float64},
                        Failed::Vector{AbstractMarkedParticleInstance},
                        G::OneDGrid,
                        u,
                        DT::Float64, #ODEs::AbstractODESettings, 
                        log_energy_maximum::Float64,
                        wind_min_squared::Float64,
                        periodic_boundary::Bool, 
                        default_particle::PP,
                        ) where {PP<:Union{ParticleDefaults,Nothing}}

                        
        t_start = copy(PI.ODEIntegrator.t)
        add_saveat!(PI.ODEIntegrator, PI.ODEIntegrator.t)
        savevalues!(PI.ODEIntegrator)

        # advance particle
        if PI.on & ~PI.boundary # if Particle is on and not boundary
                
                try
                        step!(PI.ODEIntegrator, DT, true)
                catch e
                        @printf "error on advancing ODE:\n"
                        print("- time after fail $(PI.ODEIntegrator.t)\n ")
                        print("- error message: $(e)\n")
                        print("- push to failed\n")
                        print("- state of particle: $(PI.ODEIntegrator.u)\n")
                        print("- winds are: $(u( PI.ODEIntegrator.u[3], PI.ODEIntegrator.t))\n")
                        push!(Failed,
                                MarkedParticleInstance(
                                        copy(PI),
                                        copy(PI.ODEIntegrator.t),
                                        copy(PI.ODEIntegrator.u),
                                        PI.ODEIntegrator.sol.retcode
                                ))
                        return

                end

        elseif ~PI.on & ~PI.boundary # particle is off, test if there was windsea at the end of the integration 
                        # includes boundary points
                
                #@info "particle is off," PI.position_ij, PI.on, PI.boundary
                t_end = t_start + DT
                wind_end = convert( Float64, u(PI.position_xy, t_end) )::Float64

                # test if winds where strong enough
                if wind_end^2 >= wind_min_squared
                        # winds are large eneough, reinit
                        ui = ResetParticleValues(default_particle, PI, wind_end, DT)
                        reset_PI!(PI, ui)
                        PI.on = true
                end

                # else: winds are not strong enough, particle stays off
        
        else    # particle is on and boundary
                #@info "particle is on and boundary"
                # particle stays off or is bounaday. do not advance
                PI.on = false
                return
        end

        # # check if integration reached limits or is nan, or what ever. if so, reset
        if sum(isnan.(PI.ODEIntegrator.u[1:3])) > 0
                @info "position or Energy is nan, reset"
                @info PI.position_ij
                @show PI
                
                t_end = t_start + DT
                wind_end = convert(Float64, u(PI.position_xy, t_end))::Float64
                @show winds_start

                ui = ResetParticleValues(default_particle, PI, wind_end, DT)
                @show PI.ODEIntegrator.u
                reset_PI!(PI, ui)

        elseif  sum(isinf.(PI.ODEIntegrator.u[1:3])) > 0
                @info "position or Energy is inf"
                @show PI

                winds_start = convert(Float64, u(PI.position_xy, t_start))::Float64

                ui = ResetParticleValues(default_particle, PI, winds_start, DT)
                reset_PI!(PI, ui)

        elseif PI.ODEIntegrator.u[1] > log_energy_maximum
                @info "e_max_log is reached"
                #@show PI

                winds_start = convert(Float64, u(PI.position_xy, t_start))::Float64

                ui = ResetParticleValues(default_particle, PI, winds_start, DT)
                reset_PI!(PI, ui)
        end

        #if PI.ODEIntegrator.u[1] > -13.0 #ODEs.log_energy_minimum # the minimum enerçy is distributed to 4 neighbouring particles
        if PI.on 
                ParticleToNode!(PI, S, G, periodic_boundary)
        end
        
        #return PI
end

"""
        remesh!(PI::ParticleInstance1D, S::SharedMatrix{Float64, 3})
        Wrapper function that does everything necessary to remesh the particles.
        - pushes the Node State to particle instance
"""
function remesh!(PI::ParticleInstance1D, S::SharedMatrix{Float64}, 
                u,
                ti::Number, 
                ODEs::AbstractODESettings, DT::Float64,
                minimal_particle::Vector{Float64}, minimal_state::Vector{Float64},
                default_particle::PP) where {PP<:Union{ParticleDefaults,Nothing}}
        
        ui = u(PI.position_xy[1], ti)
        
        NodeToParticle!(PI, S, 
                        ui, 
                        minimal_particle,
                        minimal_state,
                        ODEs.wind_min_squared,
                        default_particle,
                        ODEs.log_energy_minimum,
                        DT)
        
        #return PI
end


"""
        NodeToParticle!(PI::AbstractParticleInstance, S::SharedMatrix)
Pushes node value to particle:
- If Node value is smaller than a minimal value, the particle is renintialized
- If Node value is okey, it is converted to state variable and pushed to particle.
- The particle position is set to the node positions
"""
function NodeToParticle!(PI::AbstractParticleInstance, S::SharedMatrix,
        u_wind::Number,
        minimal_particle::Vector{Float64},
        minimal_state::Vector{Float64},
        wind_min_squared::Float64,
        default_particle::PP,
        e_min_log::Number,
        DT::Float64) where {PP<:Union{ParticleDefaults,Nothing}}

        # load data from shared array
        u_state = Get_u_FromShared(PI, S)

        #last_t = ti 
        last_t = PI.ODEIntegrator.t

        # notes :
        # minimal_state[1] is the minmal Energy  
        # minimal_state[2] is the minmal momentum squared
        # u_state[3] is 0 in 1D version, so we good. 


        if ~PI.boundary & (u_state[1] >= minimal_state[1]) & (u_state[2]^2 >= minimal_state[2])
                # all interior nodes that have a higher energy than the minimal energy 
                # convert note to particle values and push to ODEIntegrator

                ui = GetVariablesAtVertex(u_state, PI.position_xy)

                # this method keeps the correct time for time varying forcing (~may 2023)
                set_u_and_t!(PI.ODEIntegrator, ui, last_t)
                u_modified!(PI.ODEIntegrator, true)
                auto_dt_reset!(PI.ODEIntegrator)

                PI.on = true

        elseif ~PI.boundary & (u_wind^2 >= wind_min_squared)
                # all interior nodes that don't have a enoug energy but there is windsea 
                #minimal windsea is not big enough but local winds are strong enough  #(u_state[1] < exp(e_min_log)) | PI.boundary

                ui = ResetParticleValues(default_particle, PI, u_wind, DT)

                # set wind sea to shared array in hindsight to avoid empty cells in divergent regions
                #ParticleToNode!(PI, S, GetParticleEnergyMomentum(ui))

                reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)#, reinit_callbacks=true)
                set_t!(PI.ODEIntegrator, last_t)
                u_modified!(PI.ODEIntegrator, true)
                auto_dt_reset!(PI.ODEIntegrator)

                PI.on = true

        else    # all other cases

                PI.on = false
        end

        nothing

end




""" shows total energy of the all particles """
function ShowTotalEnergyChange(ParticleCollection, u_sum_m1)
        u_sum = zeros(Nstate)
        for a_particle in ParticleCollection
                u_sum += GetParticleEnergyMomentum(a_particle.ODEIntegrator.u)
        end
        @show u_sum_m1 - u_sum
        return u_sum
end

end
