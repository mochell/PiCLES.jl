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

function _update_integrator_forcing!(PI::AbstractParticleInstance, winds::Tuple{Float64,Float64})
        forcing = PI.ODEIntegrator.forcing
        if forcing isa ForcingData
                forcing.u_wind = winds[1]
                forcing.v_wind = winds[2]
        end
        nothing
end

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

function ParticleToNode!(PI::AbstractParticleInstance, S::StateTypeL1, G::TwoDGrid, periodic_boundary::Bool, spline::Val=Val(1), wind::Tuple{Float64,Float64}=(0.0, 0.0))
        # NOTE: deprecated TwoDGrid path (issue #43); `wind` accepted for signature compatibility
        # with the MeshGrids method but unused here (this path deposits additively at order 1).
        # NOTE: TwoDGrid is deprecated (issue #43); higher-order B-spline deposition is only
        # supported on the live MeshGrids/CartesianGrid path. `spline` is accepted for dispatch
        # compatibility but ignored here — this path always deposits at order 1 (CIC).
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

function ParticleToNode!(PI::AbstractParticleInstance, S::StateTypeL1, G::MeshGrids, periodic_boundary::Bool, spline::Val{P}=Val(1), wind::Tuple{Float64,Float64}=(0.0, 0.0)) where {P}

        #u[4], u[5] are the x and y positions of the particle. For the CartesianGrid2D these are cooridnates relative to the particle node
        weights_and_index = PIC.compute_weights_and_index_mininal(PI.position_ij, PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5], spline)
        # @info PI.position_ij, weights_and_index

        #ui[1:2] .= PI.position_xy

        u_state = GetParticleEnergyMomentum(PI.ODEIntegrator.u)
        #@show u_state

        # `wind` is the local (u,v) at the particle; it drives the wind-sea-aware merge! contest
        # at each stencil node (so opposing groups do not additively cancel their momentum).
        PIC.push_to_grid!(S, u_state, weights_and_index, G.stats.Nx, G.stats.Ny, wind)
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


function reset_PI_u!(PI::AbstractParticleInstance; ui::CC) where {CC<:AbstractVector{Float64}}
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
                        wind_i::FF,
                        DT::Float64, 
                        log_energy_maximum::Float64,
                        wind_min_squared::Float64,
                        periodic_boundary::Bool,
                        default_particle::PP,
                        spline::Val{P}=Val(1),
                        windsea_merge::Bool=false,
                        ) where {PP<:Union{ParticleDefaults,Nothing},FF<:Union{ForcingCollection,ForcingData,NamedTuple{(:u, :v)},Tuple{Float64,Float64}},P}
        #@show PI.position_ij

        #if ~PI.boundary # if point is not a 

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

        winds_i_local = convert(Tuple{Float64,Float64},wind_i)::Tuple{Float64,Float64}
        winds_i_local_end = convert(Tuple{Float64,Float64},wind_i)::Tuple{Float64,Float64} # <--- this should be at final time after advanced. Not implimented yet.

        # advance particle
        if PI.on #& ~PI.boundary # if Particle is on and not boundary
        
                try
                        _update_integrator_forcing!(PI, winds_i_local)
                        step!(PI.ODEIntegrator, DT, true)
                catch e
                        @printf "error on advancing ODE:\n"
                        print("- time after fail $(PI.ODEIntegrator.t)\n ")
                        print("- error message: $(e)\n")
                        print("- push to failed\n")
                        print("- state of particle: $(PI.ODEIntegrator.u)\n")
                        uw, vw = winds_i_local
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

                # test if winds where strong enough
                if speed_square(winds_i_local_end[1], winds_i_local_end[2]) >= wind_min_squared
                        # winds are large eneough, reinit
                        ui = ResetParticleValues(default_particle, xy, winds_i_local_end, DT)
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
                
                @show winds_i_local_end
                ui = ResetParticleValues(default_particle, xy, winds_i_local_end, DT)
                @show PI.ODEIntegrator.u
                reset_PI_u!(PI, ui=ui)

        elseif  sum(isinf.(PI.ODEIntegrator.u[1:3])) > 0
                @info "position or Energy is inf"
                @show PI

                ui = ResetParticleValues(default_particle, xy, winds_i_local, DT)
                reset_PI_u!(PI, ui=ui)

        elseif PI.ODEIntegrator.u[1] > log_energy_maximum
                @info "e_max_log is reached"
                #@show PI

                ui = PI.ODEIntegrator.u
                ui[1] = log_energy_maximum
                # ui = ResetParticleValues(default_particle, xy, winds_start, DT)
                reset_PI_u!(PI, ui=ui)

        end

        #if PI.ODEIntegrator.u[1] > -13.0 #ODEs.log_energy_minimum # the minimum enerçy is distributed to 4 neighbouring particles
        if PI.on
                # windsea_merge=true forwards the local wind so the deposit runs the wind-sea
                # merge! contest; false forwards zero wind, for which merge! == additive deposit.
                merge_wind = windsea_merge ? winds_i_local : (0.0, 0.0)
                ParticleToNode!(PI, S, Grid, periodic_boundary, spline, merge_wind)
        end

        return PI
end

"""
        remesh!(PI::ParticleInstance2D, S::SharedMatrix{Float64, 3})
        Wrapper function that does everything necessary to remesh the particles.
        - pushes the Node State to particle instance
        - absorbed former `NodeToParticle!` logic (2D variant)
"""
function remesh!(PI::ParticleInstance2D, S::StateTypeL1,
                wind_i::FF,
                ti::Number, 
                ODEs::AbstractODESettings, DT::Float64,  #
                grid_stats::AbstractGridStatistics,
                minimal_state::Vector{Float64},
                default_particle::PP) where {PP<:Union{ParticleDefaults,Nothing},FF<:Union{ForcingCollection,ForcingData,NamedTuple{(:u, :v)},Tuple{Float64,Float64}}}        

        wind_tuple::Tuple{Float64,Float64} = wind_i
        wind_speed_squared = speed_square(wind_tuple[1], wind_tuple[2])

        last_t = PI.ODEIntegrator.t
        # load data from shared array
        u_state = Get_u_FromShared(PI, S)

        if typeof(grid_stats) <: MeshGridStatistics
                xy = (0.0, 0.0)
        else
                xy = PI.position_xy
        end



        # minimal_state[1] is the minimal Energy
        # minimal_state[2] is the minimal momentum squared
        #
        # Reconstruction floor on |m|. ODEs.m_amp_minimum is the absolute crash floor
        # (guards the c = m·e/(2|m|²) division, #63). When ODEs.windsea_alpha > 0 the floor
        # is raised to a local, wind-sea-aware level: α · |m| of a fully developed PM sea at
        # the local wind speed (FetchRelations.windsea_momentum_PM ∝ U³). This discards the
        # tiny, ill-defined net momentum that drives the #64 remesh limit cycle at wind/calm
        # interfaces, while scaling with the wind so it never over-deactivates an energetic
        # wind sea. α is tunable (PM is fully developed → an upper bound, so α is small).
        # dt-scaled wind-sea floor: the momentum eroded per unit time ∝ α/dt, so a constant
        # α drains the field at small dt (collapse). Scaling α ∝ min(1, dt/dt_ref) with
        # dt_ref = 10 min (the validated sweet spot) bounds the erosion and prevents the
        # small-dt collapse, making `windsea_alpha` safe to leave on by default. See the
        # eyewall-stability FINDINGS (analysis/stability) for the α/dt collapse calibration.
        α_eff = ODEs.windsea_alpha * min(1.0, ODEs.timestep / 600.0)
        m_amp_min = α_eff > 0 ?
                max(ODEs.m_amp_minimum, α_eff * FetchRelations.windsea_momentum_PM(sqrt(wind_speed_squared))) :
                ODEs.m_amp_minimum
        m_amp    = speed(u_state[2], u_state[3])

        if ~PI.boundary & (u_state[1] >= minimal_state[1]) & (speed_square(u_state[1], u_state[2]) >= minimal_state[2])
                if m_amp < m_amp_min
                        # M3: zero-net-momentum — deactivate; wind branch below will re-seed if winds are present
                        PI.on = false
                else
                        # interior nodes: convert node state to particle values and push to ODEIntegrator
                        ui = GetVariablesAtVertex(u_state, xy[1], xy[2], m_amp_min=m_amp_min)
                        reset_PI_ut!(PI, ui, last_t; dt_init=ODEs.dt)
                        PI.on = true
                end

        elseif ~PI.boundary & (wind_speed_squared >= ODEs.wind_min_squared)
                # local wind is strong enough to reset from default particle
                ui = ResetParticleValues(default_particle, xy, wind_tuple, DT)
                reset_PI_ut!(PI, ui, last_t; dt_init=ODEs.dt)
                PI.on = true

        elseif PI.boundary & (wind_speed_squared >= ODEs.wind_min_squared)
                # at boundary, reset particle if winds are strong enough
                ui = ResetParticleValues(default_particle, xy, wind_tuple, DT)
                reset_PI_ut!(PI, ui, last_t; dt_init=ODEs.dt)
                PI.on = true

        elseif (u_state[1] >= minimal_state[1])
                if m_amp < m_amp_min
                        # M3: zero-net-momentum with no wind to re-seed — deactivate
                        PI.on = false
                else
                        ui = GetVariablesAtVertex(u_state, xy[1], xy[2], m_amp_min=m_amp_min)
                        reset_PI_ut!(PI, ui, last_t; dt_init=ODEs.dt)
                        PI.on = true
                end

        else
                PI.on = false
        end

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
