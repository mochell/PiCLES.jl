module mapping_2D

using SharedArrays
using StaticArrays
using DifferentialEquations
using Printf
using Random, Distributions

using ...ParticleMesh: TwoDGrid, TwoDGridNotes
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, ProjetionKernel, TwoDCartesianGridStatistics
import ...ParticleInCell as PIC

using ...FetchRelations

using ...custom_structures: ParticleInstance1D, ParticleInstance2D, MarkedParticleInstance

using ..core_2D_spread: GetParticleEnergyMomentum, GetVariablesAtVertex, Get_u_FromShared, ResetParticleValues, ParticleDefaults, InitParticleInstance

using ...Architectures: AbstractParticleInstance, AbstractMarkedParticleInstance, AbstractODESettings, StateTypeL1, Abstract2DModel, Abstract2DStochasticModel
using ...Architectures: Grid2D, CartesianGrid, CartesianGridStatistics, CartesianGrid2D, CartesianGrid1D, AbstractGridStatistics, AbstractGrid, StandardRegular2D_old, MeshGrids, MeshGridStatistics

###### remeshing routines ############


speed(x::Float64, y::Float64) = sqrt(x^2 + y^2)
speed_square(x::Float64, y::Float64) = x^2 + y^2

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

function ParticleToNode!(PI::AbstractParticleInstance, particlesAtNode::Array{Array{Array{Any,1},1},1}, S::SharedArray, G::TwoDCartesianGridMesh, periodic_boundary::Bool)
        
        #u[4], u[5] are the x and y positions of the particle
        #index_positions, weights = PIC.compute_weights_and_index(G, PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5])
        weights_and_index = PIC.compute_weights_and_index_mininal(G, PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5])

        #ui[1:2] .= PI.position_xy
        u_state = GetParticleEnergyMomentum(PI.ODEIntegrator.u)
        #@show u_state

        #PIC.push_to_grid!(S, particlesAtNode, PI, u_state , index_positions,  weights, G.stats.Nx.N, G.stats.Ny.N , periodic_boundary)
        PIC.push_to_grid!(S, particlesAtNode, PI, u_state , weights_and_index, G.stats.Nx.N, G.stats.Ny.N , periodic_boundary)
        nothing
end

function ParticleToNode!(PI::AbstractParticleInstance, S::StateTypeL1, G::TwoDGrid, periodic_boundary::Bool)

        #u[4], u[5] are the x and y positions of the particle
        #index_positions, weights = PIC.compute_weights_and_index(G, PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5])
        weights_and_index = PIC.compute_weights_and_index_mininal(G, PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5])

        #ui[1:2] .= PI.position_xy
        #@show index_positions
        u_state = GetParticleEnergyMomentum(PI.ODEIntegrator.u)
        #@show u_state

        #PIC.push_to_grid!(S, u_state , index_positions,  weights, G.stats.Nx.N, G.stats.Ny.N , periodic_boundary)
        PIC.push_to_grid!(S, u_state , weights_and_index, G.stats.Nx.N, G.stats.Ny.N , periodic_boundary)
        nothing
end

function ParticleToNode!(PI::AbstractParticleInstance, S::StateTypeL1, G::MeshGrids, periodic_boundary::Bool)
        
        #u[4], u[5] are the x and y positions of the particle. For the CartesianGrid2D these are cooridnates relative to the particle node
        weights_and_index = PIC.compute_weights_and_index_mininal(PI.position_ij, PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5])
        # @info PI.position_ij, weights_and_index

        #ui[1:2] .= PI.position_xy

        u_state = GetParticleEnergyMomentum(PI.ODEIntegrator.u)
        #@show u_state

        #PIC.push_to_grid!(S, u_state , index_positions,  weights, G.stats.Nx.N, G.stats.Ny.N , periodic_boundary)
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

function set_u_and_t_stochas!(integrator, u_new::CC, t_new::Number) where CC <:Union{Vector{Float64},MVector}
        # adding a small deviation due to directional spreading
        d = Normal(0, u_new[6]^2)
        delta_phi = rand(d,1)
        c_x = u_new[2] * cos(delta_phi[1]) - u_new[3] * sin(delta_phi[1])
        c_y = u_new[2] * sin(delta_phi[1]) + u_new[3] * cos(delta_phi[1])
        u_new[2] = c_x
        u_new[3] = c_y
        integrator.u = u_new
        integrator.t = t_new
end

function set_u_and_t!(integrator, u_new::CC, t_new::Number) where CC <:Union{Vector{Float64},MVector}
        integrator.u = u_new
        integrator.t = t_new
end


function reset_PI_u!(PI::AbstractParticleInstance; ui::CC) where CC<:Union{Vector{Float64},MVector{Float64}}  
        # this method keeps the correct time for time varying forcing (~may 2023)
        set_u!(PI.ODEIntegrator, ui)
        u_modified!(PI.ODEIntegrator, true)
        auto_dt_reset!(PI.ODEIntegrator)
end


function reset_PI_ut!(PI::AbstractParticleInstance; ui::CC, ti::Number, stochas::Bool) where CC <:Union{Vector{Float64},MVector}
        # this method keeps the correct time for time varying forcing (~may 2023)
        if stochas
                set_u_and_t_stochas!(PI.ODEIntegrator, ui, ti)
                @info "called the correct version"
        else
                @info "called the wrong version"
                set_u_and_t!(PI.ODEIntegrator, ui, ti)
        end
        u_modified!(PI.ODEIntegrator, true)
        auto_dt_reset!(PI.ODEIntegrator)
end

function reset_PI_t!(PI::AbstractParticleInstance; ti::Number)
        # this method keeps the correct time for time varying forcing (~may 2023)
        set_t!(PI.ODEIntegrator, ti)
        u_modified!(PI.ODEIntegrator, true)
        auto_dt_reset!(PI.ODEIntegrator)
end

######### Core routines for advancing and remeshing

"""
        advance!(PI::AbstractParticleInstance, S::SharedMatrix{Float64}, G::TwoDGrid, DT::Float64)
"""
function advance!(PI::AbstractParticleInstance,
                        particlesAtNode::Array{Array{Array{Any,1},1},1},
                        S::StateTypeL1,
                        Failed::Vector{AbstractMarkedParticleInstance},
                        Grid::Union{Grid2D,MeshGrids},
                        winds::NamedTuple{(:u, :v)},
                        DT::Float64, 
                        log_energy_maximum::Float64,
                        wind_min_squared::Float64,
                        periodic_boundary::Bool, 
                        default_particle::PP,
                        ) where {PP<:Union{ParticleDefaults,Nothing}}
        #@show PI.position_ij

        #if ~PI.boundary # if point is not a 
        t_start  =  copy(PI.ODEIntegrator.t)
        add_saveat!(PI.ODEIntegrator, PI.ODEIntegrator.t )
        savevalues!(PI.ODEIntegrator)
        
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
                        PI.ODEIntegrator.u[4] += DT*PI.ODEIntegrator.u[2]
                        PI.ODEIntegrator.u[5] += DT*PI.ODEIntegrator.u[3]
                        # step!(PI.ODEIntegrator, DT, true)
                catch e
                        @printf "error on advancing ODE:\n"
                        print("- time after fail $(PI.ODEIntegrator.t)\n ")
                        print("- error message: $(e)\n")
                        print("- push to failed\n")
                        print("- state of particle: $(PI.ODEIntegrator.u)\n")
                        print("- winds are: $(winds.u( PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5], PI.ODEIntegrator.t))\n")
                        print("- winds are: $(winds.v( PI.ODEIntegrator.u[4], PI.ODEIntegrator.u[5], PI.ODEIntegrator.t))\n")
                        push!(Failed,
                                MarkedParticleInstance(
                                        copy(PI),
                                        copy(PI.ODEIntegrator.t),
                                        copy(PI.ODEIntegrator.u),
                                        PI.ODEIntegrator.sol.retcode
                                ))
                        return

                end
        
        elseif ~PI.on #& ~PI.boundary # particle is off, test if there was windsea

                t_end = t_start + DT
                wind_end = convert(Tuple{Float64,Float64},
                                (winds.u(PI.position_xy[1], PI.position_xy[2], t_end),
                                winds.v(PI.position_xy[1], PI.position_xy[2], t_end)))::Tuple{Float64,Float64}

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
                winds_start = convert(  Tuple{Float64,Float64},
                        (winds.u(PI.position_xy[1], PI.position_xy[2], t_end),
                        winds.v(PI.position_xy[1], PI.position_xy[2], t_end)))::Tuple{Float64,Float64}
                @show winds_start

                ui = ResetParticleValues(default_particle, xy, winds_start, DT)
                @show PI.ODEIntegrator.u
                reset_PI_u!(PI, ui=ui)

        elseif  sum(isinf.(PI.ODEIntegrator.u[1:3])) > 0
                @info "position or Energy is inf"
                @show PI

                winds_start = convert(Tuple{Float64,Float64},
                                        (winds.u(PI.position_xy[1], PI.position_xy[2], t_start),
                                        winds.v(PI.position_xy[1], PI.position_xy[2], t_start)))::Tuple{Float64,Float64}

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
                ParticleToNode!(PI, particlesAtNode, S, Grid, periodic_boundary)
                # ParticleToNode!(PI, S, Grid, periodic_boundary)
        end

        #return PI
end

"""
        remesh!(PI::ParticleInstance2D, S::SharedMatrix{Float64, 3})
        Wrapper function that does everything necessary to remesh the particles.
        - pushes the Node State to particle instance
"""
function remesh!(PI::ParticleInstance2D, S::StateTypeL1,
                winds::NamedTuple{(:u, :v)}, 
                ti::Number, 
                ODEs::AbstractODESettings, DT::Float64,  #
                grid_stats::AbstractGridStatistics,
                minimal_state::Vector{Float64},
                default_particle::PP) where {PP<:Union{ParticleDefaults,Nothing}}        
                
        winds_i::Tuple{Float64,Float64} = winds.u(PI.position_xy[1], PI.position_xy[2], ti), winds.v(PI.position_xy[1], PI.position_xy[2], ti)
        
        NodeToParticle!(PI, S, 
                        winds_i, 
                        grid_stats,
                        minimal_state,
                        ODEs.wind_min_squared,
                        default_particle, 
                        ODEs.log_energy_minimum, 
                        DT)
        #return PI
end

"""
        remesh!(PI::ParticleInstance2D, S::SharedMatrix{Float64, 3})
        Wrapper function that does everything necessary to remesh the particles.
        - pushes the Node State to particle instance
"""
function remesh!(i::Int64, j::Int64, model::Abstract2DStochasticModel, DT::Float64)
        winds = model.winds
        ti = model.clock.time
        grid = model.grid
        
        x = grid.xmin + grid.dx*(i-1)
        y = grid.ymin + grid.dy*(j-1)
        winds_i::Tuple{Float64,Float64} = winds.u(x, y, ti), winds.v(x, y, ti)
        
        NodeToParticle!(i, j, x, y, model, winds_i, DT)
end

"""
        NodeToParticle!(PI::AbstractParticleInstance, S::SharedMatrix)
Pushes node value to particle:
- If Node value is smaller than a minimal value, the particle is renintialized
- If Node value is okey, it is converted to state variable and pushed to particle.
- The particle position is set to the node positions
"""
function NodeToParticle!(i::Int64, j::Int64, x::Float64, y::Float64, model::Abstract2DStochasticModel, wind_tuple::Tuple{Float64,Float64}, DT::Float64)
        S = model.State
        minimal_particle = model.minimal_particle
        minimal_state = model.minimal_state
        wind_min_squared = model.ODEsettings.wind_min_squared
        default_particle = model.ODEdefaults
        e_min_log = model.ODEsettings.log_energy_minimum
        current_is_boundary = i==0 || i==model.grid.stats.Nx || j==0 || j==model.grid.Ny

        # definition of a norm 2
        norm(vec) = sqrt(sum([vec[i]^2 for i in eachindex(vec)]))
        one_norm(vec) = sum([vec[i] for i in eachindex(vec)])


        # load data from shared array
        u_state = S[i, j, :]

        #last_t = ti 
        last_t = model.clock.time
        # minimal_state[1] is the minmal Energy  
        # minimal_state[2] is the minmal momentum squared  
        if ~current_is_boundary & (u_state[1] >= minimal_state[1]) & (speed_square(u_state[2], u_state[3]) >= minimal_state[2]) # all interior nodes: convert note to particle values and push to ODEIntegrator
                if model.angular_spreading_type == "geometrical"
                        nNewParticles = Int64((u_state[4] ÷ model.angular_spreading_thresh)*2+1)
                        if nNewParticles == 1
                                nNewParticles = 3
                        end
                        midParticleInd = Int64(nNewParticles ÷ 2 + 1)
                        ui = GetVariablesAtVertex(u_state, x, y)
                        energies = [exp(-(k-midParticleInd)^2/(2*π*midParticleInd)^2) for k in 1:nNewParticles]
                        energies = energies ./ sum(energies) .* exp(ui[1])

                        for k in 1:nNewParticles
                                delta_phi = (k-midParticleInd)/midParticleInd*u_state[4]
                                c_x = ui[2] * cos(delta_phi) - ui[3] * sin(delta_phi)
                                c_y = ui[2] * sin(delta_phi) + ui[3] * cos(delta_phi)
                                z_init = ParticleDefaults(log(energies[k]), c_x, c_y, x, y, u_state[4]/nNewParticles)
                                push!(model.ParticleCollection,InitParticleInstance(model.ODEsystem, z_init, model.ODEsettings, (i,j), false, true))
                        end
                elseif model.angular_spreading_type == "nonparametric"
                        nNewParticles = model.n_particles_launch
                        nPartToDrawFrom = length(model.ParticlesAtNode[i][j])
                        energiesNormed = [model.ParticlesAtNode[i][j][k][1]*exp(model.ParticlesAtNode[i][j][k][3].ODEIntegrator.u[1]) for k in 1:nPartToDrawFrom]
                        energiesNormed = energiesNormed / u_state[1]
                        a = Categorical(energiesNormed)
                        particlesDrawn = rand(a, nNewParticles)
                        new_energy_tot = sum([energiesNormed[k] for k in particlesDrawn])
                        #@info "u_state = ", u_state[1]
                        #@info "new_energy_tot = ", new_energy_tot
                        #@info particlesDrawn
                        energy_factor = u_state[1] / new_energy_tot
                        #@info "new val = ", sum([energiesNormed[particlesDrawn[k]]*energy_factor for k in 1:nNewParticles])
                        #@info model.ParticlesAtNode[i][j][particlesDrawn[1]][3].ODEIntegrator.u[4], model.ParticlesAtNode[i][j][particlesDrawn[1]][3].position_xy[1], model.ParticlesAtNode[i][j][particlesDrawn[1]][3].position_xy[1] + model.grid.dx
                        #@info model.ParticlesAtNode[i][j][particlesDrawn[1]][3].ODEIntegrator.u[5], model.ParticlesAtNode[i][j][particlesDrawn[1]][3].position_xy[2], model.ParticlesAtNode[i][j][particlesDrawn[1]][3].position_xy[2] + model.grid.dy
                        for k in 1:nNewParticles

                                log_energy = log(energiesNormed[particlesDrawn[k]]*energy_factor)
                                c_x = model.ParticlesAtNode[i][j][particlesDrawn[k]][3].ODEIntegrator.u[2]
                                c_y = model.ParticlesAtNode[i][j][particlesDrawn[k]][3].ODEIntegrator.u[3]
                                spreading = model.ParticlesAtNode[i][j][particlesDrawn[k]][3].ODEIntegrator.u[6]

                                z_init = ParticleDefaults(log_energy, c_x, c_y, x, y, spreading)
                                push!(model.ParticleCollection,InitParticleInstance(model.ODEsystem, z_init, model.ODEsettings, (i,j), false, true))
                        end
                end
        elseif ~PI.boundary & (speed_square(wind_tuple[1], wind_tuple[2]) >= wind_min_squared) #minimal windsea is not big enough but local winds are strong enough  #(u_state[1] < exp(e_min_log)) | PI.boundary
                # test if particle is below energy threshold, or
                #      if particle is at the boundary

                ui = ResetParticleValues(default_particle, PI, wind_tuple, DT) # returns winds sea given DT and winds
                reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)#, reinit_callbacks=true)
                reset_PI_t!(PI, ti=last_t)

                PI.on = true

        elseif PI.boundary & (speed_square(wind_tuple[1], wind_tuple[2]) >= wind_min_squared) # at the boundary, reset particle if winds are strong enough

                ui = ResetParticleValues(default_particle, PI, wind_tuple, DT) # returns winds sea given DT and winds
                reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)#, reinit_callbacks=true)
                reset_PI_t!(PI, ti=last_t)

                PI.on = true

 
        else # particle is below energy threshold & on boundary
                #PI.ODEIntegrator.u = ResetParticleValues(minimal_particle, PI, wind_tuple, DT)
                # if ~PI.boundary
                #         @info u_state
                # end
                PI.on = false
        end

end

"""
        remesh!(PI::ParticleInstance2D, S::SharedMatrix{Float64, 3})
        Wrapper function that does everything necessary to remesh the particles.
        - pushes the Node State to particle instance
"""
function remesh!(i::Int64, j::Int64, model::Abstract2DStochasticModel, DT::Float64)
        winds = model.winds
        ti = model.clock.time
        grid = model.grid
        
        x = grid.xmin + grid.dx*(i-1)
        y = grid.ymin + grid.dy*(j-1)
        winds_i::Tuple{Float64,Float64} = winds.u(x, y, ti), winds.v(x, y, ti)
        
        NodeToParticle!(i, j, x, y, model, winds_i, DT)
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

        last_t = PI.ODEIntegrator.t
        # minimal_state[1] is the minmal Energy  
        # minimal_state[2] is the minmal momentum squared  
        if ~PI.boundary & (u_state[1] >= minimal_state[1]) & (speed_square(u_state[2], u_state[3]) >= minimal_state[2]) # all integrior nodes: convert note to particle values and push to ODEIntegrator

                #@show "u_state", u_state
                ui = GetVariablesAtVertex(u_state, xy[1], xy[2])
                #@info exp(ui[1]), ui[2], ui[4]/1e3, ui[5]/1e3
                reset_PI_ut!(PI; ui=ui, ti=last_t, stochas=true)
                PI.on = true

                # this method is more robust than the set_u! method (~february 2023)
                # reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)#, reinit_callbacks=true)
                # set_t!(PI.ODEIntegrator, last_t )
                # u_modified!(PI.ODEIntegrator,true)
                # auto_dt_reset!(PI.ODEIntegrator)

                # # this method is more robust than the set_u! method
                # reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)#, reinit_callbacks=true)
                # #set_u!(PI.ODEIntegrator, ui )
                # #set_t!(PI.ODEIntegrator, last_t )
                # u_modified!(PI.ODEIntegrator,true)

                #@show PI.ODEIntegrator.t, PI.ODEIntegrator.u
                
        elseif ~PI.boundary & (speed_square(wind_tuple[1], wind_tuple[2]) >= wind_min_squared) #minimal windsea is not big enough but local winds are strong enough  #(u_state[1] < exp(e_min_log)) | PI.boundary
                # test if particle is below energy threshold, or
                #      if particle is at the boundary

                ui = ResetParticleValues(default_particle, xy, wind_tuple, DT) # returns winds sea given DT and winds
                reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)#, reinit_callbacks=true)
                reset_PI_t!(PI, ti=last_t)

                PI.on = true

        elseif PI.boundary & (speed_square(wind_tuple[1], wind_tuple[2]) >= wind_min_squared) # at the boundary, reset particle if winds are strong enough

                ui = ResetParticleValues(default_particle, xy, wind_tuple, DT) # returns winds sea given DT and winds
                reinit!(PI.ODEIntegrator, ui, erase_sol=false, reset_dt=true, reinit_cache=true)#, reinit_callbacks=true)
                reset_PI_t!(PI, ti=last_t)
                # @info default_particle, ui
                PI.on = true

 
        else # particle is below energy threshold & on boundary
                #PI.ODEIntegrator.u = ResetParticleValues(minimal_particle, xy, wind_tuple, DT)
                # if ~PI.boundary
                #         @info u_state
                # end
                PI.on = false
        end
        nothing

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
