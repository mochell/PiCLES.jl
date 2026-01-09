module TimeSteppers

export time_step!
using ...Architectures
using ..mapping_1D
using ..mapping_2D

# for debugging
using Statistics
using Base.Threads
using Printf

using Random, Distributions
using ..core_2D_spread: ParticleDefaults, InitParticleInstance
using ...custom_structures: ParticleInstance2D

using Oceananigans.TimeSteppers: tick!

using DataFrames, CSV, Dates

function mean_of_state(model::Abstract2DModel)
    return mean(model.State[:, :, 1])
end

function max_energy(model::Abstract2DModel)
    return maximum(model.State[:, :, 1])
end

function max_cgx(model::Abstract2DModel)
    return maximum(model.State[:, :, 2])
end

function max_cgy(model::Abstract2DModel)
    return maximum(model.State[:, :, 3])
end


function mean_of_state(model::Abstract1DModel)
    return mean(model.State[:, 1])
end


################# 1D ####################

function write_particles_to_csv(wave_model, Δt)
    sec=string(Int64(floor((wave_model.clock.time+Δt)/60)))
    dec=string(Int64(floor(10*((wave_model.clock.time+Δt)/60-floor((wave_model.clock.time+Δt)/60)))))
    save_path = wave_model.plot_savepath

    nParticles = wave_model.n_particles_launch
    @info wave_model.clock.time/60

    parts = wave_model.ParticleCollection[(end-nParticles+1):end]

    logE = zeros(nParticles)
    cx = zeros(nParticles)
    cy = zeros(nParticles)
    x = zeros(nParticles)
    y = zeros(nParticles)
    for i in 1:nParticles
            logE[i] = parts[i].ODEIntegrator[1]
            cx[i] = parts[i].ODEIntegrator[2]
            cy[i] = parts[i].ODEIntegrator[3]
            x[i] = parts[i].ODEIntegrator[4]
            y[i] = parts[i].ODEIntegrator[5]
    end

    data = DataFrame(id=1:nParticles, logE = logE, cx = cx, cy = cy, x = x, y = y)
    data2 = Tables.table(transpose(wave_model.State[:, :, 1]))
    CSV.write(save_path*"/data/particles_"*sec*","*dec*".csv", data)
    CSV.write(save_path*"/data/mesh_values_"*sec*","*dec*".csv", data2)
end

"""
time_step!(model, Δt; callbacks=nothing)

advances model by 1 time step:
1st) the model.ParticleCollection is advanced and then 
2nd) the model.State is updated.
clock is ticked by Δt

callbacks are not implimented yet

"""
function time_step!(model::Abstract1DModel, Δt; callbacks=nothing, debug=false)

    # temporary FailedCollection to store failed particles
    FailedCollection = Vector{AbstractMarkedParticleInstance}([])

    for a_particle in model.ParticleCollection
            #@show a_particle.position_ij
            mapping_1D.advance!(    a_particle, model.State, FailedCollection, 
                                    model.grid, model.winds , Δt , 
                                    model.ODEsettings.log_energy_maximum, 
                                    model.ODEsettings.wind_min_squared,
                                    model.periodic_boundary,
                                    model.ODEdefaults)
    end
    if debug
            model.FailedCollection = FailedCollection
            @info "advanced: "
            #@info model.State[8:12, 1], model.State[8:12, 2]
            @info model.clock.time, model.ParticleCollection[10].ODEIntegrator.t
            @info model.winds(model.ParticleCollection[10].ODEIntegrator.u[3], model.ParticleCollection[10].ODEIntegrator.t)

    end

    #@printf "re-mesh"
    for a_particle in model.ParticleCollection
            mapping_1D.remesh!(     a_particle, model.State, 
                                    model.winds, model.clock.time, 
                                    model.ODEsettings, Δt,
                                    model.minimal_particle,
                                    model.minimal_state,
                                    model.ODEdefaults)
    end

    if debug
            @info "remeshed: "
            #@info model.State[8:12, 1], model.State[8:12, 2]
            @info model.clock.time, model.ParticleCollection[10].ODEIntegrator.t

    end

    tick!(model.clock, Δt)
end



################# 2D ####################

"""
time_step!(model, Δt; callbacks=nothing)

advances model by 1 time step:
1st) the model.ParticleCollection is advanced and then 
2nd) the model.State is updated.
clock is ticked by Δt

callbacks are not implimented yet

"""
function time_step!(model::Abstract2DModel, Δt::Float64; callbacks=nothing, debug=false)

    # temporary FailedCollection to store failed particles
    FailedCollection = Vector{AbstractMarkedParticleInstance}([])

    #print("mean energy before advance ", mean_of_state(model), "\n")
    if debug
        @info "before advance"
        @info maximum(model.State[:, :, 1]), maximum(model.State[:, :, 2]), maximum(model.State[:, :, 3])
        model.FailedCollection = FailedCollection
    end 

    for (i,j) in [(i,j) for i in 1:model.grid.Nx for j in 1:model.grid.Ny]
        for k in 1:length(model.ParticlesAtNode[i][j])
            pop!(model.ParticlesAtNode[i][j])
        end
    end

    @threads for a_particle in model.ParticleCollection
        #@info a_particle.position_ij
        mapping_2D.advance!(    a_particle, model.ParticlesAtNode, model.State, FailedCollection,
                                model.grid, model.winds, Δt,
                                model.ODEsettings.log_energy_maximum, 
                                model.ODEsettings.wind_min_squared,
                                model.periodic_boundary,
                                model.ODEdefaults)
    end

    if model.save_particles && length(model.ParticleCollection) > 0
        write_particles_to_csv(model, Δt)
    end

    for (i,j) in [(i,j) for i in 1:model.grid.Nx for j in 1:model.grid.Ny]
        weights = [model.ParticlesAtNode[i][j][k][1] for k in 1:length(model.ParticlesAtNode[i][j])]
        values = [model.ParticlesAtNode[i][j][k][2] for k in 1:length(model.ParticlesAtNode[i][j])]
        if length(weights) > 0
            model.State[i,j,4] = sum(weights .* values) / sum(weights)
        end
    end    

    if debug
        print("mean energy after advance ", mean_of_state(model), "\n")

        @info "advanced: "
        @info maximum(model.State[:, :, 1]), maximum(model.State[:, :, 2]), maximum(model.State[:, :, 3])
        #@info model.State[8:12, 1], model.State[8:12, 2]
        @info model.clock.time, model.ParticleCollection[10].ODEIntegrator.t
        @info "winds:", model.winds.u(model.ParticleCollection[10].ODEIntegrator.u[4], model.ParticleCollection[10].ODEIntegrator.u[5], model.ParticleCollection[10].ODEIntegrator.t)
    end

    #@printf "re-mesh"
    if model.angular_spreading_type == "stochast"
        @threads for a_particle in model.ParticleCollection
            mapping_2D.remesh!(a_particle, model.State, 
                            model.winds, model.clock.time, 
                            model.ODEsettings, Δt,
                            model.minimal_particle, 
                            model.minimal_state,
                            model.ODEdefaults)
        end
    elseif model.angular_spreading_type == "geometrical"
        i = 1
        particlesToBeReset = []
        
        for _ in 1:length(model.ParticleCollection)
            pos_ij = model.ParticleCollection[i].position_ij
            big_enough = model.State[pos_ij[1], pos_ij[2],2]^2+model.State[pos_ij[1], pos_ij[2],3]^2>model.minimal_state[2]
            if model.ParticleCollection[i].boundary || ~big_enough
                i=i+1
            elseif !(model.ParticleCollection[i].position_ij in particlesToBeReset)
                push!(particlesToBeReset, model.ParticleCollection[i].position_ij)
                deleteat!(model.ParticleCollection, i)
            end
        end
        @threads for a_particle in model.ParticleCollection
            mapping_2D.remesh!(a_particle, model.State, 
                            model.winds, model.clock.time, 
                            model.ODEsettings, Δt,
                            model.minimal_particle, 
                            model.minimal_state,
                            model.ODEdefaults)
        end
        @threads for (i,j) in particlesToBeReset
            mapping_2D.remesh!(i, j, model, Δt)
        end
        
    elseif model.angular_spreading_type == "nonparametric"

        # This next part needs to change

        nPreviousParticles = 0
        model.ParticlePool = []

        @threads for (i,j) in [(i,j) for i in 1:model.grid.Nx for j in 1:model.grid.Ny]
            locallen = length(model.ParticlesAtNode[i][j])
            nPreviousParticles += locallen
            @threads for k in 1:locallen

                Upart=model.ParticlesAtNode[i][j][k][3].ODEIntegrator.u
                # newodeint=init(model.ParticlesAtNode[i][j][k][3].ODEIntegrator.t,[deepcopy(model.ParticlesAtNode[i][j][k][3].ODEIntegrator.u[1]), deepcopy(model.ParticlesAtNode[i][j][k][3].ODEIntegrator.u[2]), deepcopy(model.ParticlesAtNode[i][j][k][3].ODEIntegrator.u[3]), deepcopy(model.ParticlesAtNode[i][j][k][3].ODEIntegrator.u[4]), deepcopy(model.ParticlesAtNode[i][j][k][3].ODEIntegrator.u[5]), deepcopy(model.ParticlesAtNode[i][j][k][3].ODEIntegrator.u[6])])
                # println(newodeint)
                # println()

                z_init = ParticleDefaults(Upart[1],Upart[2],Upart[3],Upart[4],Upart[5],Upart[6])
                newpart=InitParticleInstance(model.ODEsystem,z_init,model.ODEsettings,(i,j),false,true)
                push!(model.ParticlePool, [newpart, i, j, exp(model.ParticlesAtNode[i][j][k][3].ODEIntegrator.u[1])*model.ParticlesAtNode[i][j][k][1]])

                #newpart2=[deepcopy(model.ParticlesAtNode[i][j][k][3]), i, j, exp(model.ParticlesAtNode[i][j][k][3].ODEIntegrator.u[1])*model.ParticlesAtNode[i][j][k][1]]
                #push!(model.ParticlePool, newpart2)
            end
        end

        ParticuleEnergiesNorm = sum([model.ParticlePool[k][4] for k in 1:length(model.ParticlePool)])
        for k in 1:length(model.ParticlePool)
            model.ParticlePool[k][4] = model.ParticlePool[k][4] / ParticuleEnergiesNorm
        end

        energiesNormed = [model.ParticlePool[k][4] for k in 1:length(model.ParticlePool)]
        a = Categorical(energiesNormed)
        nNewParticles = model.n_particles_launch
        particlesDrawn = rand(a, nNewParticles)
        #new_energy_tot = sum([energiesNormed[k] for k in particlesDrawn])
        #energy_factor = totEnergyDomain[1] / new_energy_tot

        i = 1
        j = 0
        counter = 0
        for _ in 1:length(model.ParticleCollection)
            if model.ParticleCollection[i].on
                if counter >= nNewParticles
                    deleteat!(model.ParticleCollection, i)
                    j=j+1
                else
                    i=i+1
                end
                counter +=1
            else
                i=i+1
            end
        end
        #@info "removed particles :", j
        #@info "new energy :", ParticuleEnergiesNorm
        
        #@info "particlesDrawn", model.ParticlePool[particlesDrawn[k]][1].ODEIntegrator.u[2]

        for k in 1:nNewParticles
            i = model.ParticlePool[particlesDrawn[k]][2]
            j = model.ParticlePool[particlesDrawn[k]][3]
            x = model.grid.xmin + model.grid.dx*(i-1)
            y = model.grid.ymin + model.grid.dy*(j-1)
            #log_energy = log(totEnergyDomain[1]/nPreviousParticles)
            log_energy = log(ParticuleEnergiesNorm/nNewParticles)
            c_x = model.ParticlePool[particlesDrawn[k]][1].ODEIntegrator.u[2]
            c_y = model.ParticlePool[particlesDrawn[k]][1].ODEIntegrator.u[3]
            spreading = model.ParticlePool[particlesDrawn[k]][1].ODEIntegrator.u[6]

            #z_init = ParticleDefaults(log_energy, c_x, c_y, x, y, spreading)
            model.ParticleCollection[end-k+1].position_ij = (i,j)
            model.ParticleCollection[end-k+1].position_xy = (x,y)
            model.ParticleCollection[end-k+1].ODEIntegrator.u[1] = log_energy
            model.ParticleCollection[end-k+1].ODEIntegrator.u[2] = c_x
            model.ParticleCollection[end-k+1].ODEIntegrator.u[3] = c_y
            model.ParticleCollection[end-k+1].ODEIntegrator.u[4] = x
            model.ParticleCollection[end-k+1].ODEIntegrator.u[5] = y
            model.ParticleCollection[end-k+1].ODEIntegrator.u[6] = spreading
            #push!(model.ParticleCollection,InitParticleInstance(model.ODEsystem, z_init, model.ODEsettings, (i,j), false, true))
        end

        #@info nPreviousParticles
        #@info length(model.ParticleCollection)

        """
        @threads for k in eachindex(particlesToBeResetIndex)
            i, j = particlesToBeResetIndex[k]
            mapping_2D.remesh!(i, j, model, Δt)
        end
        """
    end

    if debug
        @info "remeshed: "
        #@info model.State[8:12, 1], model.State[8:12, 2]
        @info maximum(model.State[:, :, 1]), maximum(model.State[:, :, 2]), maximum(model.State[:, :, 3])
        @info model.clock.time, model.ParticleCollection[10].ODEIntegrator.t

    end
    #print("mean energy after remesh ", mean_of_state(model), "\n")

    # @printf("------- max state E=%.4e cgx=%.4e cgy=%.4e \n", max_energy(model), max_cgx(model), max_cgy(model))
    tick!(model.clock, Δt)


end

function time_step!_advance(model::Abstract2DModel, Δt::Float64, FailedCollection::Vector{AbstractMarkedParticleInstance})

    @threads for a_particle in model.ParticleCollection[model.ocean_points]
        #@info a_particle.position_ij
        mapping_2D.advance!(    a_particle, model.State, FailedCollection,
                                model.grid, model.winds, Δt,
                                model.ODEsettings.log_energy_maximum,
                                model.ODEsettings.wind_min_squared,
                                model.periodic_boundary,
                                model.ODEdefaults)
    end

end

function time_step!_remesh(model::Abstract2DModel, Δt::Float64)

    @threads for a_particle in model.ParticleCollection[model.ocean_points]
        mapping_2D.remesh!(a_particle, model.State, 
                        model.winds, model.clock.time, 
                        model.ODEsettings, Δt,
                        model.grid.stats, 
                        model.minimal_state,
                        model.ODEdefaults)
    end

end

#build wrapper
advance_wrapper(f, state, Fcol, grid, winds, dt, emax, windmin, boundary, defaults) = x -> f(x, state, Fcol, grid, winds, dt, emax, windmin, boundary, defaults)
remesh_wrapper(f, state, winds, time, sets, dt, minpar, minstate, defaults) = x -> f(x, state, winds, time, sets, dt, minpar, minstate, defaults)



"""
movie_time_step!(model, Δt; callbacks=nothing)

advances model by 1 time step:
1st) the model.ParticleCollection is advanced and then 
2nd) the model.State is updated.
clock is ticked by Δt

callbacks are not implimented yet

"""
function movie_time_step!(model::Abstract2DModel, Δt; callbacks=nothing, debug=false)

    # temporary FailedCollection to store failed particles
    FailedCollection = Vector{AbstractMarkedParticleInstance}([])

    for a_particle in model.ParticleCollection[model.ocean_points]
        #@show a_particle.position_ij
        mapping_2D.advance!(a_particle, model.State, FailedCollection,
            model.grid, model.winds, Δt,
            model.ODEsettings.log_energy_maximum,
            model.ODEsettings.wind_min_squared,
            model.periodic_boundary,
            model.ODEdefaults)

    end

    model.MovieState = copy(model.State)

    if debug
        model.FailedCollection = FailedCollection
    end

    #@printf "re-mesh"
    for a_particle in model.ParticleCollection[model.ocean_points]
        mapping_2D.remesh!(a_particle, model.State,
            model.winds, model.clock.time,
            model.ODEsettings, Δt,
            model.grid.stats,
            model.minimal_state,
            model.ODEdefaults)
    end

    
    model.State .= 0.0
    tick!(model.clock, Δt)
end


end
