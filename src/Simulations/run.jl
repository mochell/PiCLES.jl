using ..Operators.core_2D_spread: SeedParticle as StochasticSeedParticle2D
using ..Operators.core_2D: SeedParticle as SeedParticle2D
using ..Operators.core_2D_spread: ParticleDefaults as ParticleDefaults2D
using ..Operators.core_1D: ParticleDefaults as ParticleDefaults1D

using ..Operators.core_1D: SeedParticle! as SeedParticle1D!
using ..Operators.core_2D_spread: SeedParticle! as SeedParticle2D!
# using ..Operators.core_2D: SeedParticle 

using ..Architectures: Abstract2DModel, Abstract1DModel, Abstract2DStochasticModel
using ..ParticleMesh: OneDGrid, OneDGridNotes, TwoDGrid, TwoDGridNotes

#using WaveGrowthModels: init_particles!
#using WaveGrowthModels2D: init_particles!
using ..Operators.TimeSteppers

using ..Operators.mapping_1D
using ..Operators.mapping_2D
using Statistics

using StructArrays

import Plots as plt

using DataFrames, CSV
using Random, Distributions
using Dates

#using ThreadsX

function mean_of_state(model::Abstract2DModel)
        return mean(model.State[:, :, 1])
end

function mean_of_state(model::Abstract1DModel)
        return mean(model.State[:, 1])
end

function plot_state_and_error_points(wave_simulation, gn)
        plt.plot()

        X = (0:(gn.stats.Nx.N-1)).*gn.stats.dx .+ gn.stats.xmin
        Y = (0:(gn.stats.Ny.N-1)).*gn.stats.dy .+ gn.stats.ymin
        energy = get_tot_energy_domain(wave_simulation)
        p1 = plt.heatmap(X, Y, transpose(wave_simulation.model.State[:, :, 1]), aspect_ratio=:equal, size=(1080, 1080))#,clim=(0,1))

        plt.plot!(legend=:none,
                title="total energy = "*string(round(energy,digits=3))*"; max = "*string(round(maximum(wave_simulation.model.State[:,:,1]),
                        digits=3))*"; pos = ("*string(argmax(wave_simulation.model.State[:,:,1])[1])*","*
                        string(argmax(wave_simulation.model.State[:,:,1])[2])*")",
                ylabel="y position",
                xlabel="x position",
                xlims=(gn.stats.xmin, gn.stats.xmax),
                ylims=(gn.stats.ymin, gn.stats.ymax)) |> display
end

function write_particles_to_csv(wave_model)
        sec=string(Int64(floor((wave_model.clock.time)/60)))
        dec=string(Int64(floor(10*(wave_model.clock.time/60-floor((wave_model.clock.time)/60)))))
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

function get_tot_energy_domain(wave_simulation)
        return sum(wave_simulation.model.State[:,:,1])
end

"""
run!(sim::Simulation; store = false, pickup=false)
main method to run the Simulation sim.
Needs time_step! to be defined for the model, and push_state_to_storage! to be defined for the store.
"""
function run!(sim; store=false, pickup=false, cash_store=false, debug=false)
        if sim.model isa Abstract2DStochasticModel
                save_path = sim.model.plot_savepath

                if sim.model.save_particles
                save_path = sim.model.plot_savepath
                filename = save_path*"/data/simu_info.csv"

                Nx = sim.model.grid.stats.Nx.N
                Ny = sim.model.grid.stats.Ny.N
                xmin = sim.model.grid.stats.xmin
                xmax = sim.model.grid.stats.xmax
                ymin = sim.model.grid.stats.ymin
                ymax = sim.model.grid.stats.ymax
                lne_source = sim.model.ODEdefaults.lne
                c_x_source = sim.model.ODEdefaults.c̄_x
                c_y_source = sim.model.ODEdefaults.c̄_y
                x_source = sim.model.ODEdefaults.x
                y_source = sim.model.ODEdefaults.y
                angular_spread_source = sim.model.ODEdefaults.angular_σ
                Δt = sim.Δt
                stop_time = sim.stop_time

                data = DataFrame(Nx=Nx, Ny=Ny, xmin=xmin,
                                xmax=xmax, ymin=ymin, ymax=ymax,
                                lne_source=lne_source, c_x_source=c_x_source,
                                c_y_source=c_y_source, x_source=x_source,y_source=y_source,
                                angular_spread_source=angular_spread_source,
                                Δt=Δt, stop_time=stop_time)
                CSV.write(filename, data)

                filename2 = save_path*"/data/sigma.csv"

                covariance_init = sim.model.proba_covariance_init
                data2 = DataFrame(covariance_init, :auto)
                CSV.write(filename2, data2)
                end
        end

        start_time_step = time_ns()

        if !(sim.initialized) # execute initialization step
                initialize_simulation!(sim)
        end

        if sim.model isa Abstract2DStochasticModel
                if sim.model.save_particles && length(sim.model.ParticleCollection) > 0
                        write_particles_to_csv(sim.model)
                end
        end

        #sim.running = true
        sim.run_wall_time = 0.0

        if sim.stop_time >= sim.model.clock.time
                sim.running = true
        else
                sim.running = false
                @info "stop_time exceeded, run not executed"
        end

        if cash_store
                sim.store = CashStore([], 1)
                sim.store.iteration += 1
                push!(sim.store.store, copy(sim.model.State))
                if sim.verbose
                        @info "write inital state to cash store..."
                end
        end

        if store
                push_state_to_storage!(sim)
                sim.store.iteration += 1
                if sim.verbose
                        @info "write inital state to store..."
                end
        end


        while sim.running

                #reset State
                if isa(sim.model.State, SharedArray)
                        sim.model.State[:,:,:] .= 0.0
                else
                        sim.model.State .= 0.0
                end
                # do time step

                time_step!(sim.model, sim.Δt, debug=debug)

                if debug & (length(sim.model.FailedCollection) > 0)
                        @info "debug mode:"
                        @info "found failed particles"
                        @info "failed particles: ", length(sim.model.FailedCollection)
                        @info "break"
                        #sim.running = false
                        # break while loop
                        break
                end

                if store
                        push_state_to_storage!(sim)
                        sim.store.iteration += 1
                        if sim.verbose
                                @info "write state to store..."
                                #@info "max energy ", maximum(sim.model.State[:,:,1])
                        end

                end

                if cash_store
                        push!(sim.store.store, copy(sim.model.State))
                        sim.store.iteration += 1
                        if sim.verbose
                                @info "write state to cash store..."
                                #print("mean energy ", mean_of_state(sim.model), "\n")
                        end

                end
                sim.running = sim.stop_time >= sim.model.clock.time ? true : false

                if sim.model isa Abstract2DStochasticModel
                        if sim.model.plot_steps
                                plot_state_and_error_points(sim, sim.model.grid)
                                sec=string(Int64(floor((sim.model.clock.time)/60)))
                                dec=string(Int64(floor(10*(sim.model.clock.time/60-floor((sim.model.clock.time)/60)))))
                                plt.savefig(joinpath([save_path*"/plots/", "energy_plot_no_spread_"*sec*","*dec*".png"]))
                        end
                end
        end

        end_time_step = time_ns()

        # Increment the wall clock
        sim.run_wall_time += 1e-9 * (end_time_step - start_time_step)

end


"""
initialize_simulation!(sim::Simulation)
initialize the simulation sim by calling init_particles! to initialize the model.ParticleCollection.
-particle_initials::T=nothing  was removed from arguments
"""
function initialize_simulation!(sim::Simulation)# where {PP<:Union{ParticleDefaults2D,Nothing}}
        # copy(ParticleDefaults2D(log(4e-8), 1e-2, 0.0)))

        if sim.verbose
                @info "init particles..."
        end
        init_particles!(sim.model, defaults=sim.model.ODEdefaults, verbose=sim.verbose)
        
        if sim.model.clock.iteration != 0
                sim.model.clock.iteration = 0
                sim.model.clock.time = 0
        end
        
        sim.initialized = true

        nothing
end


"""
reset_simulation!(sim::Simulation)
reset the simulation sim by calling init_particles! to reinitialize the model.ParticleCollection, sets the model.clock.time, model.clock.iteration, and model.state to 0.
- particle_initials::Dict{Num, Float64} was removed from arguments
"""
function reset_simulation!(sim::Simulation)# where {PP<:Union{ParticleDefaults2D,Nothing}}

        sim.running = false
        sim.run_wall_time = 0.0

        sim.model.clock.iteration = 0
        sim.model.clock.time = 0

        # particles
        if sim.verbose
                @info "reset time..."
                @info "re-init particles..."
        end
        init_particles!(sim.model, defaults=sim.model.ODEdefaults, verbose=sim.verbose)

        # state
        if sim.verbose
                @info "clear state..."
        end
        sim.model.State .= 0

        sim.initialized = true

        if sim.store isa StateStore
                reset_state_store!(sim)
        end
        nothing
end


# depreciate, just used for 1D version
"""
SeedParticle_mapper(f, p, s, b1, b2, b3, c1, c2, c3, c4, d1, d2 ) = x -> f( p, s, x, b1, b2, b3, c1, c2, c3, c4, d1, d2 )
maps to SeedParticle! function
"""
SeedParticle_mapper(f, p, s, b1, b2, b3, c1, c2, c3, d1, d2) = x -> f(p, s, x, b1, b2, b3, c1, c2, c3, d1, d2)


"""
init_particle!(model ; defaults::PP, verbose::Bool=false )

initialize the model.ParticleCollection based on the model.grid and the defaults. 
If defaults is nothing, then the model.ODEdev is used.
usually the initilization uses wind constitions to seed the particles.
"""
function init_particles!(model::Abstract2DStochasticModel; defaults::PP=nothing, verbose::Bool=false) where {PP<:Union{ParticleDefaults1D,ParticleDefaults2D,Array{Any,1},Nothing}}
        #defaults        = isnothing(defaults) ? model.ODEdev : defaults
        if verbose
                @info "seed PiCLES ... \n"
                @info "defaults is $(defaults)"
                if defaults isa Dict
                        @info "found particle initials, just replace position "
                else
                        @info "no particle defaults found, use windsea to seed particles"
                end
        end

        # gridnotes = TwoDGridNotes(model.grid)

        # SeedParticle_i = SeedParticle_mapper(SeedParticle2D!,
        #         ParticleCollection, model.State,
        #         model.ODEsystem, nothing, model.ODEsettings,
        #         gridnotes, model.winds, model.ODEsettings.timestep,
        #         model.boundary, model.periodic_boundary)

        ParticleCollection = []
        model.ParticleCollection = ParticleCollection

        if defaults isa ParticleDefaults2D
                i = Int64(floor((defaults.x - model.grid.stats.xmin) / model.grid.stats.dx)) + 1
                j = Int64(floor((defaults.y - model.grid.stats.ymin) / model.grid.stats.dy)) + 1
                # gridnotes = TwoDGridNotes(model.grid)
                if model.angular_spreading_type == "nonparametric"
                        if sum(model.proba_covariance_init)==4e-50
                                # if "nonparametric" is used, initialize a bigger number of particles at the original perturbation
                                n_part = model.n_particles_launch
                                ij = CartesianIndices((i,j))
                                ij_mesh = model.grid.data[ij]
                                ij_wind = (model.winds.u(ij_mesh.x, ij_mesh.y, 0.0),
                                                model.winds.v(ij_mesh.x, ij_mesh.y, 0.0)
                                                )
                                for _ in 1:n_part
                                        defaults_temp = deepcopy(defaults)
                                        delta_phi = rand() * defaults_temp.angular_σ - 0.5*defaults_temp.angular_σ
                                        c_x = defaults_temp.c̄_x * cos(delta_phi) - defaults_temp.c̄_y * sin(delta_phi)
                                        c_y = defaults_temp.c̄_x * sin(delta_phi) + defaults_temp.c̄_y * cos(delta_phi)
                                        defaults_temp.lne += -log(n_part)
                                        defaults_temp.c̄_x = c_x
                                        defaults_temp.c̄_y = c_y
                                        push!(ParticleCollection, StochasticSeedParticle2D(model.State,
                                                        (i,j), model.ODEsystem, defaults_temp,
                                                        model.ODEsettings,
                                                        model.grid.stats,model.grid.ProjetionKernel,model.grid.PropagationCorrection,
                                                        ij_mesh[i,j], ij_wind, model.ODEsettings.timestep, model.boundary,
                                                        model.periodic_boundary))
                                end
                        else
                                n_part = model.n_particles_launch
                                for _ in 1:n_part
                                        defaults_temp = deepcopy(defaults)
                                        mu = [defaults_temp.c̄_x, defaults_temp.c̄_y, defaults_temp.x, defaults_temp.y]
                                        d = MvNormal(mu, model.proba_covariance_init)
                                        real = rand(d,1)
                                        # delta_phi = real[1]
                                        # delta_phi < 0 ? delta_phi = - delta_phi : delta_phi = delta_phi
                                        # delta_phi < 0.01 ? delta_phi+=1 : delta_phi +=0
                                        # c_x = (real[2]+1)*(defaults_temp.c̄_x * cos(delta_phi) - defaults_temp.c̄_y * sin(delta_phi))
                                        # c_y = (real[2]+1)*(defaults_temp.c̄_x * sin(delta_phi) + defaults_temp.c̄_y * cos(delta_phi))
                                        c_x = real[1]
                                        c_y = real[2]
                                        defaults_temp.lne += -log(n_part)
                                        defaults_temp.c̄_x = c_x
                                        defaults_temp.c̄_y = c_y
                                        defaults_temp.x = real[3]
                                        defaults_temp.y = real[4]
                                        push!(ParticleCollection, SeedParticle(model.State,
                                                        (i,j), model.ODEsystem, defaults_temp,
                                                        model.ODEsettings,gridnotes, model.winds,
                                                        model.ODEsettings.timestep, model.boundary,
                                                        model.periodic_boundary))
                                end
                        end
                else
                        push!(ParticleCollection, SeedParticle(model.State,
                                                (i,j), model.ODEsystem, defaults,
                                                model.ODEsettings,gridnotes, model.winds,
                                                model.ODEsettings.timestep, model.boundary,
                                                model.periodic_boundary))
                end
        elseif defaults isa Array{Any,1}
                for k in 1:length(defaults)
                        i = Int64(floor((defaults[k].x - model.grid.stats.xmin) / model.grid.stats.dx)) + 1
                        j = Int64(floor((defaults[k].y - model.grid.stats.ymin) / model.grid.stats.dy)) + 1
                        gridnotes = OneDGridNotes(model.grid)
                        push!(ParticleCollection, SeedParticle(model.State,
                                                (i,j), model.ODEsystem, defaults[k],
                                                model.ODEsettings,gridnotes, model.winds,
                                                model.ODEsettings.timestep, model.boundary,
                                                model.periodic_boundary))
                end
        end
        nothing
end

function init_particles!(model::Abstract2DModel; defaults::PP=nothing, verbose::Bool=false) where {PP<:Union{ParticleDefaults1D,ParticleDefaults2D,Nothing}}
        #defaults        = isnothing(defaults) ? model.ODEdev : defaults
        if verbose
                @info "seed PiCLES ... \n"
                @info "defaults is $(defaults)"
                if defaults isa Dict
                        @info "found particle initials, just replace position "
                else
                        @info "no particle defaults found, use windsea to seed particles"
                end
        end

        ParticleCollection = StructArray(map(ij -> begin

                        ij_mesh = model.grid.data[ij]
                        ij_wind = (     model.winds.u(ij_mesh.x, ij_mesh.y, 0.0), 
                                        model.winds.v(ij_mesh.x, ij_mesh.y, 0.0)
                                        )

                        SeedParticle2D(
                                model.State, ij,
                                model.ODEsystem, defaults, model.ODEsettings,
                                model.grid.stats, model.grid.ProjetionKernel, model.grid.PropagationCorrection,
                                ij_mesh, ij_wind,
                                model.ODEsettings.timestep,
                                model.boundary, model.periodic_boundary)

                end, CartesianIndices(model.grid.data)))


        # threads for loop version
        # ParticleCollection = StructArray{ParticleInstance2D}(undef, grid.stats.Nx, grid.stats.Ny)

        # speed tests
        # 1 thread  8.736 ms (124253 allocations: 12.39 MiB)
        # 4 thread   4.443 ms (123316 allocations: 12.35 MiB)
        # @btime @threads for ij in CartesianIndices(mesh)
        #         ParticleCollection4[ij] = SeedParticle(
                                # model.State, ij,
                                # model.ODEsystem, defaults, model.ODEsettings,
                                # model.grid.stats, ij_mesh, ij_wind,
                                # model.DT,
                                # model.boundary, model.periodic_boundary)
        # end
        @info typeof(ParticleCollection)
        # @info ParticleCollection
        model.ParticleCollection = ParticleCollection
        nothing
end



### 1D version ###
# """
# SeedParticle_mapper(f, p, s, b1, b2, b3, c1, c2, c3, c4, d1, d2 ) = x -> f( p, s, x, b1, b2, b3, c1, c2, c3, c4, d1, d2 )
# maps to SeedParticle! function
# """
# SeedParticle_mapper(f, p, s, b1, b2, b3, c1, c2, c3, d1, d2 )  = x -> f( p, s, x, b1, b2, b3, c1, c2, c3, d1, d2 )


"""
init_particle!(model ; defaults::PP, verbose::Bool=false )

initialize the model.ParticleCollection based on the model.grid and the defaults. 
If defaults is nothing, then the model.ODEdev is used.
usually the initilization uses wind constitions to seed the particles.
"""
function init_particles!(model::Abstract1DModel; defaults::PP=nothing, verbose::Bool=false) where {PP<:Union{ParticleDefaults1D,ParticleDefaults2D,Nothing}}
        #defaults        = isnothing(defaults) ? model.ODEdev : defaults
        if verbose
                @info "seed PiCLES ... \n"
                @info "defaults is $(defaults)"
                if defaults isa Dict
                        @info "found particle initials, just replace position "
                else
                        @info "no particle defaults found, use windsea to seed particles"
                end
        end

        gridnotes = OneDGridNotes(model.grid)

        ParticleCollection = []
        SeedParticle_i = SeedParticle_mapper(SeedParticle1D!, 
                ParticleCollection, model.State,
                model.ODEsystem, defaults, model.ODEsettings,
                gridnotes, model.winds, model.ODEsettings.timestep,
                model.boundary, model.periodic_boundary)

        map(SeedParticle_i, range(1, length=model.grid.stats.Nx))


        # print(defaults)
        # ParticleCollection=[]
        # for i in range(1, length=gridnotes.Nx)
        #         SeedParticle!(ParticleCollection, model.State, i,
        #                         model.ODEsystem, defaults , model.ODEsettings,
        #                         gridnotes, model.winds, model.ODEsettings.timestep,
        #                         model.boundary, model.periodic_boundary  )
        # end

        model.ParticleCollection = ParticleCollection
        nothing
end


