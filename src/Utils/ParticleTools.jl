module ParticleTools

using DataFrames

using ...Architectures: AbstractGrid, AbstractODESettings, AbstractParticleInstance
using ...custom_structures: MarkedParticleInstance

using ...Operators.core_2D: GetParticleEnergyMomentum

using Plots

function CreateIterationMask(time)
        mask = BitArray([z < 0 for z in diff(time)])
        #breaks = findall(x->x==1, mask)
        lower_bound = insert!(findall(x->x==1, mask), 1, 0)
        lower_bound .+= 1
        upper_bound = insert!(findall(x->x==1, mask), length(findall(x->x==1, mask))+1 , length(time))
        time_mask = time * 0
        for (i1, i2, timem) in zip(lower_bound, upper_bound, range(1,length(upper_bound), step=1))
                #@show i1, i2, time[i1:i2], ParticleData[i1:i2, :]
                time_mask[i1:i2] .= timem
        end
        return time_mask
end

function ParticleToDataframe(a_particle::AbstractParticleInstance)
        return DataFrame(Tables.columntable(a_particle.ODEIntegrator.sol))
end

function ParticleToDataframe(a_particle::MarkedParticleInstance)
        return DataFrame(Tables.columntable(a_particle.Particle.ODEIntegrator.sol))
end

function ParticleToDataframe(Collection::Vector{AbstractParticleInstance})
        DD = []
        for a_particle in Collection
                D         = ParticleToDataframe(a_particle)
                D.mask    = CreateIterationMask(D[!, "timestamp"])
                push!(DD, D[!, [1, 2, 3, 4]] ) # time, x(t), c_x(t), lne(t)
        end
        return DD
end

function ParticleToDataframe(Collection)
        DD = []
        for a_particle in Collection
                D         = ParticleToDataframe(a_particle.Particle)
                D.mask    = CreateIterationMask(D[!, "timestamp"])
                push!(DD, D[!, [1, 2, 3, 4]] ) # time, x(t), c_x(t), lne(t)
        end
        return DD
end

function FormatParticleData(PI4)
        statelist = GetParticleEnergyMomentum.(PI4.ODEIntegrator.sol.u)

        statematix = hcat(statelist...)
        u_matrix = hcat(PI4.ODEIntegrator.sol.u...)

        if (length(PI4.ODEIntegrator.sol.u[1]) == 3)
                AA = (x=u_matrix[3, :],
                        time=PI4.ODEIntegrator.sol.t,
                        cgx=u_matrix[2, :],
                        E=statematix[1, :],
                        mx=statematix[2, :])
        else
                AA = (x=u_matrix[4, :],
                        y=u_matrix[5, :],
                        time=PI4.ODEIntegrator.sol.t,
                        cgx=u_matrix[2, :],
                        cgy=u_matrix[3, :],
                        E=statematix[1, :],
                        mx=statematix[2, :],
                        my=statematix[3, :])
        end
        return AA
end


function ParticleStatsToDataframe(Collection)
        df = DataFrame()
        df[!, "position_ij"] = [PF.Particle.position_ij for PF in Collection]
        df[!, "position_xy"] = [PF.Particle.position_xy for PF in Collection]
        df[!, "boundary"] = [PF.Particle.boundary for PF in Collection]
        df[!, "time"] = [PF.time for PF in Collection]
        df[!, "errorReturnCode"] = [PF.errorReturnCode for PF in Collection]
        return df
end

function ParticleStatsToDataframe_simple(Collection)
        df = DataFrame()
        df[!, "position_ij"] = [ PF.position_ij for PF in Collection][:]
        df[!, "position_xy"] = [PF.position_xy for PF in Collection][:]
        df[!, "boundary"] = [PF.boundary for PF in Collection][:]
        df[!, "time"] = [PF.ODEIntegrator.t for PF in Collection][:]
        df[!, "u"] = [PF.ODEIntegrator.u for PF in Collection][:]
        # df[!, "errorReturnCode"] = [PF.errorReturnCode for PF in Collection]
        return df
end


function PlotFailedParticles(Collection, title, DT, dx)
        energy_list = []
        cg_list = []
        time_list = []
        x_list = []

        time_fail_list =[]
        x_fail_list =[]
        for Fdi in Collection

                di = ParticleTools.ParticleToDataframe(Fdi)

                time_pos = di[:, 1]/DT
                lne_pos = di[:, 2]
                x_pos = di[:, 4]/dx
                #Energy = @. exp.(di[:, 4])
                cg = @. di[:, 3] + 0* time_pos
                Energy = @. di[:, 2] + 0* time_pos

                push!(cg_list, cg)
                push!(energy_list, Energy)
                push!(time_list, time_pos)
                push!(x_list, x_pos)

                try
                        push!(time_fail_list, Fdi.time / DT)
                catch
                        push!(time_fail_list, Fdi.ODEIntegrator.t / DT)
                end

                push!(x_fail_list, x_pos[1])

                #scatter!(x, lne, marker=2) |> display
                #scatter!(time, Energy, marker=2) #|> display
                #scat = scatter(time, [ Energy,  cg], layout= (2, 1), marker=2) #|> display
                #plot(alpha_range,  Δ_β.(alpha_range)  , label="Delta_beta")

        end

        p1 = plot(x_list, time_list, title  = "Time", ylabel= "time/DT" ) #|> display
        p2 = plot(x_list, energy_list, marker=2, title  = "Energy", ylabel="energy") #|> display
        p3 = plot(x_list, cg_list, marker=3, title  = "c_g", ylabel="group velocity") #|> display

        plot(p1, p2, p3,  layout= (3, 1) , legend= false, size= (600,1200) )
        scatter!( x_fail_list ,time_fail_list, color = "red",marker = 4) #

        plot!(xlabel ="position", plot_title=title)

end


"""
Plot Particles in FailedCollection
inputs:
FailedCollection: Array of FailedParticles
store_wave_data: Array with the stored wave state
ID: String with ID of simulation
DT: timestep of simulation
dx: grid spacing of simulation
Npar: number of particles to plot
savepath: path to save plot (default: false)
"""
function PlotFailedParticles_summary(FailedCollection, store_wave_data, ID,  DT, dx; savepath=false, Npar = 4)

        if Npar == nothing
                Npar = size(FailedCollection)[1]
        end
        #ParticleTools.PlotFailedParticles(FailedCollection, ID, DT, dx) |> display
        ParticleTools.PlotFailedParticles(FailedCollection[1:Npar], ID, DT, dx) |> display
        if savepath != false
                mkpath(plot_path)
                savefig( joinpath(plot_path , "failed_ov_" * ID * ".png" ) )
        end

        ParticleTools.plot_cg_version1(FailedCollection) |> display
        if savepath != false
                savefig( joinpath(plot_path , "cg_failed_" * ID * ".png" ) )
        end

        ParticleTools.plot_cg_version2(store_waves_data[:, :, :]) |> display
        if savepath != false
                savefig( joinpath(plot_path , "cg_failed2_" * ID * ".png" ) )
        end

end





function plot_cg_version1(Collection)
        plot()

        for Fdi in Collection
                di = ParticleTools.ParticleToDataframe(Fdi)
                #plot!( di[:,1]/DT, di[:,2]/dx, label = "particle trajectory")
                plot!(di[:,3])
                #scatter!([Fdi.time/DT],  [0] ,  marker =2,  color   ="red")
        end
        plot!(legend = :none, title="c_g of initally failed particles", ylabel = "cg m/s", xlabel="position in Particle array") |> display

end


function plot_cg_version2(State_collect::Vector{Any})
        plot()

        for i in 1:length(State_collect)

                statei = State_collect[i]
                energy  =  statei[:,1] #GP.sel(state= 0 ).T # energy
                m_x     =  statei[:,2] #GP.sel(state= 1 ).T # energy
                cg    = energy ./ m_x ./ 2 # c_g
                plot!(cg, color ="black")
        end

        plot!(legend = :none, title="c_g", ylabel = "cg m/s", xlabel="position index x") |> display
end


function plot_cg_version2(State_collect::Array{Float64, 3})
        plot()

        for i in 1:size(State_collect[:,:,:])[1]

                statei = State_collect[i,:,:]
                energy  =  statei[:,1] #GP.sel(state= 0 ).T # energy
                m_x     =  statei[:,2] #GP.sel(state= 1 ).T # energy
                cg    = energy ./ m_x ./ 2 # c_g
                plot!(cg, color ="black")
        end

        plot!(legend = :none, title="c_g", ylabel = "cg m/s", xlabel="position index x") |> display
end



end
