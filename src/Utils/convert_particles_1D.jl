using HDF5
using JLD2
using DataFrames
using Printf

import Base: filter!
#using OrdinaryDiffEq
using Statistics

# %% IO - fine the right path and read input arguments

using InputOutput: Argsettings, parse_args

#https://argparsejl.readthedocs.io/en/latest/argparse.html
parset = "1D_static/"

arg_test = ["--ID", "dx1655_dt2400", "--parset", "1D_static"]
passed_argument = parse_args(arg_test, Argsettings)
@unpack ID, parset = passed_argument

#save_path = joinpath( "data/1D_static/", parsed_args["ID"] )
save_path = save_path_base*parset*"/"*ID*"/"

# %%



mutable struct ParticleInstance
        position_ij :: Tuple
        position_xy :: Tuple
        ODEIntegrator::OrdinaryDiffEq.ODEIntegrator
end


ParticleCollection = load_object( joinpath(save_path , "particles.jld2")  )


function filter!(mask::BitArray, x::AbstractVector)
        pos = findall(x -> x == 0, mask)
        [deleteat!(x, p) for p in pos]
end

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
# %%
DD = []
for a_particle in ParticleCollection

        D         = DataFrame(Tables.columntable(a_particle.ODEIntegrator.sol))
        D.mask    = CreateIterationMask(D[!, "timestamp"])
        push!(DD, D[!, [1, 2, 3, 4]] ) # time, x(t), c_x(t), lne(t)

end


# %%
@show "save data"
#using HDF5

mkpath(save_path)
rm(joinpath(save_path , "particle_tables.h5"), force=true)
file = h5open( joinpath(save_path , "particle_tables.h5") , "w" )

Np = size(DD)[1]
Ns, Nvar = size(DD[1])

store_waves      = create_group(file, "particles")
#store_waves_data = create_dataset(store_waves, "data", Float64, ( (Np), (Nvar) ) )#, chunk=(2,))

# determine maximu length
Ns_lengths =  []
for i in 1:length(DD)
        push!(Ns_lengths, size(Array(DD[i]))[1] )
end
Ns = maximum(Ns_lengths)

store_particle = create_dataset(store_waves, "data" , Float64, ( (Np), (Ns), (Nvar)  ) )#, chunk=(2,))
@show (Np), (Ns), (Nvar)

for i in 1:length(DD)
        Ns_i = size(DD[i])[1]
        if Ns_i == Ns
                store_particle[i, :,:] = Array(DD[i])
        elseif Ns_i < Ns
                @show Ns_i,  Ns
                store_particle[i, 1:Ns_i,:] = Array(DD[i])
        end
end

write_attribute(store_waves, "var_names", names(DD[1]) )
close(file)

# %%

# # version for separated arrays
# @show (Np), (Ns), (Nvar)
# for i in 1:length(DD)
#         Ns, Nvar = size(DD[i])
#         particle_name =  string(i, base = 5,pad= 5)
#         store_particle = create_dataset(store_waves,particle_name , Float64, ( (Ns),  (Nvar) ) )#, chunk=(2,))
#         @show  size(Array(DD[i]))
#         store_particle[:,:] = Array(DD[i])
# end
