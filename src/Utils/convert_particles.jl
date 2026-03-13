using HDF5
using JLD2
using DataFrames

import Base: filter!
#using OrdinaryDiffEq
using Statistics


mutable struct ParticleInstance
        position_ij :: Tuple
        position_xy :: Tuple
        ODEIntegrator::OrdinaryDiffEq.ODEIntegrator

end

save_path = "data/first_try/"

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

DD = []
for a_particle in ParticleCollection

        D         = DataFrame(Tables.columntable(a_particle.ODEIntegrator.sol))
        D.mask    = CreateIterationMask(D[!, "timestamp"])
        push!(DD, D[!, [1, 9, 2, 3, 4, 5, 6, 7, 8]] )

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
store_waves_data = create_dataset(store_waves, "data", Float64, ( (Np), (Ns), (Nvar) ) )#, chunk=(2,))

for i in 1:length(DD)
        store_waves_data[i,:,:] = Array(DD[i])
end

write_attribute(store_waves, "var_names", names(DD[1]) )
close(file)
