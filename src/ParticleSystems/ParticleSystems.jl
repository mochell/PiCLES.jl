module ParticleSystems

export particle_waves_v3, particle_waves_v3beta, particle_waves_v4, particle_waves_v5

# include("particle_waves_v3beta.jl")
# include("particle_waves_v3.jl")
# include("particle_waves_v4.jl")
include("particle_waves_v5.jl")
include("particle_waves_fake.jl")

# using .particle_waves_v3beta
# using .particle_waves_v3
# using .particle_waves_v4
using .particle_waves_v5
using .particle_waves_fake

end