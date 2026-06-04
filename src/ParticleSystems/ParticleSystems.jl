module ParticleSystems

# `particle_waves_v6` is the canonical, maintained particle system. Its public API is
# re-exported here under version-agnostic names so that downstream code can simply do
#     using PiCLES.ParticleSystems
#     sys = particle_equations(...); pars, ID, Scg = ODEParameters(...)
# without ever naming a version. Specific systems remain available by their module name
# (e.g. `ParticleSystems.particle_waves_fake.particle_equations`) for users who want to
# pick one explicitly.
#
# NOTE: we deliberately do NOT `using` the individual systems here. Several of them export
# the same names (`particle_equations`, `ODESettings`), so bringing more than one into this
# namespace would collide and emit "conflicts with an existing identifier" warnings. Only
# the canonical system's API is bound, explicitly, below.

export particle_waves_v6, particle_waves_fake
export particle_equations, ODESettings, ODEParameters, ForcingData

include("particle_waves_v6.jl")
include("particle_waves_fake.jl")

# Canonical (default) particle-system API -> currently particle_waves_v6
using .particle_waves_v6: particle_equations, ODESettings, ODEParameters, ForcingData

end
