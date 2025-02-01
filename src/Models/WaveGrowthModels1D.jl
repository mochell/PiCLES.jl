module WaveGrowthModels1D

export WaveGrowth1D
export fields, reset_boundary!, show

using ...Architectures


#using core_1D: MarkedParticleInstance
using ...ParticleMesh: OneDGrid, OneDGridNotes, TwoDGrid, TwoDGridNotes

using ...Operators.core_1D: ParticleDefaults as ParticleDefaults1D
using ...Operators.mapping_1D

using SharedArrays
using Printf

import Oceananigans: fields
using Oceananigans.TimeSteppers: Clock, tick!
using ...FetchRelations 

#includet("mapping_1D.jl")



"""
        WaveGrowth1D{Grid, Lay, Tim, Clo, stat, PC, Ovar, Osys, Oses, Odev, bl_flag, bl, wnds, cur}

        WaveGrowth1D is a model for simulating wave growth in a 1D domain. 
        The model is based on the particle method, where each particle represents wave statistics at a given point in space.
        This structure contains all the information needed to run the model, including the grid, the particles, the ODE system, the boundary conditions, and the state of the model. 

"""
mutable struct WaveGrowth1D{Grid<:AbstractGrid,
                            Lay,
                            Tim,
                            Clo,
                            Int,
                            stat,
                            PC,
                            FPC,
                            Ovar,
                            Osys,
                            Oses,
                            Odev,
                            MinPar,
                            MinStat,
                            bl_flag,
                            bl,
                            bl_type,
                            wnds,
                            cur} <: Abstract1DModel
    grid::Grid
    layers::Lay            # number of layers used in the model, 1 is eneough
    timestepper::Tim            # silly Oceananigans
    clock::Clo
    dims::Int            # number of dimensions

    State::stat         # state of of the model at each grid point, for each layer it contains, energy, positions, group speed
    ParticleCollection::PC        # Collection (list) of Particles
    FailedCollection::FPC            # Collection (list) of Particles that failed to integrate

    ODEvars::Ovar         # list of variables in ODE system, have type Num from ModelingToolkit, obsolute when ModelingToolkit is removed.
    ODEsystem::Osys         # the ODE system used at each particle
    ODEsettings::Oses         # All setting needed to solve the ODEsystem
    ODEdefaults::Odev         # Dict{NUm, Float64} ODE defaults
    minimal_particle::MinPar
    minimal_state::MinStat

    periodic_boundary::bl_flag # If true we use a period boundary 
    boundary::bl                # List of boundary points
    boundary_defaults::bl_type # Dict{NUm, Float64} ODE defaults

    winds::wnds         # u, v, if needed u_x, u_y
    currents::cur            # u, v, currents

end

"""    
    WaveGrowth1D(; grid, winds, ODEsys, ODEvars, layers, clock, ODEsets, ODEdefaults, currents, periodic_boundary, CBsets)
    This is the constructor for the WaveGrowth1D model. The inputs are:
            grid                         : the grid used in the model,
            winds                        : the wind interpolation function here only 1D,
            ODEsys                     : the ODE system used in the model,
            ODEvars                    : the variables in the ODE system,
            layers                     : the number of layers used in the model (default 1),
            clock                        : the clock used in the model,
            ODEsets                    : the ODE settings (type: ODESettings),
            ODEdefaults            : the ODE defaults (type: ParticleDefaults),
            currents                 : the currents (not implimented yet),
            periodic_boundary: if true we use a periodic boundary (default true),
            boundary_type        : the type of boundary, either "wind_sea" or "flat" (default "wind_sea"),
            CBsets                     : the callback settings (not implimented yet).
"""
function WaveGrowth1D(; grid::OneDGrid, winds, ODEsys, 
    ODEvars=nothing, #neede dfr MTK for ODEsystem. will be depriciated
    layers::Int=1,
    clock=Clock{eltype(grid)}(time=0.0),
    ODEsets::AbstractODESettings=nothing,    # ODE_settings
    ODEinit_type::PP="wind_sea", # or "minimal", or ParticleDefaults1D instance, default is wind_sea nothing,    # default_ODE_parameters
    minimal_particle=nothing, # minimum particle the model falls back to if a particle fails to integrate
    minimal_state=nothing, # minimum state threshold needed for the state to be advanced 
    currents=nothing,    # 
    periodic_boundary=true,
    boundary_type="same", # or "minimal", "same", default is same, only used if periodic_boundary is false
    CBsets=nothing) where {PP<:Union{ParticleDefaults1D,String}}

    # initialize state {SharedArray} given grid and layers
    # Number of state variables 
    Nstate = 2
    if layers > 1
        State = SharedArray{Float64,3}(grid.Nx, Nstate, layers)
    else
        State = SharedMatrix{Float64}(grid.Nx, Nstate)
    end

    if ODEinit_type isa ParticleDefaults1D
        ODEdefaults = ODEinit_type
    elseif ODEinit_type == "wind_sea"
        ODEdefaults = nothing
    elseif ODEinit_type == "mininmal"
        ODEdefaults = ParticleDefaults1D(-11.0, 1e-3, 0.0)
    else
        error("ODEinit_type must be either 'wind_sea','mininmal', or ParticleDefaults1D instance ")
    end

    if isnothing(minimal_particle)
        @info "initalize minimum particle"
        minimal_particle = FetchRelations.MinimalParticle(2, 0, ODEsets.timestep)
    else
        @info "use minimal particle"
    end

    if isnothing(minimal_state)
        @info "initalize minimum state"
        minimal_state = FetchRelations.MinimalState(2, 0, ODEsets.timestep)
    else
        @info "use minimal state"
    end


    # initliaze boundary points (periodic_boundary)
    if ~periodic_boundary    # if false, define boundary points here:
        boundary = [1, grid.Nx]
    else
        boundary = []
    end

    if boundary_type == "wind_sea"
        boundary_defaults = nothing
        @info "use wind_sea boundary"
    elseif boundary_type == "mininmal"
        @info "use zero boundary"
        #FetchRelations.MinimalWindsea(u(0, 0), ODEsets.DT)
        boundary_defaults = copy(ParticleDefaults2D(-11.0, 1e-3, 0.0))
    elseif boundary_type == "same"
        @info "use same default value boundary"
        boundary_defaults = ODEdefaults
    else
        error("boundary_type must be either 'wind_sea','mininmal', or 'same' ")
    end


    #ODEdev = copy(ODEdefaults)
    ### seed particles
    # @printf "seed PiCLES ... \n"
    # #seed particles
    ParticleCollection = []
    FailedCollection = Vector{AbstractMarkedParticleInstance}([])
    # particle initialization is    not done in the init_particle! method
    # for i in range(1,length = grid.Nx)
    #                 SeedParticle!(ParticleCollection, State, i,
    #                                             ODEsys, ODEdev, ODEsets,
    #                                             gridnotes, winds, ODEsets.timestep, grid.Nx, boundary, periodic_boundary)
    # end                

    # check dimensions of all components:
    # check Nstate ODEvars

    # return WaveGrowth1D structure
    return WaveGrowth1D(grid, layers,
        nothing,
        clock, # ???
        1, # This is a 1D model
        State,
        ParticleCollection,
        FailedCollection,
        ODEvars,
        ODEsys,
        ODEsets,
        ODEdefaults,
        minimal_particle,
        minimal_state,
        periodic_boundary,
        boundary,
        boundary_defaults,
        winds,
        currents)
end



"""
        fields(model::WaveGrowth1D)

Return a flattened `NamedTuple` of the State vector for a `WaveGrowth1D` model.
"""
fields(model::WaveGrowth1D) = (State=model.State,)
# # Oceananigans.Simulations interface
# fields(m::ContinuumIceModel) = merge(m.velocities, m.stresses)


function reset_boundary!(model::WaveGrowth1D)
    if model.periodic_boundary
        model.boundary = []
    else
        model.boundary = [1, model.grid.Nx]
    end
    nothing
end


# function Base.show(io::IO, ow::WaveGrowth1D)

#     if ow.ODEsystem isa ODESystem
#         sys_print = get_states(ow.ODEsystem)
#     else
#         sys_print = ow.ODEsystem
#     end

#     print(io, "WaveGrowth1D ", "\n",
#         "├── grid: ", ow.grid, "\n",
#         "├── layers: ", ow.layers, "\n",
#         "├── clock: ", ow.clock,
#         "├── State: ", size(ow.State), "\n",
#         "├── ParticleCollection size: ", length(ow.ParticleCollection), "\n",
#         "├── ODEs    \n",
#         "|        ├── System: ", sys_print, "\n",
#         "|        ├── Defaults: ", ow.ODEdefaults, "\n",
#         "|        └── Settings:    \n", ow.ODEsettings, "\n",
#         "├── winds ", ow.winds, "\n",
#         "├── currents ", ow.currents, "\n",
#         "└── Perdiodic Boundary ", ow.periodic_boundary, "\n")
# end




# end of module
end