module WaveGrowthModels2D

export WaveGrowth2D, init_particles!, init_StateArray, mark_boundary, reset_boundary!, fields
export fields

using ...Architectures


#using core_1D: MarkedParticleInstance
using ...ParticleMesh: OneDGrid, OneDGridNotes, TwoDGrid, TwoDGridNotes

using ...Operators.core_2D: ParticleDefaults as ParticleDefaults2D
using ...Operators.mapping_2D

using SharedArrays
using StaticArrays
using StructArrays
# using DistributedArrays
using Printf

import Oceananigans: fields
using Oceananigans.TimeSteppers: Clock
using ...FetchRelations

using ...custom_structures: ParticleInstance2D

using ...Grids: make_boundary_lists
#includet("mapping_1D.jl")





"""
    WaveGrowth2D{Grid, Lay, Tim, Clo, stat, PC, Ovar, Osys, Oses, Odev, bl_flag, bl, wnds, cur}

    WaveGrowth2D is a model for simulating wave growth in a 1D domain. 
    The model is based on the particle method, where each particle represents wave statistics at a given point in space.
    This structure contains all the information needed to run the model, including the grid, the particles, the ODE system, the boundary conditions, and the state of the model. 

"""
mutable struct WaveGrowth2D{Grid<:AbstractGrid,
                        Lay,
                        Tim,
                        Clo,
                        Int,
                        stat,
                        PCollection,
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
                        cur,
    Mstat} <: Abstract2DModel where {Mstat<:Union{Nothing,stat},PCollection<:Union{Vector,Array,StructArray}}
    #Union{Vector,DArray}
    grid::Grid
    layers::Lay      # number of layers used in the model, 1 is eneough
    timestepper::Tim      # silly Oceananigans
    clock::Clo
    dims::Int      # number of dimensions

    State::stat     # state of of the model at each grid point, for each layer it contains, energy, positions, group speed
    ParticleCollection::PCollection    # Collection (list) of Particles
    FailedCollection::FPC      # Collection (list) of Particles that failed to integrate

    ODEvars::Ovar     # list of variables in ODE system, have type Num from OrdinaryDiffEq and ModelingToolkit
    ODEsystem::Osys     # the ODE system used at each particle
    ODEsettings::Oses     # All setting needed to solve the ODEsystem
    ODEdefaults::Odev     # Dict{NUm, Float64} ODE defaults
    minimal_particle::MinPar
    minimal_state::MinStat

    periodic_boundary::bl_flag # If true we use a period boundary 
    boundary::bl       # List of boundary points
    boundary_defaults::bl_type # Dict{NUm, Float64} ODE defaults
    ocean_points::Vector
    boundary_points::Vector

    winds::wnds     # u, v, if needed u_x, u_y
    currents::cur      # u, v, currents

    MovieState::Mstat     # state of of the model. Only used for producing movieframes

end




## 2D version

"""
    init_StateArray(grid::TwoDGrid, Nstate, layers)

    # Arguments
    - `grid::TwoDGrid`: The TwoDGrid object representing the grid.
    - `Nstate`: The number of states in the StateArray.
    - `layers`: The number of layers in the StateArray.

    # Returns
    - `State`: The initialized StateArray.

    If `layers` is greater than 1, a 4-dimensional SharedArray of size (grid.Nx, grid.Ny, Nstate, layers) is created.
    Otherwise, a 3-dimensional SharedArray of size (grid.Nx, grid.Ny, Nstate) is created.
"""
function init_StateArray(grid::TwoDGrid, Nstate, layers)
    if layers > 1
        State = SharedArray{Float64,4}(grid.Nx, grid.Ny, Nstate, layers)
    else
        State = SharedArray{Float64,3}(grid.Nx, grid.Ny, Nstate)
    end
    return State
end


"""
    init_StateArray(grid::MeshGrids, Nstate, layers)

    # Arguments
    - `grid::MeshGrids`: The grid object representing the grid.
    - `Nstate`: The number of states in the StateArray.
    - `layers`: The number of layers in the StateArray.

    # Returns
    - `State`: The initialized StateArray.

    If `layers` is greater than 1, a 4-dimensional SharedArray of size (grid.stats.Nx, grid.stats.Ny, Nstate, layers) is created.
    Otherwise, a 3-dimensional SharedArray of size (grid.stats.Nx, grid.stats.Ny, Nstate) is created.
"""
function init_StateArray(grid::MeshGrids, Nstate, layers)
    if layers > 1
        State = SharedArray{Float64,4}(grid.stats.Nx.N, grid.stats.Ny.N, Nstate, layers)
    else
        State = SharedArray{Float64,3}(grid.stats.Nx.N, grid.stats.Ny.N, Nstate)
    end
    return State
end


"""
mark_boundary(grid::TwoDGrid)
function that returns list of boundary nodes (tuples of indixes)
"""
function mark_boundary(grid::TwoDGrid)
    #get x and y coordinates
    xi = collect(range(1, stop=grid.Nx, step=1))
    yi = collect(range(1, stop=grid.Ny, step=1))

    # make boundary nodes
    a = [(xi[i], yi[j]) for j in 1:grid.Ny   , i in [1, grid.Nx]]
    b = [(xi[i], yi[j]) for j in [1, grid.Ny], i in 1:grid.Nx   ]
    #merge a and b
    return vcat(vec(a), vec(b)) #vec(vcat(a, b))
end


"""
mark_boundary(grid::MeshGrids)
function that returns list of boundary nodes (tuples of indixes)
"""
function mark_boundary(grid::MeshGrids)
    #get x and y coordinates
    xi = collect(range(1, stop=grid.stats.Nx.N, step=1))
    yi = collect(range(1, stop=grid.stats.Ny.N, step=1))

    # make boundary nodes
    a = [(xi[i], yi[j]) for j in 1:grid.stats.Ny.N, i in [1, grid.stats.Nx.N]]
    b = [(xi[i], yi[j]) for j in [1, grid.stats.Ny.N], i in 1:grid.stats.Nx.N]
    #merge a and b
    return [CartesianIndex(i) for i in vcat(vec(a), vec(b))] #vec(vcat(a, b))
end

"""  
WaveGrowth2D(; grid, winds, ODEsys, ODEvars, layers, clock, ODEsets, ODEdefaults, currents, periodic_boundary, CBsets)
This is the constructor for the WaveGrowth2D model. The inputs are:
    grid             : the grid used in the model,
    winds            : the wind interpolation function here only 1D,
    ODEsys           : the ODE system used in the model,
    ODEvars          : the variables in the ODE system,
    layers           : the number of layers used in the model (default 1),
    clock            : the clock used in the model,
    ODEsets          : the ODE settings (type: ODESettings),
    ODEdefaults      : the ODE defaults (type: ParticleDefaults),
    currents         : the currents (not implimented yet),
    periodic_boundary: if true we use a periodic boundary (default true),
    CBsets           : the callback settings (not implimented yet).
"""
function WaveGrowth2D(; grid::GG,
    winds::NamedTuple{(:u, :v)}, 
    ODEsys, 
    ODEvars=nothing, #needed for MTK for ODEsystem. will be depriciated later
    layers::Int=1,
    clock=Clock{eltype(grid)}(time=0.0),
    ODEsets::AbstractODESettings=nothing,  # ODE_settings
    ODEinit_type::PP= "wind_sea",  # default_ODE_parameters
    minimal_particle=nothing, # minimum particle the model falls back to if a particle fails to integrate
    minimal_state=nothing, # minimum state threshold needed for the state to be advanced 
    currents=nothing,  # 
    periodic_boundary=true,
    boundary_type="same", # or "minimal", "same", default is same, only used if periodic_boundary is false
    CBsets=nothing,
    movie=false) where {PP<:Union{ParticleDefaults2D,String},GG<:AbstractGrid}

    # initialize state {SharedArray} given grid and layers
    # Number of state variables 
    Nstate = 3
    State = init_StateArray(grid, Nstate, layers)

    # depriciated for new Grid logic
    # if layers > 1
    #     State = SharedArray{Float64,4}(grid.Nx, grid.Ny, Nstate, layers)
    # else
    #     State = SharedArray{Float64,3}(grid.Nx, grid.Ny, Nstate) # StateTypeL1
    #     #State = @MArray zeros(grid.Nx, grid.Ny, Nstate)
    # end

    if ODEinit_type isa ParticleDefaults2D
        ODEdefaults = ODEinit_type
    elseif ODEinit_type == "wind_sea"
        ODEdefaults = nothing
    elseif ODEinit_type == "mininmal"
        ODEdefaults = ParticleDefaults2D(-11.0, 1e-3, 0.0)
    else
        error("ODEinit_type must be either 'wind_sea','mininmal', or ParticleDefaults2D instance ")
    end


    if isnothing(minimal_particle)
        @info "initalize minimum particle"
        minimal_particle = FetchRelations.MinimalParticle(2, 2, ODEsets.timestep)
    else
        @info "use minimal particle"
    end

    if isnothing(minimal_state)
        @info "initalize minimum state"
        minimal_state = FetchRelations.MinimalState(2, 2, ODEsets.timestep)
    else
        @info "use minimal state"
    end

    # initliaze boundary points (periodic_boundary)
    if ~periodic_boundary  # if false, define boundary points here:
        boundary = boundary = mark_boundary(grid)
    else
        boundary = []
    end

    # create grid points lists.
    Glists = make_boundary_lists(grid.data.mask)
    if periodic_boundary
        # all wave points to concider:
        ocean_points = vcat(Glists.ocean, Glists.grid_boundary)

        # all boundary points to concider
        boundary_points = Glists.land_boundary
    else
        # all wave points to concider:
        ocean_points= Glists.ocean

        # all boundary points to concider
        boundary_points = vcat(Glists.land_boundary, Glists.grid_boundary)

    end


    if boundary_type == "wind_sea"
        boundary_defaults = nothing
        @info "use wind_sea boundary"
    elseif boundary_type == "mininmal"
        @info "use 'mininmal' boundary (1min with 2m/s)"
        #FetchRelations.MinimalWindsea(u(0, 0), ODEsets.DT)
        WindSeamin = FetchRelations.MinimalWindsea(1, 1, 5*60) # 5 min with 2 m/s
        #WindSeamin = FetchRelations.MinimalWindsea(u(0, 0, 0), v(0, 0, 0), DT / 2)
        #WindSeamin = FetchRelations.get_initial_windsea(u(0, 0, 0), v(0, 0, 0), DT/5)
        lne_local = log(WindSeamin["E"])
        cg_u_local = WindSeamin["cg_bar_x"]
        cg_v_local = WindSeamin["cg_bar_y"]
        boundary_defaults = copy(ParticleDefaults2D(lne_local, cg_u_local, cg_v_local, 0.0, 0.0))

    elseif boundary_type == "same"
        @info "use default value boundary"
        boundary_defaults = ODEdefaults
    else
        error("boundary_type must be either 'wind_sea','mininmal', or 'same' ")
    end


    if typeof(grid) <: MeshGrids
        Nx, Ny = grid.stats.Nx.N, grid.stats.Ny.N
    elseif typeof(grid) <: StandardRegular2D_old
        Nx, Ny = grid.Nx, grid.Ny
        # @info "WaveGrowthModel: StandardRegular2D_old"
    else
        @info "WaveGrowthModel: no grid detected"
    end


    # ParticleCollection = []
    ParticleCollection = StructArray{ParticleInstance2D}(undef, Nx, Ny)
    FailedCollection = Vector{AbstractMarkedParticleInstance}([])
    # particle initialization is  not done in the init_particle! method
    # for i in range(1,length = grid.Nx)
    #         SeedParticle!(ParticleCollection, State, i,
    #                       ODEsys, ODEdev, ODEsets,
    #                       gridnotes, winds, ODEsets.timestep, grid.Nx, boundary, periodic_boundary)
    # end        

    # check dimensions of all components:
    # check Nstate ODEvars
    if movie
        Mstat = State
    else
        Mstat = nothing
    end

    # return WaveGrowth2D structure
    return WaveGrowth2D(
        grid,
        layers,
        nothing,
        clock, # ???
        2, # This is a 2D model
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
        boundary, boundary_defaults,
        ocean_points, boundary_points,
        winds,
        currents,
        Mstat)
end




"""
    fields(model::WaveGrowth)

Return a flattened `NamedTuple` of the State vector for a `WaveGrowth2D` model.
"""
fields(model::Abstract2DModel) = (State=model.State,)
# # Oceananigans.Simulations interface
# fields(m::ContinuumIceModel) = merge(m.velocities, m.stresses)

function reset_boundary!(model::Abstract2DModel)

    if model.periodic_boundary  # if false, define boundary points here:
        boundary = []
    else
        boundary = boundary = mark_boundary(grid)
    end

end


function Base.show(io::IO, ow::WaveGrowth2D)

    sys_print = ow.ODEsystem
    print(io, "WaveGrowth2D ", "\n",
        "├── grid: ", ow.grid, "\n",
        "├── layers: ", ow.layers, "\n",
        "├── clock: ", ow.clock,
        "├── State: ", size(ow.State), "\n",
        "├── ParticleCollection size: ", length(ow.ParticleCollection), "\n",
        "├── ODEs  \n",
        "|    ├── System: ", sys_print, "\n",
        "|    ├── Defaults: ", ow.ODEdefaults, "\n",
        "|    └── Settings:  \n", ow.ODEsettings, "\n",
        "├── winds ", ow.winds, "\n",
        "├── currents ", ow.currents, "\n",
        "└── Perdiodic Boundary ", ow.periodic_boundary, "\n")
end




# end of module
end