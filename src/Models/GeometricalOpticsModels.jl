module GeometricalOpticsModels

export GeometricalOptics, init_particles!
export fields

using ...Architectures

using ModelingToolkit: get_states, ODESystem

#using core_1D: MarkedParticleInstance
using ...ParticleMesh: OneDGrid, OneDGridNotes, TwoDGrid, TwoDGridNotes

using ...Operators.core_2D_spread: ParticleDefaults as ParticleDefaults2D
#using core_2D: SeedParticle! as SeedParticle2D!
using ...Operators.mapping_2D

using SharedArrays
# using DistributedArrays
using Printf

import Oceananigans: fields
using Oceananigans.TimeSteppers: Clock
using ...FetchRelations

using LinearAlgebra

#includet("mapping_1D.jl")

# TO DO : check if all the modules imported above are necessary, for now I just copied the ones from WaveGrowthModels2D.jl



"""
    GeometricalOptics{Grid, Lay, Tim, Clo, stat, PC, Ovar, Osys, Oses, Odev, bl_flag, bl, wnds, cur}

    ADD HERE A DESCRIPTION OF THE MODEL
    
"""
mutable struct GeometricalOptics{Grid<:AbstractGrid,
                        Lay,
                        Tim,
                        Clo,
                        Int,
                        stat,
                        Pan,
                        PCollection,
                        PPool,
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
                        as_type,
                        as_thresh,
                        pb_cov_init,
                        plt_steps,
                        plt_path,
                        sve_part,
                        nPart,
                        wnds,
                        cur,
                        Mstat} <: Abstract2DStochasticModel where {Mstat<:Union{Nothing,stat}, PCollection<:Union{Vector,Array}}
    #Union{Vector,DArray}
    grid::Grid
    layers::Lay      # number of layers used in the model, 1 is eneough
    timestepper::Tim      # silly Oceananigans
    clock::Clo
    dims::Int      # number of dimensions

    State::stat     # state of of the model at each grid point, for each layer it contains, energy, positions, group speed and directional spreading
    ParticlesAtNode::Pan      # list of the particles to regrid at each node
    ParticleCollection::PCollection    # Collection (list) of Particles
    ParticlePool::PPool                 # Pool of particles (used for non parametric mode)
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

    angular_spreading_thresh::as_thresh     # threshold for splitting
    angular_spreading_type::as_type     # type of angular spreading used
    proba_covariance_init::pb_cov_init     # initial probability density 4x4 covariance matrix of particles

    plot_steps::plt_steps          # Whether or not to plot the steps 
    plot_savepath::plt_path         # The path to use for saving plots

    save_particles::sve_part        # Whether or not to save particle data

    n_particles_launch::nPart       # Number of particles to launch from each node when using a nonparametric spreading type

    winds::wnds     # u, v, if needed u_x, u_y
    currents::cur      # u, v, currents

    MovieState::Mstat     # state of of the model. Only used for producing movieframes

end




## 2D version
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
function GeometricalOptics(; grid::TwoDGrid,
    winds::NamedTuple{(:u, :v)}, 
    ODEsys, 
    ODEvars=nothing, #needed for MTK for ODEsystem. will be depriciated later
    layers::Int=1,
    clock=Clock{eltype(grid)}(;time=0, last_Δt=0, last_stage_Δt=1),
    ODEsets::AbstractODESettings=nothing,  # ODE_settings
    ODEinit_type::PP= "wind_sea",  # default_ODE_parameters
    minimal_particle=nothing, # minimum particle the model falls back to if a particle fails to integrate
    minimal_state=nothing, # minimum state threshold needed for the state to be advanced 
    currents=nothing,  # 
    periodic_boundary=true,
    boundary_type="same", # or "minimal", "same", default is same, only used if periodic_boundary is false
    angular_spreading_thresh=π/8,
    angular_spreading_type="stochast",  # or "geometrical" or "nonparametric"
    proba_covariance_init = Matrix(I,4,4)*10^(-50),
    plot_steps=false,
    plot_savepath="",
    save_particles=false,
    n_particles_launch=150,
    CBsets=nothing,
    movie=false) where {PP<:Union{ParticleDefaults2D,String}}

    # initialize state {SharedArray} given grid and layers
    # Number of state variables 
    Nstate = 4
    if layers > 1
        State = SharedArray{Float64,4}(grid.Nx, grid.Ny, Nstate, layers)
    else
        State = SharedArray{Float64,3}(grid.Nx, grid.Ny, Nstate)
    end

    if layers > 1
        ParticlesAtNode = Array{Array{Array{Array{Any,1},1},1},1}()
        for i in 1:grid.Nx
            push!(ParticlesAtNode, [])
            for j in 1:grid.Ny
                push!(ParticlesAtNode[i], [])
                for _ in 1:layers
                    push!(ParticlesAtNode[i][j], [])
                end
            end
        end
        ParticlePool = Array{Array{Any,1},1}()
    else
        ParticlesAtNode = Array{Array{Array{Any,1},1},1}()
        for i in 1:grid.Nx
            push!(ParticlesAtNode, [])
            for _ in 1:grid.Ny
                push!(ParticlesAtNode[i], [])
            end
        end
        ParticlePool = Array{Any,1}()
    end

    if ODEinit_type isa ParticleDefaults2D
        ODEdefaults = ODEinit_type
    elseif ODEinit_type == "wind_sea"
        ODEdefaults = nothing
    elseif ODEinit_type == "mininmal"
        ODEdefaults = ParticleDefaults1D(-11.0, 1e-3, 0.0)
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

    if boundary_type == "wind_sea"
        boundary_defaults = nothing
        @info "use wind_sea boundary"
    elseif boundary_type == "mininmal"
        @info "use 'mininmal' boundary (1min with 2m/s)"
        #FetchRelations.get_minimal_windsea(u(0, 0), ODEsets.DT)
        WindSeamin = FetchRelations.get_minimal_windsea(1, 1, 5*60) # 5 min with 2 m/s
        #WindSeamin = FetchRelations.get_minimal_windsea(u(0, 0, 0), v(0, 0, 0), DT / 2)
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

    ParticleCollection = []
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
    return GeometricalOptics(
        grid,
        layers,
        nothing,
        clock, # ???
        2, # This is a 2D model
        State,
        ParticlesAtNode,
        ParticleCollection,
        ParticlePool,
        FailedCollection, 
        ODEvars,
        ODEsys,
        ODEsets,
        ODEdefaults,
        minimal_particle, 
        minimal_state,
        periodic_boundary,
        boundary, boundary_defaults,
        angular_spreading_thresh,
        angular_spreading_type,
        proba_covariance_init,
        plot_steps,
        plot_savepath,
        save_particles,
        n_particles_launch,
        winds,
        currents,
        Mstat)
end




"""
    fields(model::WaveGrowth)

Return a flattened `NamedTuple` of the State vector for a `GeometricalOptics` model.
"""
fields(model::GeometricalOptics) = (State=model.State,)
# # Oceananigans.Simulations interface
# fields(m::ContinuumIceModel) = merge(m.velocities, m.stresses)

function reset_boundary!(model::GeometricalOptics)

    if model.periodic_boundary  # if false, define boundary points here:
        boundary = []
    else
        boundary = boundary = mark_boundary(grid)
    end

end


function Base.show(io::IO, ow::GeometricalOptics)

    if ow.ODEsystem isa ODESystem
        sys_print = get_states(ow.ODEsystem)
    else
        sys_print = ow.ODEsystem
    end
    print(io, "GeometricalOptics ", "\n",
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