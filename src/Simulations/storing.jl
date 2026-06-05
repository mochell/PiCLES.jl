using HDF5
using DataStructures

using ..Architectures: CartesianGrid2D, SphericalGrid2D
using ..ParticleMesh
using ..Grids.CartesianGrid: CartesianGridMesh1D, gridnotes_1d
#using Simulation

struct EmptyStore{Int} <: AbstractStore
    iteration::Int
end



mutable struct CashStore{HDFg,Int} <: AbstractStore
    store::HDFg
    iteration::Int
end



mutable struct StateStore{FL,HDFg,Int,ST} <: AbstractStore
    file::FL
    store::HDFg
    iteration::Int
    shape::ST
end

const StateOrNothing = Union{AbstractStore,Nothing}

"""
state_store(path, coords; name="state", replace=true, mode="w")
Initializes the state store at `path` with the given `coords`.
    coords are the coordinates of the state store, e.g. (time, x, y, z), with time being the first dimension.
    name is the name of the state store, e.g. "state"
    replace is a boolean, if true, the state store is replaced, if false, the state store is appended to (not tested yet).
"""
function state_store(path, coords; name="state", replace=true, mode="w")

    if replace
        rm(joinpath(path, name * ".h5"), force=true)
    end

    file = h5open(joinpath(path, name * ".h5"), "w")

    shape = Tuple(Int(x) for x in [length(x) for x in coords])

    store_waves = create_group(file, "waves")
    store_waves_data = create_dataset(
        store_waves, "data", Float64,
        (shape)
    )#, chunk=(2,))

    dim_names = [String(x) for x in keys(coords)]
    write_attribute(store_waves, "dims", dim_names)
    for (k, v) in zip(keys(coords), coords)
        #println(k, " => ", v)
        store_waves[String(k)] = v
    end
    store_waves["var_names"] = ["e", "m_x", "m_y"]


    return StateStore(file, store_waves, 1, shape)
end


function make_coords(grid::TwoDGrid)
    x = collect(LinRange(0, grid.dimx, grid.Nx))
    y = collect(LinRange(0, grid.dimy, grid.Ny))
    (x=x, y=y)
end

function make_coords(grid::OneDGrid)
    x = collect(LinRange(0, grid.dimx, grid.Nx))
    (x=x,)
end

function make_coords(grid::CartesianGrid2D)
    x = collect(LinRange(0, grid.stats.dimx, grid.stats.Nx.N))
    y = collect(LinRange(0, grid.stats.dimy, grid.stats.Ny.N))
    (x=x, y=y)
end


"""
init_state_store!(sim, save_path; state= ["e", "m_x", "m_y"],   kwargs... ,  )
Initializes the state store for the simulation sim.
    save_path is the path where the state store is saved.
    state is a vector of strings, which are the names of the state variables to be stored.
"""
function init_state_store!(sim, save_path; state=["e", "m_x", "m_y"], kwargs...)

    grid = sim.model.grid
    coords = make_coords(grid)
    #x = collect(LinRange(0, grid.dimx, grid.Nx))
    time_range = range(0.0, sim.stop_time + sim.Δt, step=sim.Δt)

    coords = (; time=Array(time_range), coords..., state=state)
    #coords = (time=Array(time_range), x=x, state=state)

    if sim.verbose
        @info "init state store"
        @info "save path: ", save_path
        @info "coords: ", keys(coords)
        @info "push state store to sim"
    end
    setfield!(sim, :store, state_store(save_path, coords; kwargs...))

    nothing
end

"""
push_state_to_storage!(sim; i=nothing)
Pushes the current state of the simulation sim to the state store.
    i is the iteration number, if nothing, the current iteration number is used.
"""
function push_state_to_storage!(sim; i=nothing)
    ii = isnothing(i) ? sim.store.iteration : i
    if length(sim.store.shape) == 4
        sim.store.store["data"][ii, :, :, :] = sim.model.State
    elseif length(sim.store.shape) == 3
        sim.store.store["data"][ii, :, :] = sim.model.State
    else
        error("wrong shape")
    end
    nothing
end


"""
reset_state_store!(sim; value=0.0)
Resets the state store to the given value.
    value is to which the state store is reset (default = 0.0).
"""
function reset_state_store!(sim; value::Float64=0.0)
    sim.store.store["data"][:, :, :] .= value
    reset_store_counter(sim.store)
    nothing
end


"""
        add_winds_forcing_to_store!(store, forcing, coords)

        add winds forcing to store
        store is a store object simulation.store
        forcing is a tuple of forcing fields
        coords is a tuple of coordinates for the forcing fields
"""
function add_winds_forcing_to_store!(store, forcing, coords)

    if ~haskey(store.file, "forcing")
        store_winds = create_group(store.file, "forcing")
    else
        store_winds = store.file["forcing"]
    end

    for (name, f) in zip(keys(forcing), forcing)
        # test if f is nothing
        if f == nothing
            continue
        else
            #print(name, f)
            if ~haskey(store_winds, string(name))
                store_i = create_dataset(store_winds, string(name), Float64, size(f))#, chunk=(2,))
            else
                store_i = f
            end
        end
    end

    # add coodinates
    shape = Tuple(Int(x) for x in [length(x) for x in coords])
    dim_names = [String(x) for x in keys(coords)]

    if ~haskey(attributes(store_winds), "dims")

        print("write dims")
        write_attribute(store_winds, "dims", dim_names)
        for (k, v) in zip(keys(coords), coords)
            println(k, " => ", v)
            store_winds[String(k)] = Array(v)
        end

    end


end


function show_stored_data(sim)
    print("store has the following keys ", HDF5.keys(sim.store.file), "\n")
    sim.store.file
end

function close_store!(store::AbstractStore)
    close(store.file)
end

function close_store!(sim)
    close_store!(sim.store)
end


function reset_store_counter(store::EmptyStore)
    nothing
end

function reset_store_counter(store::StateStore)
    store.iteration = 1
end

# -------------- converting rountines ----------

"""
convert_store_to_tuple(store::CashStore)
creates a NamedTuple with the data, x, and time from the CashStore
"""
function convert_store_to_tuple(store::CashStore, sim)
    grid = sim.model.grid
    if grid isa CartesianGridMesh1D
        # 1D run: every stored frame is (Nx, Ny, 3); all y-rows are identical, so squeeze row 1.
        j = 1
        frames = [Array(s)[:, j, :] for s in store.store]   # each (Nx, 3)
        store_data = cat(frames..., dims=3)                 # (Nx, 3, T)
        store_waves_data = permutedims(store_data, (3, 1, 2))  # (T, Nx, 3)
        wave_x = gridnotes_1d(grid)
    else
        store_data = cat(store.store..., dims=3)
        store_waves_data = permutedims(store_data, (3, 1, 2))
        wave_x = OneDGridNotes(grid).x
    end
    wave_time = collect(range(0, sim.stop_time + sim.Δt, length=size(store_waves_data, 1)))
    return (data=store_waves_data, x=wave_x, time=wave_time)
end


"""
convert_store_to_tuple(store::StateStore)
creates a NamedTuple with the data, x, and time from the StateStore
"""
function convert_store_to_tuple(store::StateStore, sim)
    store_waves_data = Array(store.store["data"])
    wave_x = store.store["x"][:]
    wave_time = store.store["time"][:]
    return (data=store_waves_data, x=wave_x, time=wave_time)
end
