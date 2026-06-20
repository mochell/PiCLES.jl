module TimeSteppers

export time_step!
using ...Architectures
using ..mapping_2D
using ...ParticleSystems: ForcingData
using ...custom_structures: ForcingCollection, AbstractForcingField, FunctionForcingField, ArrayForcingField

# for debugging
using Statistics
using Base.Threads
using Printf

using Oceananigans.TimeSteppers: tick!

_sample_forcing_field(field, x, y, t) = field isa Function ? field(x, y, t) : field

"""
Lightweight container for a 2D forcing snapshot at one global model time.

Both fields are callables with signature `f(x, y)` (time is already sliced/fixed).
"""
struct ForcingSlice2D{UF,VF}
    u::UF
    v::VF
end

function Base.show(io::IO, forcing::ForcingSlice2D)
    print(io, "ForcingSlice2D\n",
          "├── u(x,y): ", typeof(forcing.u), "\n",
          "└── v(x,y): ", typeof(forcing.v))
end

_axis_from_grid_x(grid) = vec(grid.data.x[:, 1])
_axis_from_grid_y(grid) = vec(grid.data.y[1, :])

"""
Return the nearest valid axis index for a coordinate value.

Values outside the axis range are clamped to the boundary index.
"""
function _nearest_index(axis::AbstractVector{<:Real}, value::Real)
    idx = searchsortedfirst(axis, value)
    if idx <= 1
        return 1
    elseif idx > length(axis)
        return length(axis)
    else
        left = idx - 1
        return abs(axis[idx] - value) < abs(value - axis[left]) ? idx : left
    end
end

"""
Evaluate forcing function with either `(x, y, t_global)` or `(x, y)` signature.
"""
function _call_forcing_function(f, x, y, t_global)
    if applicable(f, x, y, t_global)
        return f(x, y, t_global)
    elseif applicable(f, x, y)
        return f(x, y)
    else
        error("Forcing function must support either (x, y, t_global) or (x, y)")
    end
end

"""
Convert array-backed forcing to an `(x, y) -> value` callable.

- 2D arrays are interpreted as `(x, y)` snapshots.
- 3D arrays are interpreted as `(x, y, t)` and sliced once at `t_global`.
"""
function _slice_array_field(field::AbstractArray, x_axis::AbstractVector{<:Real}, y_axis::AbstractVector{<:Real}, t_axis, t_global)
    if ndims(field) == 2
        A = field
    elseif ndims(field) == 3
        tidx = t_axis === nothing ? clamp(round(Int, t_global), 1, size(field, 3)) : _nearest_index(t_axis, t_global)
        A = @view field[:, :, tidx]
    else
        error("Forcing array must be 2D (x,y) or 3D (x,y,t)")
    end

    return (x, y) -> begin
        xi = _nearest_index(x_axis, x)
        yi = _nearest_index(y_axis, y)
        A[xi, yi]
    end
end

"""Fix time `t_global` for function-backed forcing and expose `(x, y)` callable."""
function _field_to_xy(field::FunctionForcingField, x_axis::AbstractVector{<:Real}, y_axis::AbstractVector{<:Real}, t_global)
    return (x, y) -> _call_forcing_function(field.f, x, y, t_global)
end

"""
Resolve array-backed forcing wrapper to `(x, y)` callable.

If wrapper axes are not provided, grid axes are used.
"""
function _field_to_xy(field::ArrayForcingField, x_axis::AbstractVector{<:Real}, y_axis::AbstractVector{<:Real}, t_global)
    local_x = field.x === nothing ? x_axis : field.x
    local_y = field.y === nothing ? y_axis : field.y
    return _slice_array_field(field.data, local_x, local_y, field.t, t_global)
end

"""Normalize plain function forcing to `(x, y)` by fixing time `t_global`."""
function _field_to_xy(field::Function, x_axis::AbstractVector{<:Real}, y_axis::AbstractVector{<:Real}, t_global)
    return (x, y) -> _call_forcing_function(field, x, y, t_global)
end

"""Normalize plain array forcing using grid axes and nearest-neighbor sampling."""
function _field_to_xy(field::AbstractArray, x_axis::AbstractVector{<:Real}, y_axis::AbstractVector{<:Real}, t_global)
    return _slice_array_field(field, x_axis, y_axis, nothing, t_global)
end

"""Normalize scalar forcing to constant `(x, y)` callable."""
function _field_to_xy(field::Number, x_axis::AbstractVector{<:Real}, y_axis::AbstractVector{<:Real}, t_global)
    return (x, y) -> field
end

"""Normalize missing forcing to zero-valued `(x, y)` callable."""
function _field_to_xy(field::Nothing, x_axis::AbstractVector{<:Real}, y_axis::AbstractVector{<:Real}, t_global)
    return (x, y) -> 0.0
end

"""
Build a `ForcingSlice2D` from a `ForcingCollection` at model time `t_global`.

Resulting `u`/`v` are spatial callables only (`x, y`).
"""
function time_slice_forcing(forcing::ForcingCollection, grid, t_global)
    x_axis = _axis_from_grid_x(grid)
    y_axis = _axis_from_grid_y(grid)
    return ForcingSlice2D(
        _field_to_xy(forcing.u_wind, x_axis, y_axis, t_global),
        _field_to_xy(forcing.v_wind, x_axis, y_axis, t_global),
    )
end

"""Compatibility path for legacy named-tuple forcing with time-dependent `u`/`v` callables."""
function time_slice_forcing(forcing::NamedTuple{(:u, :v)}, grid, t_global)
    return ForcingSlice2D(
        (x, y) -> _call_forcing_function(forcing.u, x, y, t_global),
        (x, y) -> _call_forcing_function(forcing.v, x, y, t_global),
    )
end

"""Compatibility path for per-particle scalar forcing state (constant in space and time slice)."""
function time_slice_forcing(forcing::ForcingData, grid, t_global)
    return ForcingSlice2D(
        (x, y) -> forcing.u_wind,
        (x, y) -> forcing.v_wind,
    )
end

function mean_of_state(model::Abstract2DModel)
    return mean(model.State[:, :, 1])
end

function max_energy(model::Abstract2DModel)
    return maximum(model.State[:, :, 1])
end

function max_cgx(model::Abstract2DModel)
    return maximum(model.State[:, :, 2])
end

function max_cgy(model::Abstract2DModel)
    return maximum(model.State[:, :, 3])
end


################# 2D ####################

"""
time_step!(model, Δt; callbacks=nothing)

advances model by 1 time step:
1st) the model.ParticleCollection is advanced and then 
2nd) the model.State is updated.
clock is ticked by Δt

callbacks are not implimented yet

"""
function time_step!(model::Abstract2DModel, Δt::Float64; forcing=nothing, callbacks=nothing, debug=false)
    
    #
    current_forcing = forcing === nothing ? (hasproperty(model, :winds) ? getproperty(model, :winds) : nothing) : forcing
    current_forcing === nothing && error("2D time_step! requires forcing data (pass `forcing=...` to Simulation or set `model.winds`)")

    forcing_xy = time_slice_forcing(current_forcing, model.grid, model.clock.time)

    # temporary FailedCollection to store failed particles
    FailedCollection = Vector{AbstractMarkedParticleInstance}([])

    #print("mean energy before advance ", mean_of_state(model), "\n")
    if debug
        @info "before advance"
        # @info maximum(model.State[:, :, 1]), maximum(model.State[:, :, 2]), maximum(model.State[:, :, 3])
        model.FailedCollection = FailedCollection
        @show model.ParticleCollection[:, 6].on
    end 

    # Resolve the B-spline deposition order Int -> Val{P} ONCE here (function barrier). Below this
    # point everything specializes on Val{P} and compiles once per order (cached across steps);
    # the only per-step cost is this single dispatch. See issues #59/#60.
    spline_val = Val(hasproperty(model, :spline_order) ? model.spline_order : 1)
    time_step!_advance(model, Δt, forcing_xy, FailedCollection, spline_val)
    # @threads for a_particle in model.ParticleCollection[model.ocean_points]
    #     #@info a_particle.position_ij
    #     mapping_2D.advance!(    a_particle, model.State, FailedCollection,
    #                             model.grid, model.winds, Δt,
    #                             model.ODEsettings.log_energy_maximum,
    #                             model.ODEsettings.wind_min_squared,
    #                             model.periodic_boundary,
    #                             model.ODEdefaults)
    # end
    

    if debug
        # print("mean energy after advance, before remesh ", mean_of_state(model), "\n")

        @info "after advance, before remesh: "
        # @info maximum(model.State[:, :, 1]), maximum(model.State[:, :, 2]), maximum(model.State[:, :, 3])
        #@info model.State[8:12, 1], model.State[8:12, 2]
        #@info model.clock.time, model.ParticleCollection[10].ODEIntegrator.t
        #@info "winds:", model.winds.u(model.ParticleCollection[10].ODEIntegrator.u[4], model.ParticleCollection[10].ODEIntegrator.u[5], model.ParticleCollection[10].ODEIntegrator.t)
        @show model.ParticleCollection[:, 6].on
    end

    #@printf "re-mesh"
    time_step!_remesh(model, Δt, forcing_xy)
    # @threads for a_particle in model.ParticleCollection[model.ocean_points]
    #     mapping_2D.remesh!(a_particle, model.State, 
    #                     model.winds, model.clock.time, 
    #                     model.ODEsettings, Δt,
    #                     model.grid.stats, 
    #                     model.minimal_state,
    #                     model.ODEdefaults)
    # end

    if debug
        @info "after remeshed: "
        #@info model.State[8:12, 1], model.State[8:12, 2]
        # @info maximum(model.State[:, :, 1]), maximum(model.State[:, :, 2]), maximum(model.State[:, :, 3])
        # @info model.clock.time, model.ParticleCollection[10].ODEIntegrator.t
        @show model.ParticleCollection[:, 6].on

    end
    #print("mean energy after remesh ", mean_of_state(model), "\n")

    # @printf("------- max state E=%.4e cgx=%.4e cgy=%.4e \n", max_energy(model), max_cgx(model), max_cgy(model))
    tick!(model.clock, Δt)


end

function time_step!_advance(model::Abstract2DModel, Δt::Float64, forcing_xy, FailedCollection::Vector{AbstractMarkedParticleInstance}, spline::Val{P}=Val(1)) where {P}

    @threads for a_particle in model.ParticleCollection[model.ocean_points]
        #@info a_particle.position_ij
        particle_on = a_particle.on

        wind_i = (
            forcing_xy.u(a_particle.position_xy[1], a_particle.position_xy[2]),
            forcing_xy.v(a_particle.position_xy[1], a_particle.position_xy[2]),
        )

        model.ParticleCollection[a_particle.position_ij[1], a_particle.position_ij[2]] = mapping_2D.advance!(
                a_particle, model.State, FailedCollection,
                model.grid, wind_i, Δt,
                model.ODEsettings.log_energy_maximum,
                model.ODEsettings.wind_min_squared,
                model.periodic_boundary,
                model.ODEdefaults,
                spline)

        # if (a_particle.position_ij[2] == 6) & (particle_on != a_particle.on)            
        #     @info "after advance! outside: $(a_particle.position_ij) particle on change :$(particle_on) to $(a_particle.on)"
        # end
    end

end

function time_step!_remesh(model::Abstract2DModel, Δt::Float64, forcing_xy)

    @threads for a_particle in model.ParticleCollection[model.ocean_points]
        particle_on = a_particle.on
        wind_i = (
            forcing_xy.u(a_particle.position_xy[1], a_particle.position_xy[2]),
            forcing_xy.v(a_particle.position_xy[1], a_particle.position_xy[2]),
        )
        model.ParticleCollection[a_particle.position_ij[1], a_particle.position_ij[2]] = mapping_2D.remesh!(
                        a_particle, model.State,
                        wind_i, model.clock.time, 
                        model.ODEsettings, Δt,
                        model.grid.stats, 
                        model.minimal_state,
                        model.ODEdefaults)
        
        # if (a_particle.position_ij[2] == 6) & (particle_on != a_particle.on)
        #     @info "after  remesh! outside: $(a_particle.position_ij) particle on change :$(particle_on) to $(a_particle.on)"
        # end
    end

end

#build wrapper
advance_wrapper(f, state, Fcol, grid, winds, dt, emax, windmin, boundary, defaults) = x -> f(x, state, Fcol, grid, winds, dt, emax, windmin, boundary, defaults)
remesh_wrapper(f, state, winds, time, sets, dt, minpar, minstate, defaults) = x -> f(x, state, winds, time, sets, dt, minpar, minstate, defaults)



"""
movie_time_step!(model, Δt; callbacks=nothing)

advances model by 1 time step:
1st) the model.ParticleCollection is advanced and then 
2nd) the model.State is updated.
clock is ticked by Δt

callbacks are not implimented yet

"""
function movie_time_step!(model::Abstract2DModel, Δt; forcing=nothing, callbacks=nothing, debug=false)

    # temporary FailedCollection to store failed particles
    FailedCollection = Vector{AbstractMarkedParticleInstance}([])

    current_forcing = forcing === nothing ? (hasproperty(model, :winds) ? getproperty(model, :winds) : nothing) : forcing
    current_forcing === nothing && error("2D movie_time_step! requires forcing data (pass `forcing=...` to Simulation or set `model.winds`)")
    forcing_xy = time_slice_forcing(current_forcing, model.grid, model.clock.time)

    # resolve B-spline deposition order Int -> Val{P} once (function barrier); see time_step!
    spline_val = Val(hasproperty(model, :spline_order) ? model.spline_order : 1)
    time_step!_advance(model, Δt, forcing_xy, FailedCollection, spline_val)

    model.MovieState = copy(model.State)

    if debug
        model.FailedCollection = FailedCollection
    end

    #@printf "re-mesh"
    time_step!_remesh(model, Δt, forcing_xy)
    
    model.State .= 0.0
    tick!(model.clock, Δt)
end


end
