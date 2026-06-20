module WaveGrowthModels1D

export WaveGrowth1D

using ...Architectures: AbstractODESettings
using ...Grids.CartesianGrid: CartesianGridMesh1D
using ..WaveGrowthModels2D: WaveGrowth2D

# A 1D wave-growth run is just a 2D run on a thin, periodic-in-y `CartesianGridMesh1D` with a
# zero meridional wind. There is no separate 1D model / core / mapping: `WaveGrowth1D` simply
# wraps the user's 1D wind and forwards to `WaveGrowth2D`, so it returns a `WaveGrowth2D` that
# the existing engine (core_2D / mapping_2D / time_step!) advances unchanged. Output helpers
# (see `Grids.CartesianGrid.gridnotes_1d` and `Simulations.convert_store_to_tuple`) squeeze the
# singleton y-dimension back to 1D.

"""
    _wrap_1d_winds(winds)

Turn a 1D wind into the `(u, v)` named tuple the 2D engine expects, with `v ≡ 0`. Accepts a
function with signature `u(x, t)` or `u(x, y, t)`, an already-formed `(u, v)` named tuple
(passed through unchanged), or `nothing`.
"""
function _wrap_1d_winds(winds::Function)
    u2(x, y, t) = applicable(winds, x, y, t) ? winds(x, y, t) : winds(x, t)
    v2(x, y, t) = 0.0
    return (u=u2, v=v2)
end
_wrap_1d_winds(winds::NamedTuple) = winds
_wrap_1d_winds(::Nothing) = nothing

"""
    WaveGrowth1D(; grid::CartesianGridMesh1D, winds, ODEsys, ODEsets, kwargs...)

Construct a 1D wave-growth model. This returns a `WaveGrowth2D` built on the thin 2D mesh
`grid`, with the 1D `winds` wrapped so the meridional component is zero. All remaining keyword
arguments are forwarded to [`WaveGrowth2D`](@ref).

`winds` is a 1D wind function `u(x, t)` (or `u(x, y, t)`). For an explicit `ODEinit_type`
particle default, use a 5-component `ParticleDefaults` with `c̄_y = 0` and `y` on the chosen row.
"""
function WaveGrowth1D(; grid::CartesianGridMesh1D,
    winds,
    ODEsys,
    ODEvars=nothing,
    layers::Int=1,
    ODEsets::AbstractODESettings,
    ODEinit_type="wind_sea",
    minimal_particle=nothing,
    minimal_state=nothing,
    currents=nothing,
    periodic_boundary=true,
    boundary_type="same",
    CBsets=nothing,
    spline_order::Int=1)

    return WaveGrowth2D(;
        grid=grid,
        winds=_wrap_1d_winds(winds),
        ODEsys=ODEsys,
        ODEvars=ODEvars,
        layers=layers,
        ODEsets=ODEsets,
        ODEinit_type=ODEinit_type,
        minimal_particle=minimal_particle,
        minimal_state=minimal_state,
        currents=currents,
        periodic_boundary=periodic_boundary,
        boundary_type=boundary_type,
        CBsets=CBsets,
        spline_order=spline_order)
end

# end of module
end
