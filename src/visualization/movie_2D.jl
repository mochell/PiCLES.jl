module movie


using CairoMakie
using ...ParticleMesh: TwoDGrid, TwoDGridNotes, TwoDGridMesh
using ...Operators.core_2D: GetGroupVelocity

using ...Architectures: Grid2D, CartesianGrid, CartesianGridStatistics, CartesianGrid2D, CartesianGrid1D, AbstractGridStatistics, AbstractGrid, StandardRegular2D_old,
SphericalGrid, SphericalGridStatistics, SphericalGrid2D

import Oceananigans.Utils: prettytime

function init_movie_2D_box_plot(wave_simulation; resolution=(900, 1200), name_string="", aspect=1, axline=0)

    n = Observable(1) # for visualization
    # Ocean vorticity
    grid = wave_simulation.model.grid
    if typeof(grid) <: TwoDGrid
        mesh = TwoDGridMesh(grid, skip=1)
        gn = TwoDGridNotes(grid)
        dx = gn.dx
    elseif typeof(grid) <: CartesianGrid
        mesh =grid.data
        gn = (x=grid.data.x[:, 1], y=grid.data.y[1, :])
        dx = grid.stats.dx
    end
    
    arrow_skip = 3
    arrow_skip_y =2

    model_time = @lift ($n; wave_simulation.model.clock.time)
    uo, vo = wave_simulation.model.winds
    # ocean_wind_u = @lift(uo.(grid.data.x[1:arrow_skip:end], grid.data.y[1:arrow_skip:end], $model_time))
    # ocean_wind_v = @lift(vo.(grid.data.x[1:arrow_skip:end], grid.data.y[1:arrow_skip:end], $model_time))


    ocean_wind_u = @lift(uo.(grid.data.x[1:arrow_skip:end, 1:arrow_skip_y:end], 
                            grid.data.y[1:arrow_skip:end, 1:arrow_skip_y:end], $model_time))
    ocean_wind_v = @lift(vo.(grid.data.x[1:arrow_skip:end, 1:arrow_skip_y:end], 
                            grid.data.y[1:arrow_skip:end, 1:arrow_skip_y:end], $model_time))

    #ocean_wind = @lift ($n; sqrt.(ocean_wind_u.^2 + ocean_wind_u.^2))
    #ocean_wind = @lift($n; sqrt(vo.(grid.data.x, grid.data.y, $model_time)^2 + uo.(grid.data.x, grid.data.y, $model_time)^2))
    ocean_wind = @lift(sqrt.(vo.(grid.data.x, grid.data.y, $model_time).^2 + uo.(grid.data.x, grid.data.y, $model_time).^2))
    strength = @lift( vec(sqrt.(vo.(grid.data.x, grid.data.y, $model_time).^2 + uo.(grid.data.x, grid.data.y, $model_time).^2)))
    #ocean_wind = @lift ($n; model_time * 2)

    #@info ocean_wind

    wave_energy = @lift ($n; 4 * sqrt.(wave_simulation.model.MovieState[:,:, 1]))
    wave_momentum_x = @lift ($n; wave_simulation.model.MovieState[:, :, 2])
    wave_momentum_y = @lift ($n; wave_simulation.model.MovieState[:, :, 3])
    cx = @lift ($n; GetGroupVelocity(wave_simulation.model.MovieState).c_x)
    cy = @lift ($n; GetGroupVelocity(wave_simulation.model.MovieState).c_y)

    #c_max = @lift ($n; sqrt(GetGroupVelocity(wave_simulation.model.MovieState).c_y^2 + GetGroupVelocity(wave_simulation.model.MovieState).c_x)^2 )
    #c_max = round(maximum(sqrt(GetGroupVelocity(wave_simulation.model.MovieState).c_y^2 + GetGroupVelocity(wave_simulation.model.MovieState).c_x)^2))

    # for testing
    #we = wave_simulation.model.State[:, :, 1]

    # Make figure
    #x, y, z = nodes((Center, Center, Center), grid)
    fig = Figure(resolution=resolution)

    ax_wind = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="Winds")
    ax_o  = Axis(fig[1, 2], aspect=1, xlabel="x (km)",  title="Hs")
    ax_mx = Axis(fig[2, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="x momentum")
    ax_my = Axis(fig[2, 2], aspect=1, xlabel="x (km)",  title="y momentum")

    ax_cx = Axis(fig[3, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="c_x")
    ax_cy = Axis(fig[3, 2], aspect=1, xlabel="x (km)",  title="c_y")


    xx = grid.data.x[1:end, 1]
    yy = grid.data.y[1, 1:end]
    # for ax in (ax_wind, ax_o, ax_mx, ax_my, ax_cx, ax_cy)
    #     ax.aspect = aspect
    #     #vlines!(ax, [axline], color=:red)
    # end
    #ax_wind = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="Winds")
    #hm_i = heatmap!(ax_wind, 1e-3 * x, 1e-3 * y, ice_speed_n)
    hm_wind = heatmap!(ax_wind, 1e-3 * xx[1:3:end], 1e-3 * yy[1:3:end], ocean_wind, colormap=:dense, colorrange=(0, 11) )
    # colormap options for heatmap 

    #quiver!(ax_wind, 1e-3 * xx, 1e-3 * yy, quiver=(ocean_wind_u, ocean_wind_v))#, color=:red, scale_unit=:data, label="wind")
    #strength = ocean_wind_u.^2 .+ ocean_wind_v.^2
    arrows2d!(ax_wind, 1e-3 * xx[1:arrow_skip:end], 
                    1e-3 * yy[1:arrow_skip_y:end],
                    ocean_wind_u, 
                    ocean_wind_v, 
                    lengthscale=2, color=:green)
    #scatter!(ax_wind, vec(gridmesh.* 1e-3), rotations=0, markersize=20, marker='↑')


    hm_o = heatmap!(ax_o, 1e-3 * xx, 1e-3 * yy, wave_energy, colormap=:dense, colorrange=(0, 8))
    hm_x = heatmap!(ax_mx, 1e-3 * xx, 1e-3 * yy, wave_momentum_x, colormap=:balance, colorrange=(-0.1, 0.1))
    hm_y = heatmap!(ax_my, 1e-3 * xx, 1e-3 * yy, wave_momentum_y, colormap=:balance, colorrange=(-0.1, 0.1))

    hm_cx = heatmap!(ax_cx, 1e-3 * xx, 1e-3 * yy, cx, colormap=:balance, colorrange=(-5, 5))
    hm_cy = heatmap!(ax_cy, 1e-3 * xx, 1e-3 * yy, cy, colormap=:balance, colorrange=(-5, 5))
    #colormaps

    #colorbar(ax_my)
    # cb_wind = Colorbar(fig[1, 0], hm_wind, label="winds [m/s]", tickalignmode=:right)
    # cb_wind.alignmode = Mixed(right=1)
    # #Colorbar(fig[1, 3], hm_o, label="Wave energy [m^2]")

    Colorbar(fig[1, 0], hm_wind, label="winds [m/s]")
    Colorbar(fig[1, 4], hm_o, label = "Wave energy [m^2]")
    Colorbar(fig[2, 4], hm_x, label = "Wave momentum x []")
    Colorbar(fig[3, 4], hm_cx, label="Group Velocity [m/s]")

    limits!(ax_wind, (1e-3 * xx[1], 1e-3 * xx[end]), (1e-3 * yy[1], 1e-3 * yy[end]))
    
    for ax in (ax_wind, ax_o, ax_mx, ax_my, ax_cx, ax_cy)
        vlines!(ax, axline, color=:black)
        ax.aspect = aspect
    end

    #hm_o = heatmap(ax_o, 1e-3 * grid.data.x, 1e-3 * grid.data.y, ocean_wind_v, colormap=:redblue)
    DT = wave_simulation.model.ODEsettings.timestep
    #c_yi = GetGroupVelocity(wave_simulation.model.MovieState).c_y
    #c_xi = GetGroupVelocity(wave_simulation.model.MovieState).c_x
    #CFL = @lift ($n; round(maximum(sqrt(cx[]^2 + cx[]^2))) * DT / gn.dx)

    title = @lift ($n; "DT=$DT , dx=$(round(dx)), CFL= $(round( maximum(sqrt.(cx[].^2+cy[].^2)) * DT /dx; digits=3 )), \ntime=" * prettytime(wave_simulation.model.clock.time) * "\n" * name_string)

    Label(fig[0, :], title)
    #display(fig)

    return fig, n
end


function init_movie_2D_box_plot_small(wave_simulation; resolution=(1100, 900), name_string="", aspect=1, axline=0)

    n = Observable(1) # for visualization
    # Ocean vorticity
    grid = wave_simulation.model.grid
    # mesh = TwoDGridMesh(grid, skip=1)
    # gn = TwoDGridNotes(grid)

    arrow_skip   = 5
    arrow_skip_y = 5

    model_time = @lift ($n; wave_simulation.model.clock.time)
    uo, vo     = wave_simulation.model.winds
    # ocean_wind_u = @lift(uo.(grid.data.x[1:arrow_skip:end], grid.data.y[1:arrow_skip:end], $model_time))
    # ocean_wind_v = @lift(vo.(grid.data.x[1:arrow_skip:end], grid.data.y[1:arrow_skip:end], $model_time))

    ocean_wind_u = @lift(uo.(grid.data.x[1:arrow_skip:end, 1:arrow_skip_y:end],
        grid.data.y[1:arrow_skip:end, 1:arrow_skip_y:end], $model_time))
    ocean_wind_v = @lift(vo.(grid.data.x[1:arrow_skip:end, 1:arrow_skip_y:end],
        grid.data.y[1:arrow_skip:end, 1:arrow_skip_y:end], $model_time))

    #ocean_wind = @lift ($n; sqrt.(ocean_wind_u.^2 + ocean_wind_u.^2))
    #ocean_wind = @lift($n; sqrt(vo.(grid.data.x, grid.data.y, $model_time)^2 + uo.(grid.data.x, grid.data.y, $model_time)^2))
    ocean_wind = @lift(sqrt.(vo.(grid.data.x, grid.data.y, $model_time) .^ 2 + uo.(grid.data.x, grid.data.y, $model_time) .^ 2))
    strength = @lift(vec(sqrt.(vo.(grid.data.x, grid.data.y, $model_time) .^ 2 + uo.(grid.data.x, grid.data.y, $model_time) .^ 2)))
    #ocean_wind = @lift ($n; model_time * 2)

    #@info ocean_wind

    wave_energy = @lift ($n; 4 * sqrt.(wave_simulation.model.MovieState[:, :, 1]))
    #wave_momentum_x = @lift ($n; wave_simulation.model.MovieState[:, :, 2])
    #wave_momentum_y = @lift ($n; wave_simulation.model.MovieState[:, :, 3])
    cx = @lift ($n; GetGroupVelocity(wave_simulation.model.MovieState).c_x)
    cy = @lift ($n; GetGroupVelocity(wave_simulation.model.MovieState).c_y)

    #c_max = @lift ($n; sqrt(GetGroupVelocity(wave_simulation.model.MovieState).c_y^2 + GetGroupVelocity(wave_simulation.model.MovieState).c_x)^2 )
    #c_max = round(maximum(sqrt(GetGroupVelocity(wave_simulation.model.MovieState).c_y^2 + GetGroupVelocity(wave_simulation.model.MovieState).c_x)^2))

    # for testing
    #we = wave_simulation.model.State[:, :, 1]

    # Make figure
    #x, y, z = nodes((Center, Center, Center), grid)
    fig = Figure(resolution=resolution)

    ax_wind = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="Winds")
    ax_o =    Axis(fig[1, 2], aspect=1, xlabel="x (km)", title="Hs")
    #ax_mx = Axis(fig[2, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="x momentum")
    #ax_my = Axis(fig[2, 2], aspect=1, xlabel="x (km)", title="y momentum")

    ax_cx = Axis(fig[2, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="c_x")
    ax_cy = Axis(fig[2, 2], aspect=1, xlabel="x (km)", title="c_y")


    # for ax in (ax_wind, ax_o, ax_mx, ax_my, ax_cx, ax_cy)
    #     ax.aspect = aspect
    #     #vlines!(ax, [axline], color=:red)
    # end
    #ax_wind = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="Winds")
    #hm_i = heatmap!(ax_wind, 1e-3 * x, 1e-3 * y, ice_speed_n)
    xx = grid.data.x[1:end, 1]
    yy = grid.data.y[1, 1:end]
    hm_wind = heatmap!(ax_wind, 1e-3 * xx[1:3:end], 1e-3 * yy[1:3:end], ocean_wind, colormap=:dense, colorrange=(0, 11))
    # colormap options for heatmap 


    #quiver!(ax_wind, 1e-3 * xx, 1e-3 * yy, quiver=(ocean_wind_u, ocean_wind_v))#, color=:red, scale_unit=:data, label="wind")
    #strength = ocean_wind_u.^2 .+ ocean_wind_v.^2
    arrows2d!(ax_wind, 1e-3 * xx[1:arrow_skip:end],
        1e-3 * yy[1:arrow_skip_y:end],
        ocean_wind_u,
        ocean_wind_v,
        #size=10, 
        lengthscale=2, color=:black)
    #scatter!(ax_wind, vec(gridmesh.* 1e-3), rotations=0, markersize=20, marker='↑')


    hm_o = heatmap!(ax_o, 1e-3 * xx, 1e-3 * yy, wave_energy, colormap=:dense, colorrange=(0, 2))
    #hm_x = heatmap!(ax_mx, 1e-3 * xx, 1e-3 * yy, wave_momentum_x, colormap=:balance, colorrange=(-0.05, 0.05))
    #hm_y = heatmap!(ax_my, 1e-3 * xx, 1e-3 * yy, wave_momentum_y, colormap=:balance, colorrange=(-0.05, 0.05))

    hm_cx = heatmap!(ax_cx, 1e-3 * xx, 1e-3 * yy, cx, colormap=:balance, colorrange=(-5, 5))
    hm_cy = heatmap!(ax_cy, 1e-3 * xx, 1e-3 * yy, cy, colormap=:balance, colorrange=(-5, 5))
    #colormaps

    #colorbar(ax_my)
    # cb_wind = Colorbar(fig[1, 0], hm_wind, label="winds [m/s]", tickalign=:right)
    # cb_wind.alignmode = Mixed(right=1)
    #Colorbar(fig[1, 3], hm_o, label="Wave energy [m^2]")

    Colorbar(fig[1, 4], hm_o, label="Wave energy [m^2]")
    #Colorbar(fig[2, 4], hm_x, label="Wave momentum x []")
    Colorbar(fig[2, 4], hm_cx, label="Group Velocity [m/s]")

    limits!(ax_wind, (1e-3 * xx[1], 1e-3 * xx[end]), (1e-3 * yy[1], 1e-3 * yy[end]))

    for ax in (ax_wind, ax_o, ax_cx, ax_cy)
        vlines!(ax, axline, color=:black)
        ax.aspect = aspect
    end

    #hm_o = heatmap(ax_o, 1e-3 * grid.data.x, 1e-3 * grid.data.y, ocean_wind_v, colormap=:redblue)
    DT = wave_simulation.model.ODEsettings.timestep
    #c_yi = GetGroupVelocity(wave_simulation.model.MovieState).c_y
    #c_xi = GetGroupVelocity(wave_simulation.model.MovieState).c_x
    #CFL = @lift ($n; round(maximum(sqrt(cx[]^2 + cx[]^2))) * DT / gn.dx)

    # title = @lift ($n; "DT=$DT , dx=$(round(gn.dx)) \ntime=" * prettytime(wave_simulation.model.clock.time) * "\n" * name_string)

    title = @lift ($n; "time=" * prettytime(wave_simulation.model.clock.time) * "\n" * name_string)

    Label(fig[0, :], title)
    display(fig)

    return fig, n
end

function init_movie_2D_rectangle(wave_simulation; resolution=(900, 1200), name_string="", aspect=1, axline=0)

    n = Observable(1) # for visualization
    # Ocean vorticity
    grid = wave_simulation.model.grid
    if typeof(grid) <: TwoDGrid
        mesh = TwoDGridMesh(grid, skip=1)
        gn = TwoDGridNotes(grid)
        dx = gn.dx
    elseif typeof(grid) <: CartesianGrid
        mesh = grid.data
        gn = (x=grid.data.x[:, 1], y=grid.data.y[1, :])
        dx = grid.stats.dx
    elseif typeof(grid) <: SphericalGrid
        mesh = grid.data
        gn = (x=grid.data.x[:, 1], y=grid.data.y[1, :])
        dx = grid.stats.dx_deg
    end

    arrow_skip = 10
    arrow_skip_y = 10

    model_time = @lift ($n; wave_simulation.model.clock.time)
    uo, vo = wave_simulation.model.winds
    # ocean_wind_u = @lift(uo.(grid.data.x[1:arrow_skip:end], grid.data.y[1:arrow_skip:end], $model_time))
    # ocean_wind_v = @lift(vo.(grid.data.x[1:arrow_skip:end], grid.data.y[1:arrow_skip:end], $model_time))


    ocean_wind_u = @lift(uo.(grid.data.x[1:arrow_skip:end, 1:arrow_skip_y:end],
        grid.data.y[1:arrow_skip:end, 1:arrow_skip_y:end], $model_time))
    ocean_wind_v = @lift(vo.(grid.data.x[1:arrow_skip:end, 1:arrow_skip_y:end],
        grid.data.y[1:arrow_skip:end, 1:arrow_skip_y:end], $model_time))

    #ocean_wind = @lift ($n; sqrt.(ocean_wind_u.^2 + ocean_wind_u.^2))
    #ocean_wind = @lift($n; sqrt(vo.(grid.data.x, grid.data.y, $model_time)^2 + uo.(grid.data.x, grid.data.y, $model_time)^2))
    ocean_wind = @lift(sqrt.(vo.(grid.data.x, grid.data.y, $model_time) .^ 2 + uo.(grid.data.x, grid.data.y, $model_time) .^ 2))
    strength = @lift(vec(sqrt.(vo.(grid.data.x, grid.data.y, $model_time) .^ 2 + uo.(grid.data.x, grid.data.y, $model_time) .^ 2)))

    # ocean_wind_max = @lift (maximum.(sqrt.(vo.(grid.data.x, grid.data.y, $model_time) .^ 2 + uo.(grid.data.x, grid.data.y, $model_time) .^ 2)))
    # ocean_wind_max = 20
    

    #@info ocean_wind

    wave_energy = @lift ($n; 4 * sqrt.(wave_simulation.model.MovieState[:, :, 1]))
    wave_momentum_x = @lift ($n; wave_simulation.model.MovieState[:, :, 2])
    wave_momentum_y = @lift ($n; wave_simulation.model.MovieState[:, :, 3])
    cx = @lift ($n; GetGroupVelocity(wave_simulation.model.MovieState).c_x)
    cy = @lift ($n; GetGroupVelocity(wave_simulation.model.MovieState).c_y)

    #c_max = @lift ($n; sqrt(GetGroupVelocity(wave_simulation.model.MovieState).c_y^2 + GetGroupVelocity(wave_simulation.model.MovieState).c_x)^2 )
    #c_max = round(maximum(sqrt(GetGroupVelocity(wave_simulation.model.MovieState).c_y^2 + GetGroupVelocity(wave_simulation.model.MovieState).c_x)^2))

    # for testing
    #we = wave_simulation.model.State[:, :, 1]

    # Make figure
    #x, y, z = nodes((Center, Center, Center), grid)
    fig = Figure(size=resolution)

    ax_wind = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="Winds")
    ax_o = Axis(fig[1, 2], aspect=1, xlabel="x (km)", title="Hs")
    ax_mx = Axis(fig[2, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="x momentum")
    ax_my = Axis(fig[2, 2], aspect=1, xlabel="x (km)", title="y momentum")

    ax_cx = Axis(fig[3, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="c_x")
    ax_cy = Axis(fig[3, 2], aspect=1, xlabel="x (km)", title="c_y")


    xx = grid.data.x[1:end, 1]
    yy = grid.data.y[1, 1:end]
    # for ax in (ax_wind, ax_o, ax_mx, ax_my, ax_cx, ax_cy)
    #     ax.aspect = aspect
    #     #vlines!(ax, [axline], color=:red)
    # end
    #ax_wind = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="Winds")
    #hm_i = heatmap!(ax_wind, 1e-3 * x, 1e-3 * y, ice_speed_n)
    ocean_wind_max = 50
    hm_wind = heatmap!(ax_wind, 1e-3 * xx, 1e-3 * yy, ocean_wind, colormap=:dense, colorrange=(0, ocean_wind_max))
    # colormap options for heatmap 

    #quiver!(ax_wind, 1e-3 * xx, 1e-3 * yy, quiver=(ocean_wind_u, ocean_wind_v))#, color=:red, scale_unit=:data, label="wind")
    #strength = ocean_wind_u.^2 .+ ocean_wind_v.^2
    arrows2d!(ax_wind, 1e-3 * xx[1:arrow_skip:end],
        1e-3 * yy[1:arrow_skip_y:end],
        ocean_wind_u,
        ocean_wind_v,
        lengthscale=2, color=:green)
    #scatter!(ax_wind, vec(gridmesh.* 1e-3), rotations=0, markersize=20, marker='↑')


    hm_o = heatmap!(ax_o, 1e-3 * xx, 1e-3 * yy, wave_energy, colormap=:dense, colorrange=(0, 16))
    hm_x = heatmap!(ax_mx, 1e-3 * xx, 1e-3 * yy, wave_momentum_x, colormap=:balance, colorrange=(-0.1, 0.1))
    hm_y = heatmap!(ax_my, 1e-3 * xx, 1e-3 * yy, wave_momentum_y, colormap=:balance, colorrange=(-0.1, 0.1))

    hm_cx = heatmap!(ax_cx, 1e-3 * xx, 1e-3 * yy, cx, colormap=:balance, colorrange=(-10, 10))
    hm_cy = heatmap!(ax_cy, 1e-3 * xx, 1e-3 * yy, cy, colormap=:balance, colorrange=(-10, 10))
    #colormaps

    #colorbar(ax_my)
    # cb_wind = Colorbar(fig[1, 0], hm_wind, label="winds [m/s]", tickalignmode=:right)
    # cb_wind.alignmode = Mixed(right=1)
    # #Colorbar(fig[1, 3], hm_o, label="Wave energy [m^2]")

    Colorbar(fig[1, 0], hm_wind, label="winds [m/s]")
    Colorbar(fig[1, 3], hm_o, label="Wave energy [m^2]")
    Colorbar(fig[2, 3], hm_x, label="Wave momentum x []")
    Colorbar(fig[3, 3], hm_cx, label="Group Velocity [m/s]")

    wind_lim = 600
    limits!(ax_wind, (-wind_lim, wind_lim), (-wind_lim, wind_lim))

    hs_lim = 2000
    limits!(ax_o, (-hs_lim, hs_lim), (-hs_lim, hs_lim))

    for ax in (ax_wind, ax_o, ax_mx, ax_my, ax_cx, ax_cy)
        vlines!(ax, axline, color=:black)
        ax.aspect = aspect
    end

    #hm_o = heatmap(ax_o, 1e-3 * grid.data.x, 1e-3 * grid.data.y, ocean_wind_v, colormap=:redblue)
    DT = wave_simulation.model.ODEsettings.timestep
    #c_yi = GetGroupVelocity(wave_simulation.model.MovieState).c_y
    #c_xi = GetGroupVelocity(wave_simulation.model.MovieState).c_x
    #CFL = @lift ($n; round(maximum(sqrt(cx[]^2 + cx[]^2))) * DT / gn.dx)

    title = @lift ($n; "DT=$DT , dx=$(round(dx)), CFL= $(round( maximum(sqrt.(cx[].^2+cy[].^2)) * DT /dx; digits=3 )), \ntime=" * prettytime(wave_simulation.model.clock.time) * "\n" * name_string)

    Label(fig[0, :], title)
    #display(fig)

    return fig, n
end


function init_movie_2D_simple(wave_simulation; resolution=(900, 1200), name_string="", aspect=1)

    n = Observable(1) # for visualization
    # Ocean vorticity
    grid = wave_simulation.model.grid
    if typeof(grid) <: TwoDGrid
        mesh = TwoDGridMesh(grid, skip=1)
        gn = TwoDGridNotes(grid)
        dx = gn.dx
        space_units = "km"
        space_scaler = 1e-3

    elseif typeof(grid) <: CartesianGrid
        mesh = grid.data
        gn = (x=grid.data.x[:, 1], y=grid.data.y[1, :])
        dx = grid.stats.dx
        space_units = "km"
        space_scaler = 1e-3

    elseif typeof(grid) <: SphericalGrid
        mesh = grid.data
        gn = (x=grid.data.x[:, 1], y=grid.data.y[1, :])
        dx = grid.stats.dx_deg
        space_units = "degrees"
        space_scaler = 1.0
    end

    arrow_skip = 5
    arrow_skip_y = arrow_skip

    model_time = @lift ($n; wave_simulation.model.clock.time)
    uo, vo = wave_simulation.model.winds
    # ocean_wind_u = @lift(uo.(grid.data.x[1:arrow_skip:end], grid.data.y[1:arrow_skip:end], $model_time))
    # ocean_wind_v = @lift(vo.(grid.data.x[1:arrow_skip:end], grid.data.y[1:arrow_skip:end], $model_time))


    ocean_wind_u = @lift(uo.(grid.data.x[1:arrow_skip:end, 1:arrow_skip_y:end],
        grid.data.y[1:arrow_skip:end, 1:arrow_skip_y:end], $model_time))
    ocean_wind_v = @lift(vo.(grid.data.x[1:arrow_skip:end, 1:arrow_skip_y:end],
        grid.data.y[1:arrow_skip:end, 1:arrow_skip_y:end], $model_time))

    #ocean_wind = @lift ($n; sqrt.(ocean_wind_u.^2 + ocean_wind_u.^2))
    #ocean_wind = @lift($n; sqrt(vo.(grid.data.x, grid.data.y, $model_time)^2 + uo.(grid.data.x, grid.data.y, $model_time)^2))
    ocean_wind = @lift(sqrt.(vo.(grid.data.x, grid.data.y, $model_time) .^ 2 + uo.(grid.data.x, grid.data.y, $model_time) .^ 2))
    strength = @lift(vec(sqrt.(vo.(grid.data.x, grid.data.y, $model_time) .^ 2 + uo.(grid.data.x, grid.data.y, $model_time) .^ 2)))


    # particle_on = @lift ($n; wave_simulation.model.ParticleCollection.on)
    # ocean_wind_max = @lift (maximum.(sqrt.(vo.(grid.data.x, grid.data.y, $model_time) .^ 2 + uo.(grid.data.x, grid.data.y, $model_time) .^ 2)))
    # ocean_wind_max = 20


    #@info ocean_wind

    wave_energy = @lift ($n; 4 * sqrt.(wave_simulation.model.MovieState[:, :, 1]))
    #wave_momentum_x = @lift ($n; wave_simulation.model.MovieState[:, :, 2])
    #wave_momentum_y = @lift ($n; wave_simulation.model.MovieState[:, :, 3])
    cx = @lift ($n; GetGroupVelocity(wave_simulation.model.MovieState).c_x)
    cy = @lift ($n; GetGroupVelocity(wave_simulation.model.MovieState).c_y)

    #c_max = @lift ($n; sqrt(GetGroupVelocity(wave_simulation.model.MovieState).c_y^2 + GetGroupVelocity(wave_simulation.model.MovieState).c_x)^2 )
    #c_max = round(maximum(sqrt(GetGroupVelocity(wave_simulation.model.MovieState).c_y^2 + GetGroupVelocity(wave_simulation.model.MovieState).c_x)^2))

    # for testing
    #we = wave_simulation.model.State[:, :, 1]

    # Make figure
    #x, y, z = nodes((Center, Center, Center), grid)
    fig = Figure(size=resolution)

    ax_wind = Axis(fig[1, 1], aspect=1, xlabel="x ($space_units)", ylabel="y ($space_units)", title="Winds")
    ax_o = Axis(fig[1, 2], aspect=1, xlabel="x ($space_units)", title="Hs")
    ax_cx = Axis(fig[2, 1], aspect=1, xlabel="x ($space_units)", ylabel="y ($space_units)", title="x momentum")
    ax_cy = Axis(fig[2, 2], aspect=1, xlabel="x ($space_units)", title="y momentum")

    # ax_cx = Axis(fig[3, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="c_x")
    # ax_cy = Axis(fig[3, 2], aspect=1, xlabel="x (km)", title="c_y")


    xx = grid.data.x[1:end, 1]
    yy = grid.data.y[1, 1:end]


    #ax_wind = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="Winds")
    #hm_i = heatmap!(ax_wind, 1e-3 * x, 1e-3 * y, ice_speed_n)
    ocean_wind_max = 50
    hm_wind = heatmap!(ax_wind, space_scaler * xx, space_scaler * yy, ocean_wind, colormap=:dense, colorrange=(0, ocean_wind_max))

    # hm_p_on = heatmap!(ax_wind, space_scaler * xx, space_scaler * yy, particle_on, colormap=:viridis, colorrange=(0, 0.5))

    # colormap options for heatmap 

    #quiver!(ax_wind, 1e-3 * xx, 1e-3 * yy, quiver=(ocean_wind_u, ocean_wind_v))#, color=:red, scale_unit=:data, label="wind")
    #strength = ocean_wind_u.^2 .+ ocean_wind_v.^2

    # arrows2d!(ax_wind, space_scaler * xx[1:arrow_skip:end],
    #     space_scaler * yy[1:arrow_skip_y:end],
    #     ocean_wind_u,
    #     ocean_wind_v,
    #     lengthscale=0.5, color=:gray, 
    #     alpha=0.2,
    #     )
    
    #scatter!(ax_wind, vec(gridmesh.* 1e-3), rotations=0, markersize=20, marker='↑')


    hm_o = heatmap!(ax_o, space_scaler * xx, space_scaler * yy, wave_energy, colormap=:dense, colorrange=(0, 12))
    # hm_x = heatmap!(ax_mx, space_scaler * xx, space_scaler * yy, wave_momentum_x, colormap=:balance, colorrange=(-0.1, 0.1))
    # hm_y = heatmap!(ax_my, space_scaler * xx, space_scaler * yy, wave_momentum_y, colormap=:balance, colorrange=(-0.1, 0.1))

    hm_cx = heatmap!(ax_cx, space_scaler * xx, space_scaler * yy, cx, colormap=:balance, colorrange=(-10, 10))
    hm_cy = heatmap!(ax_cy, space_scaler * xx, space_scaler * yy, cy, colormap=:balance, colorrange=(-10, 10))
    #colormaps

    #colorbar(ax_my)
    # cb_wind = Colorbar(fig[1, 0], hm_wind, label="winds [m/s]", tickalignmode=:right)
    # cb_wind.alignmode = Mixed(right=1)
    # #Colorbar(fig[1, 3], hm_o, label="Wave energy [m^2]")

    Colorbar(fig[1, 0], hm_wind, label="winds [m/s]")
    Colorbar(fig[1, 3], hm_o, label="Wave Hs [m^2]")
    # Colorbar(fig[2, 3], hm_x, label="Wave momentum x []")
    Colorbar(fig[2, 3], hm_cx, label="Group Velocity [m/s]")

    # wind_lim = 200
    # limits!(ax_wind, (-wind_lim, wind_lim), (-wind_lim, wind_lim))
    
    for ax in (ax_wind, ax_o, ax_cx, ax_cy)
        ax.aspect = aspect
    end

    #hm_o = heatmap(ax_o, 1e-3 * grid.data.x, 1e-3 * grid.data.y, ocean_wind_v, colormap=:redblue)
    DT = wave_simulation.model.ODEsettings.timestep
    #c_yi = GetGroupVelocity(wave_simulation.model.MovieState).c_y
    #c_xi = GetGroupVelocity(wave_simulation.model.MovieState).c_x
    CFL = @lift ($n ; round(maximum(sqrt.(cx[].^2 + cx[].^2))) * DT / dx)


    title = @lift ($n; "DT=$DT , dx=$(round(dx, digits=3)), CFL= $(CFL), \ntime=" * prettytime(wave_simulation.model.clock.time) * "\n" * name_string)

    Label(fig[0, :], title)
    #display(fig)

    return fig, n
end



end # end of module