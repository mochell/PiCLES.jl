
using PiCLES: FetchRelations as FR

plot_path_base = "plots/tests/T01_TemoporalFetchRelatoins/"
mkpath(plot_path_base)

FR.X_tilde_time_and_fetch
using Plots

t = range(60, step=5, length=720)

function plot_windsea_data(t, WS, WS2, WS2m)
    cmap = palette(:corkO, 20)
    t2 = t / 60 / 60
    plot_grid = plot(layout=(2, 2), size=(800, 600), suptitle="Windsea JONSWAP & PM for U10 = -10m/s to 15m/s")

    u_list = -10:2:14
    p_dict = Dict()
    for (uu, coll) in zip(u_list, cmap[1:end-1])
        plot!(plot_grid[1], t2, WS[uu][!, "E"], label=nothing, color=coll, linewidth= 2)
        plot!(plot_grid[2], t2, log.(WS[uu][!, "E"]), label=nothing, color=coll, linewidth= 2)
        plot!(plot_grid[3], t2, WS[uu][!, "cg_bar"], label=nothing, color=coll, linewidth= 2)
        plot!(plot_grid[4], t2, WS[uu][!, "m"], label=nothing, color=coll, linewidth= 2)
    end

    plot!(plot_grid[1], t2, WS2[!, "E"], label="U10 = +-2m/s", color="red", linestyle=:solid) 
    plot!(plot_grid[1], ylim=(0, 0.01), title="Energy", ylabel="Energy (m^2)", legend=:topright, grid=true)
    plot!(plot_grid[2], t2, log.(WS2[!, "E"]), label="U10 = +-2m/s", color="red", linestyle=:solid, title="log(Energy)", ylabel="log(Energy)", grid=true)
    plot!(plot_grid[3], t2, WS2[!, "cg_bar"], label="U10 = +-2m/s", color="red", linestyle=:solid, title="cg_bar", ylabel="Cg_bar (m/s)", xlabel="Time scale (hours)", grid=true)
    plot!(plot_grid[4], t2, WS2[!, "m"], label="U10 = +-2m/s", color="red", linestyle=:solid, title="Momentum", ylabel="momentum", xlabel="Time scale (hours)", ylim=(-0.002, 0.002), grid=true)
    # negative windsea with -2 m/s
    plot!(plot_grid[1], t2, WS2m[!, "E"], label=nothing, color="red", linestyle=:solid)
    plot!(plot_grid[2], t2, log.(WS2m[!, "E"]), label=nothing, color="red", linestyle=:solid)
    plot!(plot_grid[3], t2, WS2m[!, "cg_bar"], label=nothing, color="red", linestyle=:solid)
    plot!(plot_grid[4], t2, WS2m[!, "m"], label=nothing, color="red", linestyle=:solid)

    plot(plot_grid)
end
# %%
using DataFrames

# Example usage:
t = range(60, step=5, length=720)
WS = Dict{Float64, Any}()
for uu in range(-10, step=2, stop=14)
    WS1 = FR.get_initial_windsea.(uu, t, "JONSWAP")
    WS1 = vcat(DataFrame.(WS1)...)
    WS[uu] =WS1
    #push!(WS, WS1)    
end
WS

WS2 = vcat(DataFrame.(FR.get_initial_windsea.(2, t))...)
WS2m = vcat(DataFrame.(FR.get_initial_windsea.(-2, t))...)

WSPM = Dict{Float64, Any}()
for uu in range(-10, step=2, stop=14)
    WS1 = FR.get_initial_windsea.(10.0, t, "PM")
    WS1 = vcat(DataFrame.(WS1)...)
    WSPM[uu] = WS1
end
plot_windsea_data(t, WS, WS2, WS2m)
savefig(joinpath([plot_path_base, "T01_FR_1D_compare.png"]))

# %% compare with 2D version
function plot_windsea_data_compare(WS1d, WS2d, u10, v10)

    case_string = "u10 = $u10 m/s, v10 = $v10 m/s"
    case_string1d = "|U| = $uabs m/s"

    plot_grid = plot(layout=(2, 2), size=(800, 600), subtitle="Windsea JONSWAP & PM for U10 = -10m/s to 15m/s")

    plot!(plot_grid[1], WS1d[!, "E"], label="1D, " * case_string1d, title="Energy | " * case_string, ylabel="Energy (m^2)", legend=:topright, grid=true, linewidth=3, color="black", alpha=0.2)
    plot!(plot_grid[1], WS2d[!, "E"], label="2D, " * case_string, linewidth=1.5, color="red")

    plot!(plot_grid[2], log.(WS1d[!, "E"]), label="1D, " * case_string1d, title="log(Energy)", ylabel="log(Energy)", grid=true, linewidth=3, color="black", alpha=0.2)
    plot!(plot_grid[2], log.(WS2d[!, "E"]), label="2D, " * case_string, linewidth=1.5, color="red")

    plot!(plot_grid[3], WS1d[!, "cg_bar"], label="1D, " * case_string1d, title="cg_bar", ylabel="Cg_bar (m/s)", xlabel="Time scale (hours)", grid=true, linewidth=4, color="black", alpha=0.2)
    plot!(plot_grid[3], WS2d[!, "cg_bar"], label="2D, " * case_string, linewidth=1.5, color="red")
    plot!(plot_grid[3], WS2d[!, "cg_bar_x"], label="2D, x, " * case_string, linewidth=1.5, color="blue")
    plot!(plot_grid[3], WS2d[!, "cg_bar_y"], label="2D, y, " * case_string, linewidth=1.5, color="green")

    plot!(plot_grid[4], WS1d[!, "m"], label="1D, " * case_string1d, title="Momentum", ylabel="momentum", xlabel="Time scale (hours)", ylim=(-0.001, 0.001), grid=true, linewidth=3, color="black", alpha=0.2)
    plot!(plot_grid[4], WS2d[!, "m_x"], label="2D m_x, " * case_string, linewidth=1.5, color="blue")
    plot!(plot_grid[4], WS2d[!, "m_y"], label="2D m_y, " * case_string, linewidth=1.5, color="green")
    plot(plot_grid)
end

u10, v10 = -2, 0
uabs = sqrt(u10^2 + v10^2)
WS1d = vcat(DataFrame.(FR.get_initial_windsea.(uabs, t))...)
WS2d = vcat(DataFrame.(FR.get_initial_windsea.(u10, v10, t))...)

plot_windsea_data_compare(WS1d, WS2d, u10, v10)

plot_string = "u10_$u10"* "_v10_$v10"
savefig(joinpath([plot_path_base, "T01_FR_" * plot_string  * ".png"]))


u10, v10 = -2, 2
uabs = sqrt(u10^2 + v10^2)
WS1d = vcat(DataFrame.(FR.get_initial_windsea.(uabs, t))...)
WS2d = vcat(DataFrame.(FR.get_initial_windsea.(u10, v10, t))...)

plot_windsea_data_compare(WS1d, WS2d, u10, v10)
plot_string = "u10_$u10" * "_v10_$v10"
savefig(joinpath([plot_path_base, "T01_FR_" * plot_string * ".png"]))


u10, v10 = -2, -2
uabs = sqrt(u10^2 + v10^2)
WS1d = vcat(DataFrame.(FR.get_initial_windsea.(uabs, t))...)
WS2d = vcat(DataFrame.(FR.get_initial_windsea.(u10, v10, t))...)

plot_windsea_data_compare(WS1d, WS2d, u10, v10)
plot_string = "u10_$u10" * "_v10_$v10"
savefig(joinpath([plot_path_base, "T01_FR_" * plot_string * ".png"]))

u10, v10 = 0, -2
uabs = sqrt(u10^2 + v10^2)
WS1d = vcat(DataFrame.(FR.get_initial_windsea.(uabs, t))...)
WS2d = vcat(DataFrame.(FR.get_initial_windsea.(u10, v10, t))...)

plot_windsea_data_compare(WS1d, WS2d, u10, v10)
plot_string = "u10_$u10" * "_v10_$v10"
savefig(joinpath([plot_path_base, "T01_FR_" * plot_string * ".png"]))

u10, v10 = 3, -5
uabs = sqrt(u10^2 + v10^2)
WS1d = vcat(DataFrame.(FR.get_initial_windsea.(uabs, t))...)
WS2d = vcat(DataFrame.(FR.get_initial_windsea.(u10, v10, t))...)

plot_windsea_data_compare(WS1d, WS2d, u10, v10)
plot_string = "u10_$u10" * "_v10_$v10"
savefig(joinpath([plot_path_base, "T01_FR_" * plot_string * ".png"]))

