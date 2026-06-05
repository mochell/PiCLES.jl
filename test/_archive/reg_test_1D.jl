ENV["JULIA_INCREMENTAL_COMPILE"]=true

#using Plots
import Plots as plt
using Setfield

import particle_waves_v4#: particle_equations, ODESettings
PW4 = particle_waves_v4

import FetchRelations, ParticleTools
using core_1D: ParticleDefaults, InitParticleInstance, InitParticleValues
using TimeSteppers
using PiCLES.ParticleMesh: OneDGrid, OneDGridNotes
using WaveGrowthModels1D

using Simulations

using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans: fields
using Oceananigans.Units
import Oceananigans.Utils: prettytime

using PiCLES.Architectures
#using GLMakie

# %%
save_path = "data/1D_test/"
mkpath(save_path)

# % Parameters
U10               = 5.0
dt_ODE_save       = 60 * 30 # 3 min
DT                = Float64(60 * 60) * 12 # seconds
T                 = 6 * 24 * 60 * 60 # seconds

# version 3


# function to define constants 
Const_ID = PW4.get_I_D_constant()
#
Const_Scg = PW4.get_Scg_constants(C_alpha=-1.41, C_varphi=1.81e-5)

Const_ID.q * Const_ID.c_alpha^(-10) * Const_ID.c_e^(-2) /2


@register_symbolic u(x, t)
u(x, t) = U10 
winds = (u=u);

# define variables based on particle equation
t, x, y, c̄_x, c̄_y, lne, Δn, Δφ_p, r_g, C_α, C_φ, g, C_e = vars =  PW4.init_vars();

particle_equations = PW4.particle_equations(u, γ=Const_ID.γ, q=Const_ID.q, input=true , dissipation=true, propagation=true, peak_shift=true, info=true)
@named particle_system = ODESystem(particle_equations);

# define V4 parameters absed on Const NamedTuple:
default_ODE_parameters = Dict(      r_g => r_g0, C_α => Const_Scg.C_alpha, 
                                    C_φ => Const_ID.c_β, C_e => Const_ID.C_e);


# define setting and standard initial conditions
cg_u_local                  = FetchRelations.c_g_U_tau(U10, DT) / 1
lne_local =  log(FetchRelations.Eⱼ(0.5*abs(U10), DT))
#lne_local =  log((0.5/4)^2)#log(FetchRelations.Eⱼ(0.1*abs(U10), DT))



ODE_settings    = PW4.ODESettings(
    Parameters=default_ODE_parameters,
    # define mininum energy threshold
    log_energy_minimum=lne_local,#log(FetchRelations.Eⱼ(0.1, DT)),
    #maximum energy threshold
    log_energy_maximum=log(17),  # correcsponds to Hs about 16 m
    saving_step=dt_ODE_save,
    timestep=DT,
    total_time=T,
    callbacks=nothing,
    save_everystep=false)

particle_defaults = ParticleDefaults(ODE_settings.log_energy_minimum, cg_u_local , 0.0)


# % Define grid
grid = OneDGrid(0, 100e3, 20)
periodic_boundary =false;
#bb = mark_boundary(grid)

### build model
wave_model = WaveGrowthModels1D.WaveGrowth1D( ;grid = grid, 
                                winds             = winds,
                                ODEsys            = particle_system,
                                ODEvars           = vars,
                                layers            = 1, 
                                ODEsets           = ODE_settings,  # ODE_settings
                                ODEdefaults       = particle_defaults,  # default_ODE_parameters
                                periodic_boundary = periodic_boundary)

Revise.retry()

### build Simulation
wave_simulation = Simulation(wave_model, Δt=10minutes, stop_time=4hours)
initialize_simulation!(wave_simulation)#; defaults=copy(particle_defaults))


# init_state_store!(wave_simulation, save_path)
# reset_state_store!(wave_simulation; value=-2.0)

#reset_simulation!(wave_simulation; defaults=nothing)#copy(particle_defaults))
# run simulation
run!(wave_simulation, cash_store=true)

# %%
#show_stored_data(wave_simulation)
#store_waves_data = wave_simulation.store.store["data"]

get_1d_data(store::Vector{Any}) =  permutedims(cat(store..., dims=3), (3, 1, 2)) 


store_waves_data = permutedims(cat(wave_simulation.store.store..., dims=3), (3, 1, 2));

energy = store_waves_data[:, :, 1] # State_collect[end-1][:,1] #GP.sel(state= 0 ).T # energy
m_x = store_waves_data[:, :, 2] # State_collect[end-1][:,2] #GP.sel(state= 1 ).T # energy
cg = energy ./ m_x ./ 2 # c_g


p1 = plt.heatmap( 4 * sqrt.(store_waves_data[:, 1:end, 1]), title="Hs", colormap=:tempo)
p2 = plt.heatmap(store_waves_data[:, 2:end, 2], title="wave momentum", colormap=:tempo)
# add panel for group velocity
p3 = plt.heatmap(cg[:, 1:end], title="group velocity", colormap=:tempo)

plt.plot(p1, p2, p3, layout=(1, 3), legend=true, size=(1200, 400))

# %%


#wave_simulation.stop_time += 1hour

# %%
#pcolormesh in Julia 
gn = OneDGridNotes(grid);


# gridmesh = [(i,j) for i in gn.x, j in gn.y];
# mesh.x = [i for i in gn.x, j in gn.y];
# mesh.y = [j for i in gn.x, j in gn.y];
# mesh.x = mesh.x[1:3:end, 1:3:end];
# mesh.y = mesh.y[1:3:end, 1:3:end];

p1 = plt.heatmap(gn.x / 1e3, wave_simulation.model.State[:, 1]) 
# plt.quiver!(p1, mesh.x / 1e3, mesh.y / 1e3, quiver=(wave_simulation.model.State[1:3:end, 1:3:end, 2] * 20, wave_simulation.model.State[1:3:end, 1:3:end, 3] * 20), color=:red, scale_unit=:data, label="wind")
# # scale_unit options: :width, :height, :x, :y, :xy, :data

# #plt.heatmap(p1, gn.x / 1e3, gn.y / 1e3, u.(mesh.x, mesh.y, 0))

#plt.heatmap(wave_simulation.model.State[:, 2])
# # heatmap(wave_simulation.model.State[:, :, 3])

plt.plot(wave_simulation.model.State[:, 2])

# # %%
# # use GLMakie to plot model.State each timestep 


# %%
reset_simulation!(wave_simulation)

n = Observable(1) # for visualization
# Ocean vorticity

model_time = @lift ($n; wave_simulation.model.clock.time)
uo, vo = wave_simulation.model.winds
ocean_wind_u = @lift(uo.(mesh.x, mesh.y, $model_time))
ocean_wind_v = @lift(vo.(mesh.x, mesh.y, $model_time))
#ocean_wind = @lift(  sqrt( vo.(mesh.x, mesh.y, $model_time)^2 + uo.(mesh.x, mesh.y, $model_time)^2) )
#ocean_wind = @lift(sqrt(vo.(mesh.x, mesh.y)^2 + uo.(mesh.x, mesh.y)^2)($model_time) )


wave_energy = @lift ($n; 4 * sqrt.(wave_simulation.model.State[:,:, 1]))
wave_momentum_x = @lift ($n; wave_simulation.model.State[:,:, 2])
wave_momentum_y = @lift ($n; wave_simulation.model.State[:,:, 3])

# for testing
#we = wave_simulation.model.State[:, :, 1]


#compute!(wave_simulation.model.State)
# Ice speed
# ui, vi = ice_model.velocities
# ice_speed = Field(sqrt(ui^2 + vi^2))
# ice_speed_n = @lift ($n; interior(compute!(ice_speed))[:, :, 1])

# Make figure
#x, y, z = nodes((Center, Center, Center), grid)
fig = Figure(resolution=(800, 900))

ax_wind = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="Winds")
ax_o = Axis(fig[1, 2], aspect=1, xlabel="x (km)", ylabel="y (km)", title="Hs")
ax_mx = Axis(fig[2, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="x momentum")
ax_my = Axis(fig[2, 2], aspect=1, xlabel="x (km)", ylabel="y (km)", title="y momentum")

#hm_i = heatmap!(ax_wind, 1e-3 * x, 1e-3 * y, ice_speed_n)
hm_wind = heatmap!(ax_wind, 1e-3 * gn.x[1:3:end], 1e-3 * gn.y[1:3:end], ocean_wind_u, colormap=:blues)
# colormap options for heatmap 

#quiver!(ax_wind, mesh.x / 1e3, mesh.y / 1e3, quiver=(ocean_wind_u, ocean_wind_v))#, color=:red, scale_unit=:data, label="wind")
#scatter!(ax_wind, vec(gridmesh.* 1e-3), rotations=0, markersize=20, marker='↑')

hm_o = heatmap!(ax_o, 1e-3 * gn.x, 1e-3 * gn.y, wave_energy, colormap=:dense, colorrange=(0.1, 15))
hm_x = heatmap!(ax_mx, 1e-3 * gn.x, 1e-3 * gn.y, wave_momentum_x, colormap=:dense, colorrange=(0, 1))
hm_y = heatmap!(ax_my, 1e-3 * gn.x, 1e-3 * gn.y, wave_momentum_y, colormap=:dense, colorrange=(0, 1))
#colormaps

#colorbar(ax_my)
Colorbar(fig[1, 3], hm_o, label = "Wave energy [m^2]")
Colorbar(fig[2, 3], hm_x, label = "Wave momentum x []")
# sc

#hm_o = heatmap(ax_o, 1e-3 * mesh.x, 1e-3 * mesh.y, ocean_wind_v, colormap=:redblue)

title = @lift ($n; "Coupled ice-ocean simulation after " * prettytime(wave_simulation.model.clock.time))
Label(fig[0, :], title)
display(fig)


#wave_simulation.stop_time += 1hour

N = 25
record(fig, save_path*"viscoelastic_ice.gif", 1:N, framerate=12) do i
    @info "Plotting frame $i of $N..."
    #@time for _ = 1:10
    #run!(wave_simulation, store=false)
    @info wave_simulation.model.clock
    time_step!(wave_simulation.model, wave_simulation.Δt)
    #coupled_time_step!(ocean_simulation, ice_simulation)
    #end
    n[] = 1
end
