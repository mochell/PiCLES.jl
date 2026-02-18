using Pkg
# This will be replaced by the module load in the future
Pkg.activate(".")  # Activate the PiCLES package 

using PiCLES
using PiCLES.Operators.core_2D: ParticleDefaults
using PiCLES.Models.WaveGrowthModels2D: WaveGrowth2D
using PiCLES.Simulations
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, ProjetionKernel, TwoDCartesianGridStatistics

using PiCLES.ParticleSystems: particle_waves_v5 as PW
using Oceananigans.Units

# just for simple plotting
import Plots as plt

# Parameters
U10, V10 = 20.0, 10.0
DT = 10minutes
r_g0 = 0.85 # ratio of c / c_g (phase velocity/ group velocity).

# Define wind functions
function ind(x,a,b)
  if x>= a && x<b
    return 1
  else
    return 0
  end
end
function u(x, y, t)
  if t <= 30hour
    if x <= 90
      return U10
    else
      return U10/100
    end
  else
    return 0.0
  end
end
v(x, y, t) = V10 * 0#(sin(pi*x/50e3))
winds = (u=u, v=v)

# Define grid
# grid = TwoDCartesianGridMesh(100e3, 51, 100e3, 51)
grid = Grids.SphericalGrid.TwoDSphericalGridMesh(0.0, 180.0, 91, 0, 80.0, 61; periodic_boundary=(true, false))


# Define ODE parameters
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=r_g0)

# Define particle equations
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q);

# Calculate minimal wind sea based on characteristic winds
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT)

# Define default particle
default_particle = ParticleDefaults(WindSeamin["lne"], WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0)

# Define ODE settings
ODE_settings = PW.ODESettings(
  Parameters=ODEpars,
  # define mininum energy threshold
  log_energy_minimum=WindSeamin["lne"],
  saving_step=DT,
  timestep=DT,
  total_time=T = 6days,
  dt=1e-3, 
  dtmin=1e-4, 
  force_dtmin=true)

# Build wave model
wave_model = WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,
    # ODEinit_type=default_particle,
    periodic_boundary=false,
    minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

# Build simulation
wave_simulation = Simulation(wave_model, Δt=DT, stop_time=30hour)#1hours)

# Run simulation
run!(wave_simulation, cash_store=true)

# Plot final state
fstate = wave_simulation.store.store[end];
p1 = plt.heatmap(grid.data.x[:,1] / 1e3, grid.data.y[1,:] / 1e3, fstate[:, :, 1])

function plot_particle_collection(state_i, grid)
    # particles = wave_model.ParticleCollection
    p = plt.plot(layout=(3, 2), size=(1200, 1000))
    # heatmap!(p, transpose(particles.on), subplot=1, title="on | iter=" * string(wave_model.clock.iteration) * " | time=" * string(wave_model.clock.time))
    # heatmap!(p, transpose(particles.boundary), subplot=2, title="boundary")

    sE = state_i[:, :, 1]
    sE[grid.data.mask.==0] .= NaN
    sE[grid.data.mask.==2] .= NaN
    plt.heatmap!(p, transpose(sE), subplot=3, title="State: Energy", clims=(0, NaN))

    sm1 = state_i[:, :, 2]
    sm1[grid.data.mask.==0] .= NaN
    sm1[grid.data.mask.==2] .= NaN
    plt.heatmap!(p, transpose(sm1), subplot=4, title="State: x momentum ", clims=(0, NaN))

    sm2 = state_i[:, :, 3]
    sm2[grid.data.mask.==0] .= NaN
    sm2[grid.data.mask.==2] .= NaN
    plt.heatmap!(p, transpose(sm2), subplot=6, title="State: y momentum ")
    # title = plot!(title="Plot title", grid=false, showaxis=false, bottom_margin=-50Plots.px)
    display(p)
    return p
end


  fstate = wave_simulation.store.store[end];
  plot_particle_collection(fstate, wave_simulation.model.grid)


for i in 1:length(wave_simulation.store.store)
  fstate = wave_simulation.store.store[i];
  plot_particle_collection(fstate, wave_simulation.model.grid)
  # sm2 = wave_model.State[:, :, 3]
  # # p1 = plt.heatmap(grid.data.x[:,1], grid.data.y[1,:], fstate[:, :, 1])
  # p1 = plt.heatmap(p, transpose(sm2), subplot=6, title="State: y momentum ")
  # plt.savefig(p1, "plots/test_case_original/"*string(i)*".png")
  
end