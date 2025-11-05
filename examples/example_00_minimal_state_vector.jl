using Pkg
# this will be replace by the module load in the future
Pkg.activate("PiCLES/")  # Activate the PiCLES package 

using PiCLES
using PiCLES.Operators.core_2D: ParticleDefaults
using PiCLES.Models.WaveGrowthModels2D: WaveGrowth2D
using PiCLES.Simulations
using PiCLES.ParticleMesh: TwoDGrid, TwoDGridNotes, TwoDGridMesh

using PiCLES.ParticleSystems: particle_waves_v5 as PW
using Oceananigans.Units

# just for simple plotting
import Plots as plt

# %%
speed(x::Float64, y::Float64) = sqrt(x^2 + y^2)

function PartitionOutput(State)

    energy = State[:, :, 1] # Energy
    m_1 = State[:, :, 2] # Zonal momentum
    m_2 = State[:, :, 3] # Meridional momentum

    # m_0 - Hs
    moment_0_hs = 4 * sqrt.(energy)
    # m_1 - dummy version
    moment_1 = energy * 0.1
    # m_2 - dummy version
    moment_2 = energy * 0.01

    m_amp = speed.(m_1, m_2)

    # T_p - dummy
    # check the use of r_g
    T_p = energy * 0 .+ 10.0

    # c_g vector
    c_x = m_x * energy / (2 * m_amp^2)
    c_y = m_y * energy / (2 * m_amp^2)

    # c stokes vector - dummy 
    u_stokes_x = c_x * 0.1
    u_stokes_y = c_y * 0.1


    output_array = cat(moment_0_hs,
        moment_1, moment_2,
        T_p,
        c_x, c_y,
        u_stokes_x, u_stokes_y, dims=3)

    return output_array

end
# %%

# Parameters
U10, V10 = 10.0, 10.0
DT = 10minutes
r_g0 = 0.85 # ratio of c / c_g (phase velocity/ group velocity).

# Define wind functions
u(x, y, t) = U10
v(x, y, t) = V10
winds = (u=u, v=v)


# ---------- initialize start ---------
# Define grid
grid = TwoDGrid(100e3, 51, 100e3, 51)
gn   = TwoDGridNotes(grid)

# Define ODE parameters
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=r_g0)

# Define particle equations
particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q);

# Calculate minimal windsea based on characteristic winds
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT)

# Define default particle
default_particle = ParticleDefaults(WindSeamin["lne"], WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0)

# Define ODE settings
ODE_settings = PW.ODESettings(
    Parameters=ODEpars,
    log_energy_minimum=WindSeamin["lne"],  # define mininum energy threshold
    saving_step=DT,
    timestep=DT,
    total_time=T = 6days,
    dt=1e-3, #60*10, 
    dtmin=1e-4, #60*5, 
    force_dtmin=true)

# Build wave model
wave_model = WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,
    periodic_boundary=false,
    minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

# Build simulation
wave_simulation = Simulation(wave_model, Δt=DT, stop_time=2hour)#1hours)
# ---------- initialize end ---------

# ---------- run start ---------
# Run simulation
run!(wave_simulation, cash_store=true)

# output per time step for a single partition
SinglePartitionOutput = PartitionOutput(wave_model.State)
# ---------- run end ---------

### some explanations
# wave model State vector:
wave_model.State
wave_model.State[:, :, 1] # Energy
wave_model.State[:, :, 2] # Zonal momentum
wave_model.State[:, :, 3] # Meridional momentum


# Dimensions
# this will we change when PiCLES updated to more advanced grid module
gridnotes = TwoDGridNotes(wave_model.grid)
gridnotes.x
gridnotes.y

# plot state
istate = wave_simulation.store.store[end];
p1 = plt.heatmap(gn.x / 1e3, gn.y / 1e3, SinglePartitionOutput[:, :, 1], title="Hs (m)", xlabel="x (km)", ylabel="y (km)")
