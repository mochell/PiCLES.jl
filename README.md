# Particle-in-Cell for Efficient Swell - PiCLES ![Build Status](https://github.com/mochell/PiCLES.jl/actions/workflows/CI.yml/badge.svg?branch=main)
PiCLES is a fast and efficient wave model for Earth System Models, using Particle-in-Cell methods for better performance.


![The PiCLES on a Surfboard|200px](./img/PiCLES_v0.png)


PiCLES:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13799205.svg)](https://doi.org/10.5281/zenodo.13799205)

Paper draft version:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13799253.svg)](https://doi.org/10.5281/zenodo.13799253)

## Quick Start
A brief guide on how to use PiCLES.

## How to Install

### Step 1: Install Julia
1. Download Julia v1.10.2 or higher from the [official Julia website](https://julialang.org/downloads/).
2. Follow the installation instructions for your operating system.

### Step 2: Download PiCLES Repository

(Once registered as a Package this will be simpler, sry for the delay)

1. Open a terminal or command prompt.
2. Clone the PiCLES repository using Git:
   ```
   git clone https://github.com/mochell/PiCLES.git
   ```
3. Navigate into the cloned repository directory:
   ```
   cd PiCLES
   ```

### Step 3: Install Dependencies and Activate Environment
1. Start Julia by typing `julia` in your terminal within the PiCLES.jl directory.
2. Activate the project environment:
   ```julia
   using Pkg
   Pkg.activate(".")
   ```
3. Install the required dependencies:
   ```julia
   Pkg.instantiate()
   ```

You are now ready to use PiCLES for your simulations.

## How to Run Tests
To run the `T04_2D_reg_test.jl` file from the command line, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the directory where the `T04_2D_reg_test.jl` file is located.
3. Start Julia by typing `julia` in your terminal.
4. In the Julia REPL, include the `T04_2D_reg_test.jl` file:
    ```julia
    include("T04_2D_reg_test.jl")
    ```
5. The test will run and display the results in the terminal.

## Basic model structure
PiCLES follows the modular model structure from [Oceananigans.jl](https://clima.github.io/OceananigansDocumentation/stable/), but it does not currently share objects. Functionality from Oceananigans does not work in PiCLES.

A minimal working example is the following [examples/example_00_minimal.jl](examples/example_00_minimal.jl):
   
   ```julia
   using Pkg
   # This will be replaced by the module load in the future
   Pkg.activate("PiCLES/")  # Activate the PiCLES package 
   
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
   U10, V10 = 10.0, 10.0
   DT = 10minutes
   r_g0 = 0.85 # ratio of c / c_g (phase velocity/ group velocity).
   
   # Define wind functions
   u(x, y, t) = U10
   v(x, y, t) = V10
   winds = (u=u, v=v)
   
   # Define grid
   grid = TwoDCartesianGridMesh(100e3, 51, 100e3, 51)
   
   
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
       periodic_boundary=false,
       minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
       movie=true)
   
   # Build simulation
   wave_simulation = Simulation(wave_model, Δt=DT, stop_time=2hour)#1hours)
   
   # Run simulation
   run!(wave_simulation, cash_store=true)
   
   # Plot initial state
   istate = wave_simulation.store.store[end];
   p1 = plt.heatmap(grid.data.x[:,1] / 1e3, grid.data.y[1,:] / 1e3, istate[:, :, 1])

   ```

## How to Cite

