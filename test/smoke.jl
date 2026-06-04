using Test

using PiCLES
using PiCLES.FetchRelations
using PiCLES.ParticleSystems: particle_waves_v5 as PW
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh
using PiCLES.Models.WaveGrowthModels2D: WaveGrowth2D
using PiCLES.Simulations: Simulation, initialize_simulation!, run!
using PiCLES.Operators.TimeSteppers: time_step!

using Oceananigans.Units: minutes

@testset "loads" begin
    @test isdefined(PiCLES, :Simulations)
    @test isdefined(PiCLES, :Models)
    @test isdefined(PiCLES, :ParticleSystems)
    @test isdefined(PiCLES, :FetchRelations)
end

@testset "fetch relations" begin
    DT = 10minutes
    ws = FetchRelations.MinimalWindsea(10.0, 10.0, DT)
    @test isfinite(ws["lne"])
    @test isfinite(ws["cg_bar_x"])
    @test isfinite(ws["cg_bar_y"])
    @test ws["cg_bar_x"] > 0
end

# Build the smallest model the API supports and advance one step.
# This exercises: grid construction, ODE parameter generation, particle
# equations, model construction, simulation initialization and a single
# Particle-in-Cell timestep.
@testset "2D wave growth: build and step" begin
    U10, V10 = 10.0, 10.0
    DT = 10minutes

    u(x, y, t) = U10
    v(x, y, t) = V10
    winds = (u=u, v=v)

    grid = TwoDCartesianGridMesh(50e3, 11, 50e3, 11)

    ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=0.85)
    particle_system = PW.particle_equations(u, v; γ=Const_ID.γ, q=Const_ID.q)

    WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT)
    ODE_settings = PW.ODESettings(
        Parameters=ODEpars,
        log_energy_minimum=WindSeamin["lne"],
        saving_step=Float64(DT),
        timestep=Float64(DT),
        total_time=60.0 * 60.0,  # 1 hour, in seconds
        dt=1e-3,
        dtmin=1e-4,
    )

    wave_model = WaveGrowth2D(;
        grid=grid,
        winds=winds,
        ODEsys=particle_system,
        ODEsets=ODE_settings,
        periodic_boundary=false,
        minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    )

    sim = Simulation(wave_model; Δt=DT, stop_time=2 * Float64(DT))

    initialize_simulation!(sim)
    @test sim.initialized

    # advance one PIC step and check the state stays finite
    time_step!(sim.model, Float64(DT))
    state = sim.model.State
    @test all(isfinite, state)
    @test maximum(abs, state) < 1e10
end
