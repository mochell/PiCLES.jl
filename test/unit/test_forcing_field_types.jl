using Test

using PiCLES
using PiCLES.ParticleSystems: particle_waves_v6 as PW
using PiCLES.Models.WaveGrowthModels2D
using PiCLES.Simulations: Simulation, initialize_simulation!
using PiCLES.Operators: TimeSteppers
using PiCLES.Operators.core_2D: ParticleDefaults
using PiCLES.custom_structures: ForcingCollection, FunctionForcingField, ArrayForcingField
import PiCLES: FetchRelations

const DT = 10 * 60.0
const LX = 20e3
const LY = 20e3
const NX = 9
const NY = 9

base_u(x, y, t) = 5.0
base_v(x, y, t) = -2.0

function make_model()
    grid = PiCLES.Grids.CartesianGrid.TwoDCartesianGridMesh(LX, NX, LY, NY)

    odepars, const_id, _ = PW.ODEParameters(r_g=0.85)
    particle_system = PW.particle_equations(γ=const_id.γ, q=const_id.q, input=true, dissipation=true)

    windsea = FetchRelations.MinimalWindsea(5.0, 2.0, DT)
    defaults = ParticleDefaults(log(windsea["E"]), windsea["cg_bar_x"], windsea["cg_bar_y"], 0.0, 0.0)

    ode_settings = PW.ODESettings(
        Parameters=odepars,
        log_energy_minimum=log(windsea["E"]),
        log_energy_maximum=log(27),
        saving_step=DT,
        timestep=DT,
        total_time=6 * DT,
        dt=1e-3,
        dtmin=1e-4,
        dtmax=DT,
    )

    return WaveGrowthModels2D.WaveGrowth2D(
        grid=grid,
        winds=nothing,
        ODEsys=particle_system,
        ODEsets=ode_settings,
        ODEinit_type=defaults,
        periodic_boundary=true,
        boundary_type="same",
        movie=false,
    )
end

function init_model!(model)
    seed_forcing = ForcingCollection(u_wind=base_u, v_wind=base_v)
    sim = Simulation(model, Δt=DT, stop_time=3 * DT, forcing=seed_forcing, verbose=false)
    initialize_simulation!(sim)
    return sim
end

function run_three_steps!(model, forcing)
    t_start = model.clock.time
    for _ in 1:3
        TimeSteppers.time_step!(model, DT; forcing=forcing)
    end
    return model.clock.time - t_start
end

@testset "forcing field representations run 3 timesteps" begin
    x_axis = collect(range(0.0, LX, length=NX))
    y_axis = collect(range(0.0, LY, length=NY))
    t_axis = [0.0, DT, 2DT, 3DT]

    u_2d = fill(4.0, NX, NY)
    v_2d = fill(-1.0, NX, NY)

    u_3d = Array{Float64}(undef, NX, NY, length(t_axis))
    v_3d = Array{Float64}(undef, NX, NY, length(t_axis))
    for k in eachindex(t_axis)
        u_3d[:, :, k] .= 2.0 + 0.3 * k
        v_3d[:, :, k] .= -0.5 - 0.2 * k
    end

    forcing_cases = [
        ("function(x,y,t)", ForcingCollection(
            u_wind=(x, y, t) -> 4.0 + 0.2 * sin(t / DT),
            v_wind=(x, y, t) -> -1.0 + 0.1 * cos(t / DT),
        )),
        ("function(x,y)", ForcingCollection(
            u_wind=(x, y) -> 4.5,
            v_wind=(x, y) -> -1.5,
        )),
        ("scalar", ForcingCollection(u_wind=3.0, v_wind=-2.0)),
        ("raw 2D arrays", ForcingCollection(u_wind=u_2d, v_wind=v_2d)),
        ("FunctionForcingField", ForcingCollection(
            u_wind=FunctionForcingField((x, y, t) -> 3.8 + 0.1 * sin(0.5 * t / DT)),
            v_wind=FunctionForcingField((x, y) -> -1.2),
        )),
        ("ArrayForcingField 3D", ForcingCollection(
            u_wind=ArrayForcingField(u_3d; x=x_axis, y=y_axis, t=t_axis),
            v_wind=ArrayForcingField(v_3d; x=x_axis, y=y_axis, t=t_axis),
        )),
    ]

    for (label, forcing_case) in forcing_cases
        model = make_model()
        init_model!(model)

        elapsed = run_three_steps!(model, forcing_case)

        @test elapsed ≈ 3 * DT
        @test all(isfinite, model.State)
        @testset "case=$label" begin
            @test model.clock.iteration == 3
        end
    end
end
