#using OrdinaryDiffEq
using StaticArrays


using PiCLES.ParticleSystems: particle_waves_v6 as PW
import PiCLES: FetchRelations, ParticleTools
using PiCLES.Operators.core_2D: ParticleDefaults, InitParticleValues, InitParticleInstance
using Oceananigans.Units


using BenchmarkTools
#using Revise

using Profile
# debugging:
using ProfileView

using DataFrames

# %%

# % Parameters
initParticleDefaults(s::ParticleDefaults) = [s.lne, s.c̄_x, s.c̄_y, s.x, s.y]
U10, V10 = +10.0, +10.0

# version 3
r_g0 = 0.85
# function to define constants for grouwth and dissipation
Const_ID = PW.get_I_D_constant()
#@set Const_ID.γ = 0.88
Const_Scg = PW.get_Scg_constants()

# %%
#u(x, y, t) = 0.01 - U10 * sin(t / (6 * 60 * 60 * 2π))
#v(x, y, t) = 0.01 - V10 * cos(t / (6 * 60 * 60 * 2π))

#u(x, y, t) = - U10 + 0.01 + x * 0 + y * 0 + t *0
#v(x, y, t) = + V10 + 0.01 + x * 0 + y * 0 + t *0

u(x::Number, y::Number, t::Number) = (10.0 * cos(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
v(x::Number, y::Number, t::Number) = -(10.0 * sin(t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0

# u(x::Number, y::Number, t::Number) = (10 * (t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
# v(x::Number, y::Number, t::Number) = -(10 * (t / (3 * 60 * 60 * 2π)) + 0.1) + x * 0 + y * 0
DT = 10minutes

winds = (u=u, v=v)

# define V4 parameters absed on Const NamedTuple:
default_ODE_parameters = (
    r_g=r_g0,
    C_α=Const_Scg.C_alpha,
    C_φ=Const_ID.c_β,
    C_e=Const_ID.C_e,
    g=9.81)
WindSeamin = FetchRelations.get_initial_windsea(u(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), DT / 2)
z0 = initParticleDefaults(ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0))

z1= z0/2
# %%
Revise.retry()
@time @allocated particle_system = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q);
@time @allocated particle_system1 = PW.particle_equations(u, v, γ=Const_ID.γ, q=Const_ID.q);

#z = @MVector [0.0, 0.0, 0.0, 1e-4, 0.1]
#z = [0.0, 0.0, 0.0, 1e-4, 0.1];
@time @allocated particle_system(z0, z0, default_ODE_parameters, 0.1)
@benchmark particle_system(z0, z0, default_ODE_parameters, 0.1)
# Memory estimate: 0 bytes, allocs estimate: 0.


@time @allocated problem = ODEProblem(particle_system1, z0, (0.0, 1hour), default_ODE_parameters)
@time @allocated problem2 = ODEProblem(particle_system1, z1, (0.0, 10minutes), default_ODE_parameters)

@time @allocated sol = solve(problem, save_everystep=false, saveat=200hours, save_on=false)
@time @allocated sol2 = solve(problem2, save_everystep=false, saveat=200hours, save_on=false)
@time @allocated sol2 = solve(problem2, save_everystep=false, saveat=200hours, save_on=false, save_start=false)

# %% ----- test integration time

function benchmark_solver(solver_type, T)

    z0 = initParticleDefaults(ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0))
    problem = ODEProblem(particle_system, z0, (0.0, T), default_ODE_parameters)
    trial = @benchmark begin
        sol = solve($problem, $solver_type, save_everystep=false, saveat=200hours, save_on=false)
    end samples = 2

    return trial
end

function benchmark_to_dataframe(suite, T, name)
    msuite = median(suite)
    df = DataFrame(name=name,
        Tend=T,
        time=msuite.time / 1e6, #ms
        memory=msuite.memory / 1024, # KB
        allocs=msuite.allocs
    )
    return df
end

function allocation_test(benchmark_tester, time_tuple, solver_tuple; solver_name=("AutoTsit5", "DP5", "Tsit5") )

    test_collect = DataFrame()
    for t in time_tuple, (solver, name) in zip(solver_tuple, solver_name)
        @info "Benchmarking $name with T=$t"
        suite = benchmark_tester(solver, t)
        df = benchmark_to_dataframe(suite, t, name)
        test_collect = vcat(test_collect, df)
    end
    return sort!(test_collect, :time)

end


#allocation_test(  (1hour, 10hour, 20hour), (AutoTsit5(Rosenbrock23()),) ; solver_name=(("AutoTsit5"),) )
Solver_trials = allocation_test(benchmark_solver, (1hour, 10hour, 20hour), (AutoTsit5(Rosenbrock23()), DP5(), Tsit5(),); solver_name=(("AutoTsit5", "DP5", "Tsit5")))
Solver_trials = sort!(Solver_trials, :name)


## long-term integration
Solver_longerm = allocation_test(benchmark_solver, (10minutes, 30minutes, 1hour, 3hour, 10hour, 30hour), (DP5(), Tsit5(),); solver_name=(("DP5", "Tsit5")))
Solver_longerm = sort!(Solver_longerm, :name)

TAutoTsit5 = allocation_test(benchmark_solver, (10minutes, 30minutes, 1hour, 3hour, 10hour, 30hour), (AutoTsit5(Rosenbrock23()),); solver_name=(("AutoTsit5")))

# %% manual versions
# 1hour integration
# Memory estimate: 788.92 KiB, allocs estimate: 16668. 921.542 μs
#  Memory estimate: 343.98 KiB, allocs estimate: 4464.

# 10 hour integration
# Memory estimate: 1.65 MiB, allocs estimate: 35774. 1.921 ms
#  Memory estimate: 737.17 KiB, allocs estimate: 9590.

# 20 hour integration
# Memory estimate: 2.02 MiB, allocs estimate: 43687. 2.428 ms
#  Memory estimate: 900.02 KiB, allocs estimate: 11713.

# %% analyse single timestep

function benchmark_timestep(solver_type, DT)

    z0 = initParticleDefaults(ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0))
    problem = ODEProblem(particle_system, z0, (0.0,200hours), default_ODE_parameters)
    integrator = init(problem, solver_type, saveat=200hours, save_start=false, save_everystep=false)

    @info "Benchmarking $solver_type with DT=$DT"
    trial = @benchmark begin
        #@info $integrator.alg $DT
        step!($integrator, $DT, true);
    end samples = 1

    return trial
end

#benchmark_timestep(AutoTsit5(Rosenbrock23()), 10minutes)
#benchmark_timestep(DP5(), 10minutes)

#Tester_timestep = allocation_test(benchmark_timestep, (10minutes, 30minutes), (AutoTsit5(Rosenbrock23()), DP5(), Tsit5(),); solver_name=(("AutoTsit5", "DP5",)))
#Tester_timestep

Tester_timestep = allocation_test(benchmark_timestep, (10minutes,), (DP5(), Tsit5(), Vern7(),); solver_name=(("DP5", "AutoTsit5", "Vern7")))



