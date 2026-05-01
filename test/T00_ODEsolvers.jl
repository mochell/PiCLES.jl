# test_rk35.jl
# Run with: julia test_rk35.jl
# Ensure RK35Integrator.jl is in the same directory.

cd("PiCLES/test")  # Change to test directory to ensure relative paths work 
include("../src/Solvers/RK35Integrator.jl")
using .RK35Integrator: ODEIntegrator, step!, solve!
using LinearAlgebra


plot_flag = true

# -----------------------------
# Test 1: exponential decay (vector)
# z' = -z, z(0) = [1,2,3] => z(t)=exp(-t)*z0
# -----------------------------
function f_exp!(dz, z, p, t)
    @. dz = -z
end

z0 = [1.0, 2.0, 3.0]
t0 = 0.0
tend = 5.0


integ1 = ODEIntegrator(f_exp!, z0, t0, nothing;
    dt=0.1, reltol=1e-10, abstol=1e-12, dtmin=1e-14, dtmax=0.5)

# save at fixed output times
ts1, zs1 = solve!(integ1, tend; saveat=0.25)

z_exact1 = exp(-tend) .* z0
err1 = norm(zs1[end] .- z_exact1, Inf)

println("Test 1 (exp decay, saveat): t=$(ts1[end])  err_inf=$err1")
@assert err1 < 1e-8

# # Optional plotting
# if plot_flag
#     using Plots
#     plot(ts1, hcat(zs1...)', label=["z1" "z2" "z3"], xlabel="t", ylabel="z(t)",
#         title="Test 1: Exponential Decay", legend=:topright)
#     # savefig("test1_exp_decay.png")
#     display(plot(ts1, hcat(zs1...)', label=["z1" "z2" "z3"], xlabel="t", ylabel="z(t)",
#         title="Test 1: Exponential Decay", legend=:topright))
# end
# %%
# -----------------------------
# Test 1b: same problem, but save=false (only final)
# -----------------------------
integ1b = ODEIntegrator(f_exp!, z0, t0, nothing;
    dt=0.1, reltol=1e-10, abstol=1e-12, dtmin=1e-14, dtmax=0.5)

tfinal1b, zfinal1b = solve!(integ1b, tend; save=true)

err1b = norm(zfinal1b[end] .- z_exact1, Inf)
println("Test 1b (exp decay, save=false): t=$(tfinal1b[end])  err_inf=$err1b")
@assert abs(tfinal1b[end] - tend) < 1e-12
@assert err1b < 1e-8

# %%
if plot_flag
    using Plots
    plot(ts1, hcat(zs1...)', label=["z1 (exact)" "z2 (exact)" "z3 (exact)"], xlabel="t", ylabel="z(t)",
        title="Test 1b: Exponential Decay (save=false)", legend=:topright)
    z_result = hcat(zfinal1b...)

    plot!(tfinal1b[1:100:end], z_result[1,1:100:end], marker=:circle, label="z1 RK3.5", color=:red)
    plot!(tfinal1b[1:100:end], z_result[2,1:100:end], marker=:circle, label="z2 RK3.5", color=:green)
    plot!(tfinal1b[1:100:end], z_result[3,1:100:end], marker=:circle, label="z3 RK3.5", color=:blue)
    # display(plot(ts1, hcat(zs1...)', label=["z1" "z2" "z3"], xlabel="t", ylabel="z(t)",
    #     title="Test 1b: Exponential Decay (save=false)", legend=:topright))
    # savefig("test1b_exp_decay_save_false.png")    
end



# %%
# -----------------------------
# Test 2: harmonic oscillator (2D)
# x' = v
# v' = -ω^2 x
# x(0)=1, v(0)=0 => x=cos(ωt), v=-ω sin(ωt)
# -----------------------------
function f_ho!(dz, z, p, t)
    ω = p.ω
    x = z[1]
    v = z[2]
    dz[1] = v
    dz[2] = -(ω^2) * x
end

ω = 2.0
params = (ω=ω,)
z0_ho = [1.0, 0.0]
tend2 = 10.0

integ2 = ODEIntegrator(f_ho!, z0_ho, 0.0, params;
    dt=0.05, reltol=1e-10, abstol=1e-12, dtmin=1e-14, dtmax=0.2)

ts2, zs2 = solve!(integ2, tend2; saveat=0.1)

x_exact = cos(ω * tend2)
v_exact = -ω * sin(ω * tend2)
z_exact2 = [x_exact, v_exact]

err2 = norm(zs2[end] .- z_exact2, Inf)
println("Test 2 (harmonic osc, saveat): t=$(ts2[end])  err_inf=$err2")
@assert err2 < 1e-8


if plot_flag
    using Plots
    #plot exact solution
    ts_exact = range(0, tend2, length=1000)
    x_exact_ts = cos.(ω .* ts_exact)
    v_exact_ts = -ω .* sin.(ω .* ts_exact)
    plot(ts_exact, [x_exact_ts v_exact_ts], label=["x (exact)" "v (exact)"], xlabel="t", ylabel="z(t)", linewidth=4, color=[:black :black],
        title="Test 2: Harmonic Oscillator (exact)", legend=:topright)
    # plot numerical solution
    plot!(ts2, hcat(zs2...)', label=["x RK3.5" "v RK3.5"], xlabel="t", ylabel="z(t)", marker=:circle, color=[:red :green],linewidth=1,
        title="Test 2: Harmonic Oscillator", legend=:topright)
    # savefig("test2_harmonic_oscillator.png")
    # display(plot(ts2, hcat(zs2...)', label=["x RK3.5" "v RK3.5"], xlabel="t", ylabel="z(t)",
    #     title="Test 2: Harmonic Oscillator", legend=:topright))
end



# %%
# -----------------------------
# Test 3: stiff problem (van der Pol with large μ)
# y1' = y2
# y2' = μ(1 - y1^2)*y2 - y1
# y1(0)=2, y2(0)=0
# -----------------------------
function f_vdp!(dy, y, p, t)
    μ = p.μ
    y1 = y[1]
    y2 = y[2]
    dy[1] = y2
    dy[2] = μ * (1 - y1^2) * y2 - y1
end

μ_stiff = 10.0
params_vdp = (μ=μ_stiff,)
z0_vdp = [2.0, 0.0]
tend3 = 400.0

integ3 = ODEIntegrator(f_vdp!, z0_vdp, 0.0, params_vdp;
    dt=0.01, reltol=1e-8, abstol=1e-10, dtmin=1e-14, dtmax=0.5)

ts3, zs3 = solve!(integ3, tend3; saveat=1.0)

println("Test 3 (stiff van der Pol, μ=$μ_stiff): t=$(ts3[end])  final_state=$(zs3[end])")
@assert ts3[end] ≈ tend3

if plot_flag
    using Plots
    plot(ts3, hcat(zs3...)', label=["y1" "y2"], xlabel="t", ylabel="y(t)",
        title="Test 3: Stiff van der Pol (μ=$μ_stiff)", legend=:topright)
    # savefig("test3_vdp_stiff.png")
end

println("All tests passed ✅")
