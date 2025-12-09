using DataFrames, DelimitedFiles
using LinearAlgebra
import Statistics as Stc
using GLM
using LaTeXStrings
# import Plots as plt
using Colors
using Random, Distributions
using PlotlyJS
using SpecialFunctions
import PlotlyBase: make_subplots
using Pkg
Pkg.add(url="https://github.com/cossio/TruncatedNormal.jl")
import TruncatedNormal as tn

## Defining useful functions

function gaussD(mu, sigma, x1)
    value = 1/((2*pi)^0.5 * sigma^0.5) * exp(-0.5*(x1-mu)*sigma^(-1)*(x1-mu))
    return value
end

function print_mat(M)
    a = size(M)[1]
    b = size(M)[2]
    print("[")
    for i in 1:a
        if i != 1 println() end
        print(M[i,1])
        for j in 2:b
            print(",  ")
            print(M[i,j])
        end
    end
    println("]")
end

function gaussMultiD(mu, sigma, sigma1, x1)
    dim = length(mu)
    value = 1/((2*pi)^(dim/2) * det(sigma)^0.5) * exp(-0.5*transpose(x1-mu)*sigma1*(x1-mu))
    return value
end

function spectrum_at_point(T, PartStates, xyPoint, sigma, kxrange, kyrange, nkX, nkY)
    kxs = (0:(nkX-1))/nkX*(kxrange[2]-kxrange[1]).+kxrange[1]
    kys = (0:(nkY-1))/nkY*(kyrange[2]-kyrange[1]).+kyrange[1]

    sigma1 = inv(sigma)
    influence_radius_2 = sqrt(sigma[1,1])^2+sqrt(sigma[2,2])^2

    particles = PartStates[T,"particleList"]
    nPart = (size(particles))[1]

    result = zeros(nkX,nkY)

    for i in 1:nPart
        if (particles[i,"x"]-xyPoint[1])^2+(particles[i,"y"]-xyPoint[2])^2 < influence_radius_2*1.5
            for x in 1:nkX
                for y in 1:nkY
                    result[x,y] += gaussMultiD([xyPoint[1],xyPoint[2], kxs[x], kys[y]], sigma, sigma1, [particles[i,"x"],particles[i,"y"],particles[i,"cx"],particles[i,"cy"]])
                end
            end
        end
    end
    return result
end

function compute_spectrum_parameters(T, PartStates, xyPoint, sigma, aggregation_type)
    all_particles = PartStates[T,"particleList"][:,["cx","cy","x","y"]]

    influence_radius = (sqrt(sigma[1,1])^2+sqrt(sigma[2,2])^2)
    sigma = [influence_radius 0; 0 influence_radius]
    sigma1 = inv(sigma)
    result = DataFrame(cx=[],cy=[],x=[],y=[])
    weights = []

    for i in 1:size(all_particles)[1]
        temp_pos = Vector(all_particles[i,["x","y"]])
        dist = (temp_pos-xyPoint)[1]^2+(temp_pos-xyPoint)[2]^2
        if dist < influence_radius* (10)^2
            append!(result,DataFrame(all_particles[i,:]))
            if aggregation_type == "gauss"
                append!(weights, gaussMultiD(temp_pos, sigma, sigma1, xyPoint))
            elseif aggregation_type == "const"
                append!(weights, 1)
            elseif aggregation_type == "lin"
                if abs(temp_pos[1]-xyPoint[1]) <= sqrt(sigma[1,1]) && abs(temp_pos[2]-xyPoint[2]) <= sqrt(sigma[2,2])
                    append!(weights,(abs(temp_pos[1]-xyPoint[1])/sqrt(sigma[1,1])*abs(temp_pos[2]-xyPoint[2])/sqrt(sigma[2,2])))
                else
                    append!(weights, 0)
                end
            end
        end
    end

    if length(weights) == 0
        return [0 0; 0 0], 0
    end
    sumWeights=sum(weights)
    weights=weights./sumWeights
    cBar_cx = sum(Matrix(weights.*result[:,["cx"]]))
    cBar_cy = sum(Matrix(weights.*result[:,["cy"]]))

    cov_cxcx = 1/(1-sum(weights.^2))*sum(weights.*(Matrix(result[:,["cx"]]).-cBar_cx).^2)
    cov_cycy = 1/(1-sum(weights.^2))*sum(weights.*(Matrix(result[:,["cy"]]).-cBar_cy).^2)
    cov_cxcy = 1/(1-sum(weights.^2))*sum(weights.*(Matrix(result[:,["cx"]]).-cBar_cx).*(Matrix(result[:,["cy"]]).-cBar_cy))
    result_cov = [cov_cxcx cov_cxcy; cov_cxcy cov_cycy]

    return result_cov, sumWeights
end

function compute_derivative(y, t, step)
    results = zeros(length(t)-2*step)
    times = t[(step+1):(end-step)]
    for i in 1:(length(t)-2*step)
        results[i] = (y[i+2*step]-y[i])/(t[i+2*step]-t[i])
    end
    return times, results
end

function spectrum_parameters_at_point(T, PartStates, xyPoint, influenceZone)
    peak_speeds = Matrix{}
end

function peak_spectrum(T, PartStates, xyPoint, sigma, sigma1)
    particles = PartStates[T,"particleList"]
    nPart = (size(particles))[1]
    influence_radius_2 = (sigma[1,1])+(sigma[2,2])

    value = [0; 0]
    weights = 0
    for i in 1:nPart
        if (particles[i,"x"]-xyPoint[1])^2+(particles[i,"y"]-xyPoint[2])^2 < influence_radius_2*(25)^2
            local_weight = gaussMultiD(xyPoint, sigma, sigma1, [particles[i, "x"]; particles[i, "y"]])
            value += [particles[i,"cx"]; particles[i,"cy"]] * local_weight
            weights += local_weight
        end
    end
    if weights == 0
        weights = 1
    end
    return value/weights
end

function find_geometrical_center(T, PartStates)
    particles = PartStates[T,"particleList"]
    return [Stc.mean(particles[:,"x"]), Stc.mean(particles[:,"y"])]
end

function unit_dec(a)
    unit=string(Int64(floor((a))))
    dec=string(Int64(floor(10*(a-floor((a))))))
    return unit*","*dec
end

function plot_state_and_error_points(State, xs, ys, cRange, widthPlot, heightPlot; showTitle = true)
    custom_color_scale = [(0.0,"rgb(229, 229, 229)"),(0.3,"rgb(255, 230, 44)"), (1.0, "rgb(255, 0, 0)")]

    traceq1 = heatmap(x=xs, y=ys, z=Matrix(State[:, :]), colorscale=custom_color_scale
        ,colorbar=attr(tickformat=".1e",cmin = cRange[1], cmax = cRange[2])
        ,zmin = cRange[1], zmax = cRange[2]
        )

    LatexFont = attr(color="black",family="Computer Modern",size=22)

    if showTitle
        q1=plot(traceq1,
            Layout(
                title=attr(text="\$ \\Large{\\text{Energy distribution in space}}\$",
                    x = 0.5,
                    xanchor="center",
                    font=attr(color="black")),
                font=LatexFont,
                width=widthPlot, height=heightPlot,
                xaxis=attr(title=attr(text="\$\\Large{\\text{x position in } km}\$",font=attr(color="black"))),
                yaxis=attr(title=attr(text="\$\\Large{\\text{y position in } km}\$",font=attr(color="black")),scaleanchor="x"))
                )
    else
        q1=plot(traceq1,
        Layout(
            font=LatexFont,
            width=widthPlot, height=heightPlot,
            xaxis=attr(title=attr(text="\$\\Large{\\text{x position in } km}\$",font=attr(color="black"))),
            yaxis=attr(title=attr(text="\$\\Large{\\text{y position in } km}\$",font=attr(color="black")),scaleanchor="x"))
            )
    end
    return q1
end

function compute_translated_gaussian(mu_init, sigma_init, M, time)
    translated_mu = M(time)*mu_init
    translated_sigma = M(time)*transpose(sigma_init)*transpose(M(time))
    return translated_mu, translated_sigma
end

function pearson(data, predicted)
    meanValue = Stc.mean(data)
    return 1-sum((data-predicted).^2)/sum((data.-meanValue).^2)
end

## Reading the data
localpath = pwd()

# parentPath = localpath * "/plots/tests/paper/test_case_1/v2_pi"
# parentPath = localpath * "/plots/tests/paper/test_case_2_local/v2_pi_4"
# parentPath = localpath * "/plots/tests/paper/test_case_2/v2_pi_4"
# parentPath = localpath * "/plots/tests/paper/test_case_3/v2_half_cell"
# parentPath = localpath * "/plots/tests/paper/test_case_3/v2_half_cell"
# parentPath = localpath * "/plots/tests/paper/test_case_2/v2_pi_4_operationnel"
# parentPath = localpath * "/plots/tests/paper/test_debug/v2_half_cell"

println("------------ Reading the data from : "* parentPath * "  ------------")
path = parentPath*"/data/simu_infos.csv"
path3 = parentPath*"/data/sigma.csv"
data, header = readdlm(path, ',', header=true)
data3, header3 = readdlm(path3, ',', header=true)

simu_infos = DataFrame(data, vec(header))
init_sigma_df = DataFrame(data3, vec(header3))

init_sigma = Matrix(init_sigma_df)
Nx = simu_infos.Nx[1]
Ny = simu_infos.Ny[1]
xmin = simu_infos.xmin[1]
xmax = simu_infos.xmax[1]
ymin = simu_infos.ymin[1]
ymax = simu_infos.ymax[1]
lne_source = simu_infos.lne_source[1]
c_x_source = simu_infos.c_x_source[1]
c_y_source = simu_infos.c_y_source[1]
init_mu_C = [c_x_source, c_y_source]
x_source = simu_infos.x_source[1]
y_source = simu_infos.y_source[1]
init_mu_X = [x_source,y_source]
angular_spread_source = simu_infos.angular_spread_source[1]
Δt = simu_infos.Δt[1]
stop_time = simu_infos.stop_time[1]

nTimes = Int64(ceil(stop_time/Δt))+1

# times = (1:(nTimes))*Δt
times = (0:(nTimes-1))*Δt
timesMinutes = times/60

timesMinutesString = unit_dec.(timesMinutes)

ParticleStates = DataFrame([[],[]],["times", "particleList"])
MeshState = DataFrame([[],[]],["times", "mesh_state"])

for i in 1:nTimes
    path1 = parentPath*"/data/particles_"*timesMinutesString[i]*".csv"
    path2 = parentPath*"/data/mesh_values_"*timesMinutesString[i]*".csv"
    data1, header1 = readdlm(path1, ',', header=true)
    data2, header2 = readdlm(path2, ',', header=true)

    read_df1 = DataFrame(data1, vec(header1))
    read_df2 = DataFrame(data2, vec(header2))

    temp_df1 = DataFrame([[timesMinutes[i]], [read_df1]], ["times", "particleList"])
    temp_df2 = DataFrame([[timesMinutes[i]], [read_df2]], ["times", "mesh_state"])
    append!(ParticleStates, temp_df1)
    append!(MeshState, temp_df2)
end

## Building the spectrum at different points and different times

nParticles = size(ParticleStates[1,"particleList"])[1]

nkX = 15
nkY = 15

kxscale = [-10, 30]
kyscale = [-10, 30]

alpha = 5
alpha = alpha /180 * pi
rotMatPos = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)]
rotMatNeg = [cos(-alpha) -sin(-alpha); sin(-alpha) cos(-alpha)]

kxs = (0:(nkX-1))/nkX*(kxscale[2]-kxscale[1]).+kxscale[1]
kys = (0:(nkY-1))/nkY*(kyscale[2]-kyscale[1]).+kyscale[1]

sigma_x = (xmax-xmin)/(Nx-1)
sigma_y = (ymax-ymin)/(Ny-1)
sigma_kx = (kxscale[2]-kxscale[1])/nkX
sigma_ky = (kyscale[2]-kyscale[1])/nkY

kernel = [sigma_x^2 0 0 0;
          0 sigma_y^2 0 0;
          0 0 sigma_kx^2 0;
          0 0 0 sigma_ky^2]

sdx_end = sqrt(Stc.var(ParticleStates[nTimes, "particleList"][:,"x"]))
sdy_end = sqrt(Stc.var(ParticleStates[nTimes, "particleList"][:,"y"]))

xyCenter_1 = find_geometrical_center(1, ParticleStates)
xyCenter_2 = find_geometrical_center(Int64(floor(nTimes/2)), ParticleStates)
xyCenter_end = find_geometrical_center(nTimes, ParticleStates)
xy_right = rotMatNeg * xyCenter_end
xy_left = rotMatPos * xyCenter_end
xy_forw = xyCenter_end + 1 * [sdx_end,sdy_end]
xy_back = xyCenter_end - 1 * [sdx_end,sdy_end]
println()
println("------------------------------------------------")
println("                  POSITIONS :")
println()

println("Center at t=0          : (x,y) = ("*string(xyCenter_1[1])*","*string(xyCenter_1[2])*")")
println("Center at t="*timesMinutesString[nTimes]*"       : (x,y) = ("*string(xyCenter_end[1])*","*string(xyCenter_end[2])*")")
println("Right at t="*timesMinutesString[nTimes]*"        : (x,y) = ("*string(xy_right[1])*","*string(xy_right[2])*")")
println("Left at t="*timesMinutesString[nTimes]*"         : (x,y) = ("*string(xy_left[1])*","*string(xy_left[2])*")")
println("Forward at t="*timesMinutesString[nTimes]*"      : (x,y) = ("*string(xy_forw[1])*","*string(xy_forw[2])*")")
println("Backward at t="*timesMinutesString[nTimes]*"     : (x,y) = ("*string(xy_back[1])*","*string(xy_back[2])*")")

compute_spectra_bool = false

if compute_spectra_bool

    println()
    println("------------------------------------------------")
    println("              COMPUTING SPECTRA :")
    println()

    spectrum_1 = spectrum_at_point(1, ParticleStates, xyCenter_1, kernel, kxscale, kyscale, nkX, nkY)
    println("Spectrum 1 over")
    spectrum_2 = spectrum_at_point(Int64(floor(nTimes/2)), ParticleStates, xyCenter_2, kernel, kxscale, kyscale, nkX, nkY)
    println("Spectrum 1 over")
    spectrum_3 = spectrum_at_point(nTimes, ParticleStates, xyCenter_end, kernel, kxscale, kyscale, nkX, nkY)
    println("Spectrum 2 over")
    #spectrum_right = spectrum_at_point(nTimes, ParticleStates, xy_right, kernel, kxscale, kyscale, nkX, nkY)
    spectrum_end = zeros(nkX, nkY)
    spectrum_right = zeros(nkX, nkY)
    println("Spectrum 3 over")
    # spectrum_left = spectrum_at_point(nTimes, ParticleStates, xy_left, kernel, kxscale, kyscale, nkX, nkY)
    spectrum_left = zeros(nkX, nkY)
    println("Spectrum 4 over")
    spectrum_forw = spectrum_at_point(nTimes, ParticleStates, xy_forw, kernel, kxscale, kyscale, nkX, nkY)
    println("Spectrum 5 over")
    spectrum_back = spectrum_at_point(nTimes, ParticleStates, xy_back, kernel, kxscale, kyscale, nkX, nkY)
    println("Spectrum 6 over")
    println()

    # FIRST PLOT
    axis_template = attr(range = [-10,30], autorange = false,
                showgrid = false, zeroline = false,
                linecolor = "black", showticklabels = true,
                ticks = "" )

    sp1_trace = heatmap(x=kxs, y=kys, z=spectrum_1)
    sp1_trace2 = scatter(x=[0, 0],y=[-20, 40], line=attr(color="black", width=3), showticklabels = false, ticks = "", name="")
    sp1_trace3 = scatter(y=[0, 0],x=[-20, 40], line=attr(color="black", width=3), showticklabels = false, ticks = "", name="")
    sp1 = plot([sp1_trace, sp1_trace2, sp1_trace3], Layout(
            legend=:none,
            title = "Energy Spectrum (in velocity) at t=0",
            xaxis=axis_template,
            yaxis=axis_template,
            yaxis_title="c_y",
            xaxis_title="c_x",width=1000, height=1000,
            font=attr(size=28)))

    # SECOND PLOT
    axis_template = attr(range = [-10,30], autorange = false,
            showgrid = false, zeroline = false,
            linecolor = "black", showticklabels = true,
            ticks = "" )

    sp2_trace = heatmap(x=kxs, y=kys, z=spectrum_2)
    sp2_trace2 = scatter(x=[0, 0],y=[-20, 40], line=attr(color="black", width=3), showticklabels = false, ticks = "", name="")
    sp2_trace3 = scatter(y=[0, 0],x=[-20, 40], line=attr(color="black", width=3), showticklabels = false, ticks = "", name="")
    sp2 = plot([sp2_trace, sp2_trace2, sp2_trace3], Layout(
    legend=:none,
    title = "Energy Spectrum (in velocity) at t=0",
    xaxis=axis_template,
    yaxis=axis_template,
    yaxis_title="c_y",
    xaxis_title="c_x",width=1000, height=1000,
    font=attr(size=28)))

    # THIRD PLOT
    axis_template = attr(range = [-10,30], autorange = false,
            showgrid = false, zeroline = false,
            linecolor = "black", showticklabels = true,
            ticks = "" )

    sp3_trace = heatmap(x=kxs, y=kys, z=spectrum_3)
    sp3_trace2 = scatter(x=[0, 0],y=[-20, 40], line=attr(color="black", width=3), showticklabels = false, ticks = "", name="")
    sp3_trace3 = scatter(y=[0, 0],x=[-20, 40], line=attr(color="black", width=3), showticklabels = false, ticks = "", name="")
    sp3 = plot([sp3_trace, sp3_trace2, sp3_trace3], Layout(
    legend=:none,
    title = "Energy Spectrum (in velocity) at t=0",
    xaxis=axis_template,
    yaxis=axis_template,
    yaxis_title="c_y",
    xaxis_title="c_x",width=1000, height=1000,
    font=attr(size=28)))

    # NEXT PLOTS
    p2 = plt.heatmap(kxs, kys, spectrum_end, aspect_ratio=:equal, size=(2160, 2500))
    plt.plot!(legend=:none,
            title = "Energy Spectrum in velocity at t="*timesMinutesString[nTimes],
            ylabel="c_y",
            xlabel="c_x")

    p3 = plt.heatmap(kxs, kys, spectrum_right, aspect_ratio=:equal, size=(2160, 2500))
    plt.plot!(legend=:none,
            title = "Energy Spectrum in velocity at t="*timesMinutesString[nTimes],
            ylabel="c_y",
            xlabel="c_x")

    p4 = plt.heatmap(kxs, kys, spectrum_left, aspect_ratio=:equal, size=(2160, 2500))
    plt.plot!(legend=:none,
            title = "Energy Spectrum in velocity at t="*timesMinutesString[nTimes],
            ylabel="c_y",
            xlabel="c_x")

    p5 = plt.heatmap(kxs, kys, spectrum_forw, aspect_ratio=:equal, size=(2160, 2500))
    plt.plot!(legend=:none,
            title = "Energy Spectrum in velocity at t="*timesMinutesString[nTimes],
            ylabel="c_y",
            xlabel="c_x")

    p6 = plt.heatmap(kxs, kys, spectrum_back, aspect_ratio=:equal, size=(2160, 2500))
    plt.plot!(legend=:none,
            title = "Energy Spectrum in velocity at t="*timesMinutesString[nTimes],
            ylabel="c_y",
            xlabel="c_x")


    max_speed_id = argmax(spectrum_1)
    max_speed = [kxs[max_speed_id[1]],kys[max_speed_id[2]]]

    proba_spectrum_1 = spectrum_1 / sum(spectrum_1)
    proba_spectrum_1[max_speed_id[1],max_speed_id[2]] += 1-sum(proba_spectrum_1)
end

println("DONE")


xs = (0:(Nx-1))./(Nx-1)*(xmax-xmin).+xmin
ys = (0:(Ny-1))./(Ny-1)*(ymax-ymin).+ymin

# Plotting test case 1 plots

plot_test_case_1 = false

if plot_test_case_1
    plotting_times = [2, Int64(floor(size(MeshState)[1]/2))+1, size(MeshState)[1]]

    halfMeshStates = DataFrame([[],[]], ["times", "mesh_state"])


    yid_to_plot = 1:(size(MeshState[1, "mesh_state"])[1])
    xid_to_plot = 1:Int64(floor(size(MeshState[1, "mesh_state"])[1]/2))
    for i in 1:length(plotting_times)
        values = MeshState[plotting_times[i], "mesh_state"]
        values = values[xid_to_plot,yid_to_plot]
        df = DataFrame([[MeshState[plotting_times[i],"times"]], [values]], ["times", "mesh_state"])
        append!(halfMeshStates, df)
    end

    cRange = [0, maximum(Matrix(halfMeshStates[2, "mesh_state"]))]

    tc1_1 = plot_state_and_error_points(halfMeshStates[1, "mesh_state"], xs[xid_to_plot]./1000, ys[yid_to_plot]./1000, cRange, 1130, 573)
    tc1_2 = plot_state_and_error_points(halfMeshStates[2, "mesh_state"], xs[xid_to_plot]./1000, ys[yid_to_plot]./1000, cRange, 1130, 573)
    tc1_3 = plot_state_and_error_points(halfMeshStates[3, "mesh_state"], xs[xid_to_plot]./1000, ys[yid_to_plot]./1000, cRange, 1130, 573)

    save_plots = true
    if save_plots
        savefig(tc1_1,"test_case_1_t1.html")
        savefig(tc1_2,"test_case_1_t2.html")
        savefig(tc1_3,"test_case_1_t3.html")
        println("Plotting times for the test case 1 energy distribution : "*string(MeshState[plotting_times[1], "times"])*", "*string(MeshState[plotting_times[2], "times"])*", "*string(MeshState[plotting_times[3], "times"]))
    end

    # Plotting the energy decay rate

    axis_template = attr(autorange = true,
        showgrid = true, zeroline = false,
        linecolor = "black", showticklabels = true,
        ticks = "" )

    LatexFont = attr(color="black",family="Computer Modern",size=22)

    ed_layout = Layout(yaxis_type="log",xaxis_type="log",
                title=attr(text="Energy density evolution with distance",font=LatexFont)
                ,width=800, height=800
                ,xaxis_title=L"\Large{\text{distance in } km}"
                ,yaxis_title=L"\Large{\text{Energy density (arbitrary units)}}"
                ,legend=attr(borderwidth=1,x=0.1, y=0.4,font=LatexFont)
                ,font=LatexFont
                ,xaxis=axis_template
                ,yaxis=axis_template
                ,margin=attr(l=90,r=80,t=80,b=80)
                ,template="simple_white")

    initPos = [x_source; y_source]

    argmaxValues = [argmax(transpose(Matrix(MeshState[i,"mesh_state"][:,:]))) for i in 2:(size(MeshState)[1])]
    max_energy = [MeshState[i,"mesh_state"][argmaxValues[i-1][2],argmaxValues[i-1][1]] for i in 2:(size(MeshState)[1])]
    maxPos = [[xmin.+(argmaxValues[i][1]-1) .* sigma_x; ymin.+(argmaxValues[i][2]-1) .* sigma_y] for i in 1:length(argmaxValues)]
    maxDist = [sqrt((maxPos[i][1]-x_source)^2+(maxPos[i][2]-y_source)^2) for i in 1:length(maxPos)]

    coeff = 300
    perfect_1_r = [coeff/distance for distance in maxDist[12:end]]

    trace_energy_decay = scatter(x=maxDist./1000,y=max_energy,
        name="\$\\text{Energy density}\$",
        mode="markers"
    )
    trace_perfect_1_r = scatter(x=maxDist[12:end]./1000,y=perfect_1_r,
        name="\$\\frac{1}{r} \\text{ curve for reference}\$"
    )
    tc1_ed = plot([trace_energy_decay, trace_perfect_1_r], ed_layout)

    if save_plots
        savefig(tc1_ed,"test_case_1_energy_decay.html")
    end

end

# Plotting test case 2 plots

plot_test_case_2 = true

if plot_test_case_2
    # Plotting the energy distribution

    plotting_times = [2, Int64(floor(size(MeshState)[1]/2))+2, size(MeshState)[1]]

    plotLog = false
    colorbarDelay = 0
    if plotLog
        cRange = [-10, maximum(Matrix(log.(MeshState[plotting_times[1]+colorbarDelay, "mesh_state"])))]

        tc2_1 = plot_state_and_error_points(log.(MeshState[plotting_times[1], "mesh_state"]), xs./1000, ys./1000, cRange, 1000, 885)
        tc2_2 = plot_state_and_error_points(log.(MeshState[plotting_times[2], "mesh_state"]), xs./1000, ys./1000, cRange, 1000, 885)
        tc2_3 = plot_state_and_error_points(log.(MeshState[plotting_times[3], "mesh_state"]), xs./1000, ys./1000, cRange, 1000, 885)
    else
        cRange = [0, maximum(Matrix(MeshState[plotting_times[2]+colorbarDelay, "mesh_state"]))]

        tc2_1 = plot_state_and_error_points(MeshState[plotting_times[1], "mesh_state"], xs./1000, ys./1000, cRange, 1000, 885; showTitle=false)
        tc2_2 = plot_state_and_error_points(MeshState[plotting_times[2], "mesh_state"], xs./1000, ys./1000, cRange, 1000, 885; showTitle=false)
        tc2_3 = plot_state_and_error_points(MeshState[plotting_times[3], "mesh_state"], xs./1000, ys./1000, cRange, 1000, 885; showTitle=false)
    end

    save_plots = true
    if save_plots
        savefig(tc2_1,"test_case_2_t1.html")
        savefig(tc2_2,"test_case_2_t2.html")
        savefig(tc2_3,"test_case_2_t3.html")
        println("Plotting times for the test case 2 energy distribution : "*string(MeshState[plotting_times[1], "times"])*", "*string(MeshState[plotting_times[2], "times"])*", "*string(MeshState[plotting_times[3], "times"]))
    end

    # Plotting the two analysis plots

    xaxis_template = attr(#autorange = true,
        showgrid = true, zeroline = false,
        linecolor = "black", showticklabels = true,
        ticks = "",
        range=[xmin,xmax]./1000
    )

    yaxis_template = attr(#autorange = true,
        showgrid = true, zeroline = false,
        linecolor = "black", showticklabels = true,
        ticks = "",
        range=[ymin,ymax]./1000
    )

    LatexFont = attr(color="black",family="Computer Modern",size=22)

    ed_layout = Layout(#yaxis_type="log",xaxis_type="log",
                title=attr(text="Particle positions at t=60sec",font=LatexFont)
                ,width=1600, height=374
                ,xaxis_title=attr(text="\$\\Large{\\text{x position in } km}\$",font=attr(color="black"))
                ,yaxis_title=attr(text="\$\\Large{\\text{y position in } km}\$",font=attr(color="black"))
                ,legend=attr(borderwidth=1,x=0.1, y=0.4,font=LatexFont)
                ,font=LatexFont
                ,xaxis=xaxis_template
                ,yaxis=yaxis_template
                ,margin=attr(l=90,r=80,t=80,b=80)
                ,template="simple_white")


    cov_end = Stc.cov(Matrix(ParticleStates[1,"particleList"][:,["cx","cy","x","y"]]))

    # q1 = plot_state_and_error_points(MeshState[nTimes, "mesh_state"], xs, ys, xmin, xmax, ymin, ymax)
    println("ici 1")
    # particleSpeeds = sqrt.([sum((Matrix(ParticleStates[nTimes,"particleList"][:,["cx","cy"]]).^2)[i,:]) for i in 1:nParticles])
    particleSpeeds=zeros(nParticles)
    plotting_time = Int64(floor(nTimes/2))
    for i in 1:nParticles
        particleSpeeds[i] = sqrt(ParticleStates[plotting_time,"particleList"][:,"cx"][i]^2+ParticleStates[plotting_time,"particleList"][:,"cy"][i]^2)
    end
    println("ici 2")
    particleXs = ParticleStates[plotting_time,"particleList"][:,"x"]
    particleYs = ParticleStates[plotting_time,"particleList"][:,"y"]
    M(t) = [Diagonal(ones(2)) Diagonal(zeros(2));
            t*Diagonal(ones(2)) Diagonal(ones(2))]
    println("ici 3")
    result_mu, result_sigma = compute_translated_gaussian([c_x_source, c_y_source, x_source, y_source], init_sigma, M, 60*timesMinutes[plotting_time])
    println("ici 4")
    timeOfInterest = Int64(floor(nTimes/2))
    translated_mu_C = [c_x_source, c_y_source]
    translated_mu_XY = [x_source, y_source] + timesMinutes[timeOfInterest]*60 * [c_x_source, c_y_source]
    translated_sigma_XY = init_sigma[3:4,3:4] + (timesMinutes[timeOfInterest]*60)^2 * init_sigma[1:2,1:2]
    translated_cross_cov = timesMinutes[timeOfInterest]*60 * init_sigma[1:2,1:2]
    # translated_sigma_C = init_sigma[1:2,1:2]
    # translated_cov_total=[translated_sigma_C translated_cross_cov;
    #                     transpose(translated_cross_cov) translated_sigma_XY]
    # println("ici 5")
    spectrum(x,y) = gaussMultiD(translated_mu_XY,translated_sigma_XY,inv(translated_sigma_XY),[x,y])
    println("ici 6")
    z = @. spectrum(xs', ys)
    nPartSubset = 20000
    subsetPartID = Int64.(floor.(rand(nPartSubset)*length(particleXs))).+1
    color1=[255 0 0]
    color2=[255 230 44]
    minSpeed=minimum(particleSpeeds)
    maxSpeed=maximum(particleSpeeds)
    colorPart=color1.*((particleSpeeds[subsetPartID].-minSpeed)./maxSpeed).+color2.*(1 .-(particleSpeeds[subsetPartID].-minSpeed)./maxSpeed)
    transparencyPoints = 1
    trace_tc_2_1 = scatter(x=particleXs[subsetPartID]./1000,y=particleYs[subsetPartID]./1000,
                    mode="markers",
                    marker=attr(size=3,
                        color=particleSpeeds[subsetPartID],
                        # color="rgba(".*string.(Int64.(floor.(colorPart[:,1]))).*", ".*string.(Int64.(floor.(colorPart[:,2]))).*", ".*string.(Int64.(floor.(colorPart[:,3]))).*", ".*string(transparencyPoints).*")",
                        showscale=true,
                        colorscale=[[0,"rgb(255, 230, 44)"],[1,"rgb(255, 0, 0)"]]
                        ),
                    zorder=0)
    trace_tc_2_2 = contour(x=xs./1000,y=ys./1000,z=z,contours_coloring="lines",line_width=4,showscale=false,colorscale=[[0,"black"],[0.5,"black"],[1,"black"]],zorder=1)

    q1 = plot([trace_tc_2_1,trace_tc_2_2], ed_layout)

    if save_plots
        savefig(q1,"test_case_2_particle_contour.html")
    end

    line_plot_nPoints = 350
    line_plot_xy = zeros(line_plot_nPoints,2)
    line_plot_xy[:,1] = (0:(line_plot_nPoints-1))./(line_plot_nPoints-1)*(xmax-xmin).+xmin
    line_plot_xy[:,2] = (y_source) * ones(line_plot_nPoints)
    line_plot_length = [sqrt((line_plot_xy[i,1]-line_plot_xy[1,1])^2+(line_plot_xy[i,2]-line_plot_xy[1,2])^2) for i in 1:line_plot_nPoints]
    kernel1 = inv(kernel)
    println("Start measuring peak_spectrum")

    # Compute model output curve
    line_plot_vector = [peak_spectrum(timeOfInterest, ParticleStates, line_plot_xy[i,:],kernel[1:2,1:2],kernel1[1:2,1:2]) for i in 1:line_plot_nPoints]
    # line_plot_values = [sqrt(sum(line_plot_vector[i].^2)) for i in 1:line_plot_nPoints]
    line_plot_values = [line_plot_vector[i][1] for i in 1:line_plot_nPoints]
    println("Finished measuring peak_spectrum")

    # Compute theoretical curve
    init_mu_C_measured = [Stc.mean(Matrix(ParticleStates[1,"particleList"][:,["cx"]])), Stc.mean(Matrix(ParticleStates[1,"particleList"][:,["cy"]]))]
    # init_mu_C_measured = init_mu_C
    init_mu_X_measured = [Stc.mean(Matrix(ParticleStates[3,"particleList"][:,["x"]])), Stc.mean(Matrix(ParticleStates[3,"particleList"][:,["y"]]))] - Δt*2*init_mu_C_measured
    init_sigma_X_measured = Stc.cov(Matrix(ParticleStates[3,"particleList"][:,["x","y"]]))
    # init_sigma_X_measured = init_sigma[3:4,3:4]
    # init_mu_X_measured = init_mu_X
    translated_sigma_XY1 = inv(translated_sigma_XY)
    translated_mu_XY = init_mu_X_measured+ timesMinutes[timeOfInterest]*60 * [c_x_source, c_y_source]
    translated_cross_cov = timesMinutes[timeOfInterest]*60 * init_sigma[1:2,1:2]
    line_plot_theory_vector = [translated_mu_C+translated_cross_cov*translated_sigma_XY1*(line_plot_xy[i,:]-translated_mu_XY) for i in 1:line_plot_nPoints]
    # line_plot_theory_values = [sqrt(sum(line_plot_theory_vector[i].^2)) for i in 1:line_plot_nPoints]
    line_plot_theory_values = [line_plot_theory_vector[i][1] for i in 1:line_plot_nPoints]

    # Compute diffused curve
    coef = 0.5
    init_mu_C_measured = [Stc.mean(Matrix(ParticleStates[1,"particleList"][:,["cx"]])), Stc.mean(Matrix(ParticleStates[1,"particleList"][:,["cy"]]))]
    # init_mu_C_measured = init_mu_C
    init_mu_X_measured = [Stc.mean(Matrix(ParticleStates[3,"particleList"][:,["x"]])), Stc.mean(Matrix(ParticleStates[3,"particleList"][:,["y"]]))] - Δt*2*init_mu_C_measured
    init_sigma_X_measured = Stc.cov(Matrix(ParticleStates[3,"particleList"][:,["x","y"]]))
    init_sigma_C_measured = Stc.cov(Matrix(ParticleStates[1,"particleList"][:,["cx","cy"]]))
    # init_sigma_X_measured = init_sigma[3:4,3:4]
    # init_mu_X_measured = init_mu_X
    D = coef*sigma_x * coef*sigma_x / Δt * I[1:2,1:2]
    translated_diff_sigma_XY = init_sigma_X_measured + (timesMinutes[timeOfInterest]*60)^2 * init_sigma_C_measured + timesMinutes[timeOfInterest]*60 * D
    translated_diff_sigma_XY1 = inv(translated_diff_sigma_XY)
    translated_cross_cov = timesMinutes[timeOfInterest]*60 * init_sigma_C_measured# + (D) * timesMinutes[timeOfInterest]*60/Δt
    translated_mu_XY = init_mu_X_measured + timesMinutes[timeOfInterest]*60 * init_mu_C_measured
    line_plot_diff_theory_vector = [translated_mu_C+translated_cross_cov*translated_diff_sigma_XY1*(line_plot_xy[i,:]-translated_mu_XY) for i in 1:line_plot_nPoints]
    # line_plot_diff_theory_vector = [translated_mu_C+translated_cross_cov*translated_diff_sigma_XY1*1.09*(line_plot_xy[i,:]-translated_mu_XY) for i in 1:line_plot_nPoints]
    #line_plot_diff_theory_values = [sqrt(sum(line_plot_diff_theory_vector[i].^2)) for i in 1:line_plot_nPoints]
    line_plot_diff_theory_values = [line_plot_diff_theory_vector[i][1] for i in 1:line_plot_nPoints]

    # Compute linereg through the data and plot it
    startPlot = 1
    peakSpeedDataFrame = DataFrame(x=line_plot_length[startPlot:end]./1000, y=line_plot_values[startPlot:end])
    peak_speed_linreg = lm(@formula(y~x), peakSpeedDataFrame)
    predict_peak_speed_lm = predict(peak_speed_linreg)

    axis_template = attr(autorange = true,
        showgrid = true, zeroline = false,
        linecolor = "black", showticklabels = true,
        ticks = ""
    )

    LatexFont = attr(color="black",family="Computer Modern",size=28)
    LatexFont2 = attr(color="black",family="Computer Modern",size=22)

    psd_layout = Layout(title=attr(text=L"\Large{\text{Peak speed along the axis of propagation}}",font=LatexFont)
                ,width=900, height=900
                ,xaxis_title=attr(text=L"\Large{\text{Position along the axis of propagation in } km}",font=LatexFont)
                ,yaxis_title=attr(text=L"\Large{\text{Peak speed of particles in } m/s}",font=LatexFont)
                ,legend=attr(borderwidth=1,x=0.1, y=0.9,font=LatexFont2)
                ,font=LatexFont
                ,xaxis=axis_template
                ,yaxis=axis_template
                ,margin=attr(l=90,r=80,t=80,b=100)
                ,template="simple_white"
    )


    q2 = plot([scatter(x=line_plot_length./1000,y=line_plot_values,name="Model output"),
                    # scatter(x=line_plot_length./1000,y=line_plot_theory_values, name="Theory without diffusion"),
                    scatter(x=line_plot_length./1000,y=line_plot_diff_theory_values, name="Theory with diffusion"),
                    scatter(x=line_plot_length[startPlot:end]./1000,y=predict_peak_speed_lm, name="Linear regression through the data")],psd_layout
    )

    if save_plots
        savefig(q2,"test_case_2_peak_speed_distribution.html")
    end

end

##  Computing statistical parameters

compute_spectrum_bool = false

if compute_spectrum_bool
    counts_1 = Int64.(round.(proba_spectrum_1*10000))
    nSample = sum(counts_1)
    sample_1=Array{Any,1}()

    for i in 1:nkX
        for j in 1:nkY
            for k in 1:counts_1[i,j]
                push!(sample_1, [kxs[i],kys[j]])
            end
        end
    end

    proba_spectrum_end = spectrum_end / sum(spectrum_end)
    proba_spectrum_end[max_speed_id[1],max_speed_id[2]] += 1-sum(proba_spectrum_end)

    counts_end = Int64.(round.(proba_spectrum_end*10000))
    nSample = sum(counts_end)
    sample_end=Array{Any,1}()

    for i in 1:nkX
        for j in 1:nkY
            for k in 1:counts_end[i,j]
                push!(sample_end, [kxs[i],kys[j]])
            end
        end
    end

    # proba_spectrum_right = spectrum_right / sum(spectrum_right)
    # proba_spectrum_right[max_speed_id[1],max_speed_id[2]] += 1-sum(proba_spectrum_right)

    # counts_right = Int64.(round.(proba_spectrum_right*10000))
    # nSample = sum(counts_right)
    # sample_right=Array{Any,1}()

    # for i in 1:nkX
    #     for j in 1:nkY
    #         for k in 1:counts_right[i,j]
    #             push!(sample_right, [kxs[i],kys[j]])
    #         end
    #     end
    # end

    # proba_spectrum_left = spectrum_left / sum(spectrum_left)
    # proba_spectrum_left[max_speed_id[1],max_speed_id[2]] += 1-sum(proba_spectrum_left)

    # counts_left = Int64.(round.(proba_spectrum_left*10000))
    # nSample = sum(counts_left)
    # sample_left=Array{Any,1}()

    # for i in 1:nkX
    #     for j in 1:nkY
    #         for k in 1:counts_left[i,j]
    #             push!(sample_left, [kxs[i],kys[j]])
    #         end
    #     end
    # end

    proba_spectrum_forw = spectrum_forw / sum(spectrum_forw)
    proba_spectrum_forw[max_speed_id[1],max_speed_id[2]] += 1-sum(proba_spectrum_forw)

    counts_forw = Int64.(round.(proba_spectrum_forw*10000))
    nSample = sum(counts_forw)
    sample_forw=Array{Any,1}()

    for i in 1:nkX
        for j in 1:nkY
            for k in 1:counts_forw[i,j]
                push!(sample_forw, [kxs[i],kys[j]])
            end
        end
    end

    proba_spectrum_back = spectrum_back / sum(spectrum_back)
    proba_spectrum_back[max_speed_id[1],max_speed_id[2]] += 1-sum(proba_spectrum_back)

    counts_back = Int64.(round.(proba_spectrum_back*10000))
    nSample = sum(counts_back)
    sample_back=Array{Any,1}()

    for i in 1:nkX
        for j in 1:nkY
            for k in 1:counts_back[i,j]
                push!(sample_back, [kxs[i],kys[j]])
            end
        end
    end

    std_1 = Stc.cov(sample_1)
    mean_1 = [Stc.mean([sample_1[i][j] for i in 1:length(sample_1)]) for j in 1:2]
    std_end = Stc.cov(sample_end)
    mean_end = [Stc.mean([sample_end[i][j] for i in 1:length(sample_end)]) for j in 1:2]
    # std_right = Stc.cov(sample_right)
    # mean_right = [Stc.mean([sample_right[i][j] for i in 1:length(sample_right)]) for j in 1:2]
    # std_left = Stc.cov(sample_left)
    # mean_left = [Stc.mean([sample_left[i][j] for i in 1:length(sample_left)]) for j in 1:2]
    std_forw = Stc.cov(sample_forw)
    mean_forw = [Stc.mean([sample_forw[i][j] for i in 1:length(sample_forw)]) for j in 1:2]
    std_back = Stc.cov(sample_back)
    mean_back = [Stc.mean([sample_back[i][j] for i in 1:length(sample_back)]) for j in 1:2]



    println()
    println("------------------------------------------------")
    println("                   RESULTS :")
    println()

    ## T=0
    print("At t=0, the center energy is at (")
    print(xyCenter_1[1])
    print(", ")
    print(xyCenter_1[2])
    println(")")
    print("The mean of the spectrum at this point is (")
    print(mean_1[1])
    print(", ")
    print(mean_1[2])
    println(")")
    println("And the standard deviation is :")
    print_mat(std_1)
    println()
    println()

    ## T=end  ->  Center
    print("At t=60, the center energy is at (")
    print(xyCenter_end[1])
    print(", ")
    print(xyCenter_end[2])
    println(")")
    print("The mean of the spectrum at this point is (")
    print(mean_end[1])
    print(", ")
    print(mean_end[2])
    println(")")
    println("And the standard deviation is :")
    print_mat(std_end)
    println()
    println()

    ## T=end  ->  Right
    # print("At t=60, the right point is at (")
    # print(xy_right[1])
    # print(", ")
    # print(xy_right[2])
    # println(")")
    # print("The mean of the spectrum at this point is (")
    # print(mean_right[1])
    # print(", ")
    # print(mean_right[2])
    # println(")")
    # println("And the standard deviation is :")
    # print_mat(std_right)
    # println()
    # println()

    # ## T=end  ->  Left
    # print("At t=60, the center energy is at (")
    # print(xy_left[1])
    # print(", ")
    # print(xy_left[2])
    # println(")")
    # print("The mean of the spectrum at this point is (")
    # print(mean_left[1])
    # print(", ")
    # print(mean_left[2])
    # println(")")
    # println("And the standard deviation is :")
    # print_mat(std_left)
    # println()
    # println()

    ## T=end  ->  Forward
    print("At t=60, the center energy is at (")
    print(xy_forw[1])
    print(", ")
    print(xy_forw[2])
    println(")")
    print("The mean of the spectrum at this point is (")
    print(mean_forw[1])
    print(", ")
    print(mean_forw[2])
    println(")")
    println("And the standard deviation is :")
    print_mat(std_forw)
    println()
    println()

    ## T=end  ->  Backward
    print("At t=60, the center energy is at (")
    print(xy_back[1])
    print(", ")
    print(xy_back[2])
    println(")")
    print("The mean of the spectrum at this point is (")
    print(mean_back[1])
    print(", ")
    print(mean_back[2])
    println(")")
    println("And the standard deviation is :")
    print_mat(std_back)
    println()
    println()
end

plot_test_case_3 = true

if plot_test_case_3
    sigmasComputeType = "multiplePoints" # "singlePoint" or "multiplePoints"
    recompute_sigmas_bool = true
    timesSeconds = 60*timesMinutes

    if recompute_sigmas_bool
        nSamplePoints = 30

        sigmas_cx = zeros(nTimes)
        sigmas_cy = zeros(nTimes)
        sigmas_cxy = zeros(nTimes)
        sigmas_cx_1 = zeros(nTimes)
        sigmas_cy_1 = zeros(nTimes)
        sigmas_cxy_1 = zeros(nTimes)
        sigmas_C = [zeros(2,2) for _ in 1:nTimes]
        sigmas_C_1 = [zeros(2,2) for _ in 1:nTimes]

        for i in 1:nTimes
            timeOfInterest = i
            translated_mu_C = [c_x_source, c_y_source]
            translated_mu_XY = [x_source, y_source] + timesMinutes[timeOfInterest]*60 * [c_x_source, c_y_source]
            translated_sigma_XY = init_sigma[3:4,3:4] + (timesMinutes[timeOfInterest]*60)^2 * init_sigma[1:2,1:2]
            translated_sigma_C = init_sigma[1:2,1:2]
            if sigmasComputeType=="multiplePoints"      # Version 2
                xy_center = translated_mu_XY
                xy_sigma = translated_sigma_XY
                # xy_center = timesSeconds[i].*init_mu_C+init_mu_X
                # xy_sigma = timesSeconds[i]^2 .* init_sigma[1:2,1:2] + init_sigma[3:4,3:4]
                d = MvNormal(xy_center, xy_sigma)
                sampleXYs = rand(d,nSamplePoints)
                sampleValues = []
                sampleWeights = []
                # print("computing for sample point number ")
                for j in 1:nSamplePoints
                    # print(", "*string(j))
                    res, weight = compute_spectrum_parameters(i, ParticleStates, sampleXYs[:,j], kernel, "gauss")
                    if res != [0 0; 0 0]
                        append!(sampleValues, [res])
                        append!(sampleWeights, [weight])
                    else
                        print("FOUND ZERO; ")
                    end
                end
                sigmas_C[i] = sum(sampleWeights.*sampleValues)/sum(sampleWeights)
            elseif sigmasComputeType=="singlePoint"     # Version 1
                xy_center = translated_mu_XY
                xy_sigma = translated_sigma_XY
                xy_chosen = xy_center + sqrt.([translated_sigma_XY[1,1], translated_sigma_XY[2,2]])
                sigmas_C[i] = compute_spectrum_parameters(i, ParticleStates, xy_chosen, kernel, "gauss")[1]
            end

            sigmas_C_1[i] = inv(sigmas_C[i])

            sigmas_cx[i] = sigmas_C[i][1,1]
            sigmas_cy[i] = sigmas_C[i][2,2]
            sigmas_cxy[i] = sigmas_C[i][2,1]

            sigmas_cx_1[i] = sigmas_C_1[i][1,1]
            sigmas_cy_1[i] = sigmas_C_1[i][2,2]
            sigmas_cxy_1[i] = sigmas_C_1[i][2,1]
            if i%10==0 || i==nTimes
                println("Computing sigma for t = "*string(timesMinutes[i]))
            end
        end
    end

    time_long_index = 32

    data_sigma_cx = DataFrame(x=timesSeconds[time_long_index:end], y=sigmas_cx[time_long_index:end].^-1)
    ab_sigma_cx = lm(@formula(y~x), data_sigma_cx)
    data_sigma_cy = DataFrame(x=timesSeconds[time_long_index:end], y=sigmas_cy[time_long_index:end].^-1)
    ab_sigma_cy = lm(@formula(y~x), data_sigma_cy)

    pearson_sigma_cx = pearson(sigmas_cx[time_long_index:end].^-1, predict(ab_sigma_cx))
    pearson_sigma_cy = pearson(sigmas_cy[time_long_index:end].^-1, predict(ab_sigma_cy))

    trace1 = scatter(x=timesSeconds,y=sigmas_cx.^-1,name="sigma c_x")
    trace2 = scatter(x=timesSeconds,y=sigmas_cy.^-1,name="sigma c_y")
    trace3 = scatter(x=timesSeconds[time_long_index:end],y=predict(ab_sigma_cx),name="linreg through sigma c_x; R2="*string(round(pearson_sigma_cx,digits=3)))
    trace4 = scatter(x=timesSeconds[time_long_index:end],y=predict(ab_sigma_cy),name="linreg through sigma c_y; R2="*string(round(pearson_sigma_cy,digits=3)))

    trace5 = scatter(x=timesSeconds,y=sigmas_cx.^1,name=L"\text{Model output }(\sigma_{c_x})^2", mode="markers", marker=attr(color="blue"))
    trace6 = scatter(x=timesSeconds,y=sigmas_cy.^1,name=L"\text{Model output }(\sigma_{c_y})^2", mode="markers", marker=attr(color="orange"))

    init_sigma_1 = inv(init_sigma)

    init_sigma_C_1 = init_sigma_1[1:2,1:2]
    # init_sigma_C_1 = 1 ./(init_sigma[1:2,1:2])
    init_sigma_X_1 = init_sigma_1[3:4,3:4]
    # init_sigma_X_1 = 1 ./(init_sigma[3:4,3:4])

    sigmas_C_t = [inv(t^2*init_sigma_X_1^1 .+ init_sigma_C_1) for t in timesSeconds]
    sigmas_cx_t = [sigmas_C_t[t][1,1] for t in 1:length(timesSeconds)]
    sigmas_cy_t = [sigmas_C_t[t][2,2] for t in 1:length(timesSeconds)]

    trace7 = scatter(x=timesSeconds,y=sigmas_cx_t, name=L"\text{Theoretical } (\sigma_{c_x})^2", mode="lines", line=attr(dash="dash", color="blue"))
    trace8 = scatter(x=timesSeconds,y=sigmas_cy_t, name=L"\text{Theoretical } (\sigma_{c_y})^2", mode="lines", line=attr(dash="dash", color="orange"))

    log_sigma_cx = DataFrame(x=log.(timesSeconds[time_long_index:end]), y=log.(sigmas_cx[time_long_index:end]))
    ab_log_sigma_cx = lm(@formula(y~x), log_sigma_cx)
    r2_log_sigma_cx = pearson(log_sigma_cx.y,predict(ab_log_sigma_cx))
    log_sigma_cy = DataFrame(x=log.(timesSeconds[time_long_index:end]), y=log.(sigmas_cy[time_long_index:end]))
    ab_log_sigma_cy = lm(@formula(y~x), log_sigma_cy)
    r2_log_sigma_cy = pearson(log_sigma_cy.y,predict(ab_log_sigma_cy))

    trace9 = scatter(x=timesSeconds[time_long_index:end], y=exp.(predict(ab_log_sigma_cx)))
    trace10 = scatter(x=timesSeconds[time_long_index:end], y=exp.(predict(ab_log_sigma_cy)))

    trace11 = scatter(x=timesSeconds[time_long_index:end], y=20000000 .* timesSeconds[time_long_index:end].^-2)

    # TESTING OTHER CURVES

    #     Version 1
    # alpha_mu_Cx = init_mu_C[1]/(sigma_x/Δt) - floor(init_mu_C[1]/(sigma_x/Δt))
    # beta_sigma_Cx = sqrt(init_sigma[1,1])/(sigma_x/Δt) - floor(sqrt(init_sigma[1,1])/(sigma_x/Δt))
    # alpha_mu_Cy = init_mu_C[2]/(sigma_y/Δt) - floor(init_mu_C[2]/(sigma_y/Δt))
    # beta_sigma_Cy = sqrt(init_sigma[2,2])/(sigma_y/Δt) - floor(sqrt(init_sigma[2,2])/(sigma_y/Δt))

    # coefx = (alpha_mu_Cx-(alpha_mu_Cx^2+beta_sigma_Cx^2))
    # coefy = (alpha_mu_Cy*(1-alpha_mu_Cy/2)+beta_sigma_Cy*(sqrt(2/pi)-beta_sigma_Cy/2))
    # D = sigma_x*sigma_x / Δt * [coefx 0; 0 coefy]

        # Version 2
    # init_mu_cx_corrected = (init_mu_C[1]*Δt/sigma_x - floor.(init_mu_C[1]*Δt/sigma_x))/Δt*sigma_x
    # init_mu_cy_corrected = (init_mu_C[2]*Δt/sigma_y - floor.(init_mu_C[2]*Δt/sigma_y))/Δt*sigma_y
    # init_sigma_cx_corrected = sqrt(init_sigma[1,1])
    # init_sigma_cy_corrected = sqrt(init_sigma[2,2])
    # # init_sigma_cx_corrected = ((sqrt(init_sigma[1,1])*Δt/sigma_x - floor.(sqrt(init_sigma[1,1])*Δt/sigma_x))/Δt*sigma_x)^2
    # # init_sigma_cy_corrected = ((sqrt(init_sigma[2,2])*Δt/sigma_y - floor.(sqrt(init_sigma[2,2])*Δt/sigma_y))/Δt*sigma_y)^2
    # localx = -init_mu_cx_corrected/(init_sigma_cx_corrected)
    # localy = -init_mu_cy_corrected/(init_sigma_cy_corrected)
    # phi_x = 1/sqrt(2*pi)*exp(-0.5*(localx)^2)
    # phi_y = 1/sqrt(2*pi)*exp(-0.5*(localy)^2)
    # Phi_x = 0.5*(1+erf(localx/sqrt(2)))
    # Phi_y = 0.5*(1+erf(localy/sqrt(2)))

    # mu_c_upper_x = init_mu_cx_corrected + ((init_sigma_cx_corrected) * phi_x)/(1-Phi_x)
    # mu_c_lower_x = init_mu_cx_corrected - ((init_sigma_cx_corrected) * phi_x)/(Phi_x)
    # sigma_c_upper_x = init_sigma_cx_corrected * sqrt(1 + (localx* phi_x)/(1 - Phi_x) - (phi_x/(1 - Phi_x))^2)
    # sigma_c_lower_x = init_sigma_cx_corrected * sqrt(1 - (localx* phi_x)/(Phi_x) - (phi_x/(Phi_x))^2)

    # mu_c_upper_y = init_mu_cy_corrected + (sqrt(init_sigma_cy_corrected) * phi_y)/(1-Phi_y)
    # mu_c_lower_y = init_mu_cy_corrected - (sqrt(init_sigma_cy_corrected) * phi_y)/(Phi_y)
    # sigma_c_upper_y = sqrt(init_sigma_cy_corrected * (1 - (localy* phi_y)/(1 - Phi_y) - (phi_y/(1-Phi_y))^2))
    # sigma_c_lower_y = sqrt(init_sigma_cy_corrected * (1 + (localy* phi_y)/(Phi_y) - (phi_y/(Phi_y))^2))

    # coefx = (1-Phi_x)*(sigma_x*mu_c_upper_x-Δt*(sigma_c_upper_x^2+mu_c_upper_x^2))+Phi_x*(-sigma_x*mu_c_lower_x-Δt*(sigma_c_lower_x^2+mu_c_lower_x^2))
    # coefy = (1-Phi_y)*(sigma_y*mu_c_upper_y-Δt*(sigma_c_upper_y^2+mu_c_upper_y^2))+Phi_y*(-sigma_y*mu_c_lower_y-Δt*(sigma_c_lower_y^2+mu_c_lower_y^2))
    # D = [coefx 0; 0 coefy]

        # Version 3
    # Phi = (x,mu,sigma) -> 0.5*(1+erf((x-mu)/(sigma*sqrt(2))))
    # phi = (x,mu,sigma) -> 1/(sqrt(2*pi))*exp(-0.5*(x-mu)^2/(sigma^2))
    # p=0.0025


    # cmin_x = init_mu_C[1] + sqrt(init_sigma[1,1]*2)*erfinv(2*p-1)
    # cmax_x = init_mu_C[1] + sqrt(init_sigma[1,1]*2)*erfinv(2*(1-p)-1)
    # k_x = Int64(floor(cmin_x*Δt/sigma_x))
    # l_x = Int64(floor(cmax_x*Δt/sigma_x))
    # pos_cx = [(i)*sigma_x/Δt for i in k_x:(l_x+1)]

    # indexes_x = (k_x:l_x)
    # probs_x = [(Phi(pos_cx[i+1], init_mu_C[1], sqrt(init_sigma[1,1]))-Phi(pos_cx[i], init_mu_C[1], sqrt(init_sigma[1,1]))) for i in 1:(l_x-k_x+1)]
    # means_x = [tn.tnmean(pos_cx[i],pos_cx[i+1],init_mu_C[1],sqrt(init_sigma[1,1])) for i in 1:(l_x-k_x+1)]
    # vars_x = [tn.tnvar(pos_cx[i],pos_cx[i+1],init_mu_C[1],sqrt(init_sigma[1,1])) for i in 1:(l_x-k_x+1)]
    # coefs_x = probs_x.*(
    #     sigma_x .* (means_x - indexes_x.*(sigma_x/Δt))
    #     - Δt .* (
    #         vars_x .^ 2 + means_x .^ 2
    #         - 2 .* (indexes_x) .* (Δt / sigma_x) .* means_x
    #         + ((indexes_x) * sigma_x / Δt) .^ 2
    #     )
    # )
    # coefx = sum(coefs_x)


    # cmin_y = init_mu_C[2] + sqrt(init_sigma[2,2]*2)*erfinv(2*p-1)
    # cmax_y = init_mu_C[2] + sqrt(init_sigma[2,2]*2)*erfinv(2*(1-p)-1)
    # k_y = Int64(floor(cmin_y*Δt/sigma_y))
    # l_y = Int64(floor(cmax_y*Δt/sigma_y))
    # pos_cy = [(i)*sigma_y/Δt for i in k_y:(l_y+1)]

    # probs_y = [(Phi(pos_cy[i+1], init_mu_C[2], sqrt(init_sigma[2,2]))-Phi(pos_cy[i], init_mu_C[2], sqrt(init_sigma[2,2]))) for i in 1:(l_y-k_y+1)]
    # means_y = [tn.tnmean(pos_cy[i],pos_cy[i+1],init_mu_C[2],sqrt(init_sigma[2,2])) for i in 1:(l_y-k_y+1)]
    # # means_y = (means_y.*Δt./sigma_y - floor.(means_y.*Δt./sigma_y))./Δt.*sigma_y
    # vars_y = [tn.tnvar(pos_cy[i],pos_cy[i+1],init_mu_C[2],sqrt(init_sigma[2,2])) for i in 1:(l_y-k_y+1)]
    # coefs_y = 1
    # coefy = sum(coefs_y)

    # # coefx = 884.5
    # # coefy = 599.96
    # D = [coefx 0; 0 coefy]

        # Version 4
    p=1e-5
    nIntPoints = 100000
    f = (c) -> c - floor(c)
    g = (c) -> c^2 - 2 * c * floor(c) + floor(c)^2
    compute_Dc = (c,DeltaX,DeltaT) -> DeltaX * f(c/DeltaX*DeltaT) * (DeltaX/DeltaT) - DeltaT * g(c*DeltaT/DeltaX) * (DeltaX/DeltaT)^2

    cmin_x = init_mu_C[1] + sqrt(init_sigma[1,1]*2)*erfinv(2*p-1)
    cmax_x = init_mu_C[1] + sqrt(init_sigma[1,1]*2)*erfinv(2*(1-p)-1)
    pos_cx = (((0:nIntPoints)./nIntPoints).*(cmax_x-cmin_x)).+cmin_x
    DeltaCx = pos_cx[2]-pos_cx[1]
    probas_cx = gaussD.(init_mu_C[1],init_sigma[1,1],pos_cx)
    D_cx_values = compute_Dc.(pos_cx, sigma_x, Δt)
    D_cx_integrand = probas_cx .* D_cx_values
    coefx = sum((D_cx_integrand[1:(end-1)]+D_cx_integrand[2:(end)])/2)*DeltaCx

    cmin_y = init_mu_C[2] + sqrt(init_sigma[2,2]*2)*erfinv(2*p-1)
    cmax_y = init_mu_C[2] + sqrt(init_sigma[2,2]*2)*erfinv(2*(1-p)-1)
    pos_cy = (((0:nIntPoints)./nIntPoints).*(cmax_y-cmin_y)).+cmin_y
    DeltaCy = pos_cy[2]-pos_cy[1]
    probas_cy = gaussD.(init_mu_C[2],init_sigma[2,2],pos_cy)
    D_cy_values = compute_Dc.(pos_cy, sigma_y, Δt)
    D_cy_integrand = probas_cy .* D_cy_values
    coefy = sum((D_cy_integrand[1:(end-1)]+D_cy_integrand[2:(end)])/2)*DeltaCy

    D = [coefx 0 ; 0 coefy]


    timesSeconds2 = (0:(nTimes-1))./(nTimes-1).*(timesSeconds[end])

    sigmas_C_t_new = [inv(t^2 .* inv(t .* D+ init_sigma[3:4,3:4]) .+ init_sigma_C_1) for t in timesSeconds2]
    sigmas_cx_t_new = [sigmas_C_t_new[t][1,1] for t in 1:length(timesSeconds2)]
    sigmas_cy_t_new = [sigmas_C_t_new[t][2,2] for t in 1:length(timesSeconds2)]

    derivSigmas_D_C_t_th = [(log.(sigmas_C_t_new[t]).-log.(sigmas_C_t_new[t-1]))./(log(timesSeconds2[t]).-log(timesSeconds2[t-1])) for t in 2:length(timesSeconds2)]
    derivSigmas_D_cx_t_th = [derivSigmas_D_C_t_th[t][1,1] for t in 1:(length(timesSeconds2)-1)]
    derivSigmas_D_cy_t_th = [derivSigmas_D_C_t_th[t][2,2] for t in 1:(length(timesSeconds2)-1)]

    newTimes, derivSigmas_D_cx_t = compute_derivative(log.(sigmas_cx),log.(timesSeconds), 1)
    newTimes, derivSigmas_D_cy_t = compute_derivative(log.(sigmas_cy),log.(timesSeconds), 1)
    newTimes = exp.(newTimes)

    trace12 = scatter(x=timesSeconds2, y=sigmas_cx_t_new, name=L"\text{Diffused } (\sigma_{c_x})^2", marker=attr(color="rgb(0, 98, 255)"))
    trace13 = scatter(x=timesSeconds2, y=sigmas_cy_t_new, name=L"\text{Diffused } (\sigma_{c_y})^2", marker=attr(color="rgb(226, 45, 0)"))

    trace14 = scatter(x=timesSeconds2[2:end], y=derivSigmas_D_cx_t_th)
    trace15 = scatter(x=timesSeconds2[2:end], y=derivSigmas_D_cy_t_th)

    trace16 =  scatter(x=newTimes, y=derivSigmas_D_cx_t)
    trace17 =  scatter(x=newTimes, y=derivSigmas_D_cy_t)

    α = 1000
    β = 30000000
    γ = 17500000
    perfect_1_r = [α/t for t in timesSeconds2[10:end]]
    perfect_1_r_2 = [β/t^2 for t in timesSeconds2[10:end]]
    perfect_1_r_2_2 = [γ/t^2 for t in timesSeconds2[30:end]]

    trace_perfect_1_r = scatter(x=timesSeconds2[10:end],y=perfect_1_r,
        name="\$\\frac{1}{t} \\text{ curve for reference}\$"
        , mode="lines", line=attr(dash="dot", color="rgb(112, 187, 112)",width=4)
    )
    trace_perfect_1_r_2 = scatter(x=timesSeconds2[10:end],y=perfect_1_r_2,
        name="\$\\frac{1}{t^2} \\text{ curve for reference}\$"
        , mode="lines", line=attr(dash="dot", color="rgb(181, 105, 178)",width=4)
    )
    trace_perfect_1_r_2_2 = scatter(x=timesSeconds2[30:end],y=perfect_1_r_2_2,
        name="\$\\frac{1}{t^2} \\text{ curve for reference}\$"
        , mode="lines", line=attr(dash="dot", color="rgb(181, 105, 178)",width=4)
    )


    # END TESTING OTHER CURVES

    axis_template = attr(autorange = true,
        showgrid = true, zeroline = false,
        linecolor = "black", showticklabels = true,
        ticks = ""
    )

    axis_template_x_2 = attr(showgrid = true, zeroline = false,
        linecolor = "black", showticklabels = true,
        ticks = "", range = log10.([2500, 7500])
    )

    axis_template_y_2 = attr(showgrid = true, zeroline = false,
        linecolor = "black", showticklabels = true,
        ticks = "", range = log10.([0.12, 1.34])
    )

    LatexFont = attr(color="black",family="Computer Modern",size=28)
    LatexFontlegend = attr(color="black",family="Computer Modern",size=40)

    sn_layout = Layout(yaxis_type="log",xaxis_type="log",
                title=attr(text=L"\LARGE{\text{Evolution of the local } C_x \text{ energy spectrum variance in time}}",font=LatexFont)
                ,width=1600, height=1000
                ,xaxis_title=L"\LARGE{\text{time } (sec)}"
                ,yaxis_title=L"\LARGE{(\sigma_{c})^2}"
                #,legend=attr(borderwidth=1,x=0.5, y=0.5,font=LatexFont)
                ,legend=attr(borderwidth=1,x=0.05, y=0.05,font=LatexFontlegend)
                ,font=LatexFont
                ,xaxis=axis_template
                ,yaxis=axis_template
                ,margin=attr(l=90,r=80,t=80,b=140)
                ,template="simple_white"
    )

    sn_layout2 = Layout(yaxis_type="log",xaxis_type="log",
                title=attr(text=L"\LARGE{\text{Evolution of the local } C_y \text{ energy spectrum variance in time}}",font=LatexFont)
                ,width=1600, height=1000
                ,xaxis_title=L"\LARGE{\text{time } (sec)}"
                ,yaxis_title=L"\LARGE{(\sigma_{c})^2}"
                #,legend=attr(borderwidth=1,x=0.5, y=0.5,font=LatexFont)
                ,legend=attr(borderwidth=1,x=0.05, y=0.05,font=LatexFontlegend)
                ,font=LatexFont
                ,xaxis=axis_template
                ,yaxis=axis_template
                ,margin=attr(l=90,r=80,t=80,b=140)
                ,template="simple_white"
    )

    tc3_1 = plot([trace5, trace7, trace12, trace_perfect_1_r, trace_perfect_1_r_2
                ],sn_layout
    )

    tc3_2 = plot([trace6, trace8, trace13, trace_perfect_1_r, trace_perfect_1_r_2_2
                ],sn_layout2
    )

    save_plots = true
    if save_plots
        savefig(tc3_1,"test_case_2_spectral_peak_narrowing_cx.html")
        savefig(tc3_2,"test_case_2_spectral_peak_narrowing_cy.html")
        # savefig(tc3_3,"test_case_2_t3.html")
    end

    tc3_3 = plot([trace14, trace16, trace17]
                    ,Layout(xaxis_type="log"
                    ,title="LogSlope of the spectrum variance in time"
            )
    )

    tc3_4 = plot([trace1, trace2, trace3, trace4],Layout(#yaxis_type="log",xaxis_type="log",
                title="Evolution of the local energy spectrum variance in time"
                ,width=1000, height=1000
                ,xaxis_title="\\text{time} (sec)"
                ,yaxis_title="sigma_c"
                ,legend=attr(borderwidth=3,x=0.1, y=0.8,font=attr(size=18))
                ,font=attr(size=20)
                ,template="simple_white"
            )
    )

    # Plotting the space distribution

    plotting_times = [2, Int64(floor(size(MeshState)[1]/2))+2, size(MeshState)[1]]

    plotLog = false
    colorbarDelay = 20
    width = 930
    height = 300
    showTitle = false
    if plotLog
        cRange = [-10, maximum(Matrix(log.(MeshState[plotting_times[1]+colorbarDelay, "mesh_state"])))]

        tc3_t1 = plot_state_and_error_points(log.(MeshState[plotting_times[1], "mesh_state"]), xs./1000, ys./1000, cRange, width, height; showTitle)
        tc3_t2 = plot_state_and_error_points(log.(MeshState[plotting_times[2], "mesh_state"]), xs./1000, ys./1000, cRange, width, height; showTitle)
        tc3_t3 = plot_state_and_error_points(log.(MeshState[plotting_times[3], "mesh_state"]), xs./1000, ys./1000, cRange, width, height; showTitle)
    else
        cRange = [0, maximum(Matrix(MeshState[plotting_times[2]+colorbarDelay, "mesh_state"]))]

        tc3_t1 = plot_state_and_error_points(MeshState[plotting_times[1], "mesh_state"], xs./1000, ys./1000, cRange, width, height; showTitle)
        tc3_t2 = plot_state_and_error_points(MeshState[plotting_times[2], "mesh_state"], xs./1000, ys./1000, cRange, width, height; showTitle)
        tc3_t3 = plot_state_and_error_points(MeshState[plotting_times[3], "mesh_state"], xs./1000, ys./1000, cRange, width, height; showTitle)
    end

    save_plots = true
    if save_plots
        savefig(tc3_t1,"test_case_3_t1.html")
        savefig(tc3_t2,"test_case_3_t2.html")
        savefig(tc3_t3,"test_case_3_t3.html")
        println("Plotting times for the test case 3 energy distribution : "*string(MeshState[plotting_times[1], "times"])*", "*string(MeshState[plotting_times[2], "times"])*", "*string(MeshState[plotting_times[3], "times"]))
    end

end