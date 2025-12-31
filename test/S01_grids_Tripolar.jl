
using Pkg
Pkg.activate("PiCLES/")

using PiCLES.Grids.TripolarGridMOM6: TripolarGridMOM6, ProjetionKernel
using Plots

using Revise
using BenchmarkTools
Revise.retry()

load_path = "PiCLES/src/Grids/files/"

# %%
for i in [2, 4, 6, 8]
    Gtest = TripolarGridMOM6.MOM6GridMesh(load_path * "ocean_hgrid_221123.nc", i)
    p = plot()
    scatter!(p, Gtest.data.x[Gtest.data.mask], Gtest.data.y[Gtest.data.mask], color="black", legend=false, markersize=1)
    display(p)
end

# %%

Revise.retry()

for i in [2,4,6,8]
    Gtest = TripolarGridMOM6.MOM6GridMesh(load_path * "ocean_hgrid_221123.nc", i, MaskFile=load_path * "ocean_topo_tx2_3v2_240501.nc")
    p = plot()
    scatter!(p, Gtest.data.x[Gtest.data.mask], Gtest.data.y[Gtest.data.mask], color="black", legend=false, markersize=1)
    display(p)
end

# %% test projection kernel 
G = TripolarGridMOM6.MOM6GridMesh(load_path * "ocean_hgrid_221123.nc", 4, MaskFile=load_path * "ocean_topo_tx2_3v2_240501.nc")
@benchmark M = ProjetionKernel(G)
M = ProjetionKernel(G)

# this shows all potions where the kernel is nothing
scatter( G.data.x[isnothing.(M)], G.data.y[isnothing.(M)], color="black", legend=false, markersize=1)

