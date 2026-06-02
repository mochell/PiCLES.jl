#using PiCLES
using Test

@testset "PiCLES.jl" begin
    include("T02_update_forcing_step.jl")
    include("T07_forcing_field_types.jl")
end
