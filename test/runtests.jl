using Test

@testset "PiCLES.jl" begin
    include("unit/test_make_boundaries_unit.jl")
    include("unit/test_update_forcing_step.jl")
    include("unit/test_forcing_field_types.jl")
    include("unit/test_single_particle_2d_local_winds_smoke.jl")
    include("unit/test_single_particle_1d_alias_smoke.jl")
    include("unit/test_grids_homogeneous_forcing_smoke.jl")
    include("unit/test_pic_1d_charge_conservation.jl")
end
