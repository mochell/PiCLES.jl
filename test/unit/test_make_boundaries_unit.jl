using Test

using PiCLES
using PiCLES.custom_structures: N_Periodic, N_NonPeriodic, N_TripolarNorth

@testset "make_boundaries boundary-type behavior" begin
    base_mask = trues(7, 7)
    base_mask[3:5, 3:5] .= false

    @testset "non-periodic marks outer grid boundary" begin
        total_mask = PiCLES.Grids.make_boundaries(base_mask, N_NonPeriodic(7), N_NonPeriodic(7))

        @test all(total_mask[1, :] .== 3)
        @test all(total_mask[end, :] .== 3)
        @test all(total_mask[:, 1] .== 3)
        @test all(total_mask[:, end] .== 3)

        @test total_mask[4, 4] == 0
        @test total_mask[3, 4] == 2
        @test total_mask[5, 4] == 2
        @test total_mask[4, 3] == 2
        @test total_mask[4, 5] == 2
        @test total_mask[2, 4] == 1
    end

    @testset "periodic does not mark outer boundary" begin
        total_mask = PiCLES.Grids.make_boundaries(base_mask, N_Periodic(7), N_Periodic(7))

        @test !any(total_mask .== 3)
        @test total_mask[4, 4] == 0
        @test total_mask[3, 4] == 2
        @test total_mask[5, 4] == 2
        @test total_mask[4, 3] == 2
        @test total_mask[4, 5] == 2
        @test total_mask[2, 4] == 1
    end

    @testset "tripolar north is not forced to grid boundary" begin
        total_mask = PiCLES.Grids.make_boundaries(base_mask, N_Periodic(7), N_TripolarNorth(7))

        @test !any(total_mask .== 3)
        @test total_mask[4, 4] == 0
        @test total_mask[3, 4] == 2
        @test total_mask[5, 4] == 2
        @test total_mask[4, 3] == 2
        @test total_mask[4, 5] == 2
        @test total_mask[2, 4] == 1
    end
end