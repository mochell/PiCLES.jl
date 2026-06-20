"""
Unit tests for the generalized B-spline particle->grid deposition (issues #59, #60).

Covers:
1. Partition of unity (Σw = 1) for orders P = 1, 2, 3, both the absolute and the
   particle-node-relative weight kernels -> guarantees exact charge/energy conservation.
2. P = 1 regression: the Val(1) kernels are bit-identical to the original linear
   `get_absolute_i_and_w`, and a single-particle deposit at Val(1) reproduces the original
   4-node CIC stencil exactly.
3. Higher orders deposit onto the expected wider, conservative stencil.
"""

using Test
using StaticArrays
using SharedArrays

import PiCLES.ParticleInCell as PIC
using PiCLES.custom_structures: N_Periodic

@testset "B-spline deposition kernels" begin

    # Keep this grid construction independent of Base.floatrange internals so it works
    # across Julia versions used in CI.
    z0 = -7.3
    z1 = 9.7
    Nz = 137
    dz = (z1 - z0) / (Nz - 1)
    zs = [z0 + (k - 1) * dz for k in 1:Nz]

    @testset "partition of unity (absolute + relative), P=$(P)" for P in (1, 2, 3)
        for z in zs
            _, w = PIC.get_absolute_i_and_w(z, Val(P))
            @test length(w) == P + 1
            @test sum(w) ≈ 1.0 atol = 1e-12

            for i_node in (1, 7, 23)
                ir, wr = PIC.get_absolute_i_and_w(z, i_node, Val(P))
                @test length(wr) == P + 1
                @test sum(wr) ≈ 1.0 atol = 1e-12
                # relative indices are the absolute ones shifted by the node position
                ia, _ = PIC.get_absolute_i_and_w(z, Val(P))
                @test ir == ia .+ (i_node - 1)
            end
        end
    end

    @testset "P=1 bit-identical to legacy linear kernel" begin
        for z in zs
            i1, w1 = PIC.get_absolute_i_and_w(z, Val(1))
            i0, w0 = PIC.get_absolute_i_and_w(z)
            @test i1 == i0
            @test w1 == w0

            for i_node in (1, 7, 23)
                ir1, wr1 = PIC.get_absolute_i_and_w(z, i_node, Val(1))
                ir0, wr0 = PIC.get_absolute_i_and_w(z, i_node)
                @test ir1 == ir0
                @test wr1 == wr0
            end
        end
    end

    @testset "non-negative weights" for P in (1, 2, 3)
        for z in zs
            _, w = PIC.get_absolute_i_and_w(z, Val(P))
            @test all(w .>= 0.0)
        end
    end
end

@testset "B-spline single-particle deposit" begin
    # Small periodic grid; deposit one particle well inside so even the cubic stencil stays
    # interior. charge = (energy, mom_x, mom_y).
    Nx, Ny = 16, 16
    Bx, By = N_Periodic(Nx), N_Periodic(Ny)
    charge = SVector{3,Float64}(2.5, 0.3, -0.7)
    ij = (8, 8)
    xp, yp = 0.37, 0.62   # fractional offsets relative to the particle node

    @testset "total deposited charge conserved, P=$(P)" for P in (1, 2, 3)
        S = SharedArray{Float64,3}((Nx, Ny, 3))
        S .= 0.0
        wni = PIC.compute_weights_and_index_mininal(ij, xp, yp, Val(P))
        PIC.push_to_grid!(S, charge, wni, Bx, By)
        for c in 1:3
            @test sum(S[:, :, c]) ≈ charge[c] atol = 1e-12
        end
        # number of touched nodes is (P+1)^2 at most
        @test count(!iszero, S[:, :, 1]) <= (P + 1)^2
    end

    @testset "P=1 reproduces the 4-node CIC stencil exactly" begin
        S = SharedArray{Float64,3}((Nx, Ny, 3))
        S .= 0.0
        wni = PIC.compute_weights_and_index_mininal(ij, xp, yp, Val(1))
        PIC.push_to_grid!(S, charge, wni, Bx, By)

        # legacy CIC: node ij is floor; weights (1-δ, δ) per axis with δ = round(frac, 6)
        dx = round(xp - floor(xp), digits=6)
        dy = round(yp - floor(yp), digits=6)
        i0 = ij[1] + Int(floor(xp))
        j0 = ij[2] + Int(floor(yp))
        expected = Dict(
            (i0, j0)         => (1 - dx) * (1 - dy),
            (i0 + 1, j0)     => dx * (1 - dy),
            (i0, j0 + 1)     => (1 - dx) * dy,
            (i0 + 1, j0 + 1) => dx * dy,
        )
        for ((i, j), wexp) in expected
            @test S[i, j, 1] ≈ wexp * charge[1] atol = 1e-12
        end
        @test count(!iszero, S[:, :, 1]) == 4
    end
end
