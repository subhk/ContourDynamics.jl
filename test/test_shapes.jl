using ContourDynamics
using Test

@testset "Shape Helpers" begin
    @testset "circular_patch" begin
        c = circular_patch(0.5, 32, 2π)
        @test c isa PVContour{Float64}
        @test nnodes(c) == 32
        @test c.pv == 2π

        # Nodes lie on circle of radius 0.5
        for i in 1:nnodes(c)
            r = sqrt(c.nodes[i][1]^2 + c.nodes[i][2]^2)
            @test r ≈ 0.5 atol=1e-12
        end

        # Center offset
        c2 = circular_patch(1.0, 16, 1.0; cx=2.0, cy=3.0)
        center = sum(c2.nodes) / nnodes(c2)
        @test center[1] ≈ 2.0 atol=1e-10
        @test center[2] ≈ 3.0 atol=1e-10

        # Float32
        c32 = circular_patch(0.5, 16, 1.0; T=Float32)
        @test c32 isa PVContour{Float32}

        # Numeric args auto-promoted
        c_int = circular_patch(1, 16, 1)
        @test c_int isa PVContour{Float64}
    end

    @testset "elliptical_patch" begin
        e = elliptical_patch(2.0, 1.0, 64, 1.0)
        @test nnodes(e) == 64
        @test e.pv == 1.0

        # Area ≈ π*a*b = 2π
        @test vortex_area(e) ≈ 2π rtol=0.01

        # With rotation
        e_rot = elliptical_patch(2.0, 1.0, 64, 1.0; θ=π/4)
        @test vortex_area(e_rot) ≈ 2π rtol=0.01
        _, angle = ellipse_moments(e_rot)
        @test angle ≈ π/4 atol=0.1

        # Center offset
        e2 = elliptical_patch(1.0, 0.5, 32, 1.0; cx=1.0, cy=-1.0)
        center = sum(e2.nodes) / nnodes(e2)
        @test center[1] ≈ 1.0 atol=1e-10
        @test center[2] ≈ -1.0 atol=1e-10
    end

    @testset "rankine_vortex" begin
        v = rankine_vortex(1.0, 64, 2π)
        @test v isa Vector{PVContour{Float64}}
        @test length(v) == 1
        @test v[1].pv ≈ 2π / (π * 1.0^2)  # Γ / (π R²)
        @test nnodes(v[1]) == 64

        # Nodes on unit circle
        for i in 1:nnodes(v[1])
            r = sqrt(v[1].nodes[i][1]^2 + v[1].nodes[i][2]^2)
            @test r ≈ 1.0 atol=1e-12
        end

        # Center offset
        v2 = rankine_vortex(0.5, 32, 1.0; cx=1.0, cy=2.0)
        center = sum(v2[1].nodes) / nnodes(v2[1])
        @test center[1] ≈ 1.0 atol=1e-10
        @test center[2] ≈ 2.0 atol=1e-10
    end
end
