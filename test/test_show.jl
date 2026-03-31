using Test
using ContourDynamics
using StaticArrays

@testset "Pretty Printing" begin
    @testset "Kernel show" begin
        # EulerKernel
        s = repr("text/plain", EulerKernel())
        @test s == "EulerKernel"

        # QGKernel
        s = repr("text/plain", QGKernel(2.5))
        @test s == "QGKernel{Float64}: Ld = 2.5"

        # SQGKernel
        s = repr("text/plain", SQGKernel(0.01))
        @test s == "SQGKernel{Float64}: δ = 0.01"

        # MultiLayerQGKernel compact
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)
        mk = MultiLayerQGKernel(Ld, coupling)
        s_compact = repr(mk)
        @test s_compact == "MultiLayerQGKernel{2, Float64}"

        # MultiLayerQGKernel rich
        s_rich = repr("text/plain", mk)
        @test occursin("MultiLayerQGKernel{2, Float64}", s_rich)
        @test occursin("├── Ld:", s_rich)
        @test occursin("├── coupling: 2×2 SMatrix{Float64}", s_rich)
        @test occursin("└── eigenvalues:", s_rich)
    end
end
