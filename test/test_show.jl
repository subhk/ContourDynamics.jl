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

    @testset "Domain show" begin
        @test repr("text/plain", UnboundedDomain()) == "UnboundedDomain"

        pd = PeriodicDomain(1.0, 2.0)
        s = repr("text/plain", pd)
        @test s == "PeriodicDomain{Float64}: x ∈ [-1.0, 1.0) × y ∈ [-2.0, 2.0)"
    end

    @testset "PVContour show" begin
        nodes = [SVector{2,Float64}(cos(2π*k/32), sin(2π*k/32)) for k in 0:31]
        c = PVContour(nodes, 1.5)
        s = repr("text/plain", c)
        @test occursin("32 nodes", s)
        @test occursin("Δq = 1.5", s)
        @test occursin("closed", s)
        @test occursin("centered at (", s)

        # Spanning contour
        span_nodes = [SVector{2,Float64}(Float64(k), 0.0) for k in 0:3]
        cs = PVContour(span_nodes, 2.0, SVector(4.0, 0.0))
        s_span = repr("text/plain", cs)
        @test occursin("spanning", s_span)
        @test !occursin("centered at", s_span)

        # Empty contour (0 nodes)
        c0 = PVContour(SVector{2,Float64}[], 1.0)
        s0 = repr("text/plain", c0)
        @test occursin("0 nodes", s0)
    end
end
