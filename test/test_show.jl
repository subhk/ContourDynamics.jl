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

    @testset "ContourProblem show" begin
        nodes = [SVector{2,Float64}(cos(2π*k/32), sin(2π*k/32)) for k in 0:31]
        c = PVContour(nodes, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        s = repr("text/plain", prob)
        @test occursin("ContourProblem", s)
        @test occursin("kernel: EulerKernel", s)
        @test occursin("domain: UnboundedDomain", s)
        @test occursin("contours: 1 PVContour{Float64}", s)
        @test occursin("└──", s)
        @test occursin("32 nodes", s)

        # Compact form
        sc = repr(prob)
        @test sc == "ContourProblem{EulerKernel, UnboundedDomain, Float64}"

        # Empty contours
        prob0 = ContourProblem(EulerKernel(), UnboundedDomain(), PVContour{Float64}[])
        s0 = repr("text/plain", prob0)
        @test occursin("contours: 0 PVContour{Float64}", s0)
        @test !occursin("└── ", split(s0, "contours")[end])  # no sub-tree for empty

        # Many contours (test truncation)
        cs = [PVContour([SVector{2,Float64}(cos(2π*k/8), sin(2π*k/8)) for k in 0:7], Float64(i))
              for i in 1:7]
        prob_many = ContourProblem(EulerKernel(), UnboundedDomain(), cs)
        s_many = repr("text/plain", prob_many)
        @test occursin("contours: 7 PVContour{Float64}", s_many)
        @test occursin("… and 3 more", s_many)
    end

    @testset "MultiLayerContourProblem show" begin
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)
        kernel = MultiLayerQGKernel(Ld, coupling)

        c1 = PVContour([SVector{2,Float64}(cos(2π*k/32), sin(2π*k/32)) for k in 0:31], 1.0)
        c2 = PVContour([SVector{2,Float64}(0.5cos(2π*k/16), 0.5sin(2π*k/16)) for k in 0:15], -1.0)
        prob = MultiLayerContourProblem(kernel, UnboundedDomain(), ([c1], [c2]))

        s = repr("text/plain", prob)
        @test occursin("MultiLayerContourProblem", s)
        @test occursin("kernel: MultiLayerQGKernel{2, Float64}", s)
        @test occursin("│   ├── Ld:", s)
        @test occursin("│   └── eigenvalues:", s)
        @test occursin("domain: UnboundedDomain", s)
        @test occursin("layers: 2 layers", s)
        @test occursin("Layer 1: 1 contour", s)
        @test occursin("Layer 2: 1 contour", s)
        @test occursin("32 nodes", s)
        @test occursin("16 nodes", s)

        # Compact form
        sc = repr(prob)
        @test occursin("MultiLayerContourProblem", sc)
    end
end
