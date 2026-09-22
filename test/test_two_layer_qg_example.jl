using Test, ContourDynamics, StaticArrays, LinearAlgebra

module TwoLayerExampleSetup
    include(joinpath(@__DIR__, "..", "examples", "two_layer_qg_setup.jl"))
end

@testset "Two-layer QG example literature setup" begin
    prob, Ld, coupling = TwoLayerExampleSetup.polvani_upper_layer_merger_problem(
        nodes_per_contour=32)
    layers = prob.contour_problem.layers
    H = kernel(prob).layer_thicknesses
    @test length(layers[1]) == 2
    @test isempty(layers[2])
    @test all(c -> c.pv == 1.0, layers[1])
    @test norm(centroid(layers[1][2]) - centroid(layers[1][1])) ≈ 2.2
    @test coupling ≈ 25.0 * [-1.0 1.0; 0.2 -0.2]
    @test coupling * ones(2) ≈ zeros(2) atol=1e-14
    @test H == SVector(0.2, 1.0)
    @test Diagonal(H) * coupling ≈ coupling' * Diagonal(H)
    @test Ld[1] ≈ 1 / sqrt(30)

    circulation0 = circulation(prob)
    nodes0 = copy(layers[1][1].nodes)
    @test isfinite(energy(prob))
    evolve!(prob; nsteps=2)
    @test isfinite(energy(prob))
    @test circulation(prob) ≈ circulation0 rtol=1e-8
    @test prob.contour_problem.layers[1][1].nodes != nodes0
    @test_throws ArgumentError TwoLayerExampleSetup.polvani_upper_layer_merger_problem(depth_ratio=0.0)
end
