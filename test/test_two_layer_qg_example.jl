using Test

@testset "Two-layer QG example literature setup" begin
    example = read(joinpath(@__DIR__, "..", "examples", "two_layer_qg.jl"), String)

    @test occursin("polvani_upper_layer_merger_problem", example)
    @test occursin("initial_distance", example)
    @test occursin("depth_ratio = 0.2", example)
    @test occursin("gamma = 5.0", example)
    @test !occursin("layers=([c_upper], PVContour{Float64}[])", example)
end
