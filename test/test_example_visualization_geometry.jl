using Test

include("test_utils.jl")
include("../examples/visualization_geometry.jl")

@testset "Example visualization geometry" begin
    c = PVContour([SVector(-1.0, 0.0), SVector(0.0, 0.0)], 1.0, SVector(2.0, 0.0))
    xs, ys = _contour_curve(c)

    @test xs == [-1.0, 0.0, 1.0]
    @test ys == [0.0, 0.0, 0.0]

    wrapped_xs, wrapped_ys = _periodic_curve(xs, ys, (-1.0, 1.0, -1.0, 1.0))
    @test wrapped_xs == xs
    @test wrapped_ys == ys

    split_xs, split_ys = _periodic_curve([0.8, 1.2], [0.0, 0.0], (-1.0, 1.0, -1.0, 1.0))
    @test split_xs[1] == 0.8
    @test isnan(split_xs[2])
    @test split_xs[3] ≈ -0.8
    @test split_ys[1] == 0.0
    @test isnan(split_ys[2])
    @test split_ys[3] == 0.0
end
