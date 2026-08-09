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

    unwrapped = _unwrap_periodic_points(
        [SVector(0.0, 0.0), SVector(-0.8, 0.1), SVector(0.9, 0.2)],
        (-1.0, 1.0, -1.0, 1.0))
    @test unwrapped[1] == SVector(0.0, 0.0)
    @test unwrapped[2] == SVector(-0.8, 0.1)
    @test unwrapped[3] ≈ SVector(-1.1, 0.2)
end

@testset "Beta drift example scope is explicit" begin
    repo_root = normpath(joinpath(@__DIR__, ".."))
    example_text = read(joinpath(repo_root, "examples", "beta_drift.jl"), String)
    docs_text = read(joinpath(repo_root, "docs", "src", "examples", "beta_plane_vortex_drift.md"), String)

    @test occursin("kernel=:beta_plane_qg", example_text)
    @test occursin("beta_staircase", example_text)
    @test !occursin("BETA_DRIFT_PRESET", example_text)
    @test !occursin("demo", example_text)
    @test occursin("n_beta = 50", example_text)
    @test occursin("t_final = 28.0", example_text)
    @test occursin("save_dt = 7.0", example_text)
    @test occursin("(-L, L, -0.25L, 0.75L)", example_text)
    @test occursin("equations (3.2)-(3.3)", example_text)
    @test occursin("surgery = :none", example_text)
    @test occursin("BetaPlaneQGKernel", docs_text)
    @test occursin("analytic beta-plane correction", docs_text)
    @test occursin("q_r - beta*y", docs_text)
    @test !occursin("BetaPlaneQGModel", example_text)
    @test !occursin("regular-grid beta-plane QG", docs_text)
    @test !occursin("BETA_DRIFT_ALLOW_ACTIVE_STAIRCASE", example_text)
    @test !occursin("lam_dritschel_beta_staircase", example_text)
    @test !occursin("active PV-staircase setup", docs_text)
end
