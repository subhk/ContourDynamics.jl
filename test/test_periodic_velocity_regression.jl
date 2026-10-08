# Pins periodic single-layer velocity outputs to guard the dedup refactor.
# The baseline numbers live in data/periodic_velocity_baseline.txt, written by
# gen_periodic_velocity_baseline.jl on known-good code; regenerate them only
# from such a baseline, never by hand. The implementation-independent checks
# in test_periodic_qg_sqg.jl ("Periodic velocity vs image-sum oracle") catch
# errors that a regeneration would otherwise bake into this file.
# The QG block was regenerated after the periodic QG correction was Ewald
# split; the earlier block pinned the truncation error of an undamped series.
# It was regenerated again when the Ewald splitting parameter began to follow
# the truncation: this case then left the direct periodic-image sum, whose
# polynomial K₀ approximation is good to about 1e-7, for the Ewald split
# (1e-13 against an accurate-K₀ image sum).
# The SQG block was regenerated after the softening moved into a quasi-2-D
# Ewald split (the earlier block pinned a 1e-12 image-truncation residue) and
# curved panels gained singular subtraction (a 2.5e-9 change at ds/δ ≈ 0.8).

# Reads the `[case]` blocks of `ux uy` lines written by the generator.
function load_periodic_velocity_baseline(path)
    baseline = Dict{String,Vector{SVector{2,Float64}}}()
    current = SVector{2,Float64}[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, '#')) && continue
        if startswith(s, '[') && endswith(s, ']')
            current = baseline[String(s[2:end-1])] = SVector{2,Float64}[]
        else
            ux, uy = split(s)
            push!(current, SVector(parse(Float64, ux), parse(Float64, uy)))
        end
    end
    return baseline
end

@testset "Periodic velocity regression (refactor guard)" begin
    expected = load_periodic_velocity_baseline(
        joinpath(@__DIR__, "data", "periodic_velocity_baseline.txt"))
    dom = PeriodicDomain(10.0, 10.0)
    mkpatch() = circular_patch(0.5, 200, 1.0)

    expected_euler = expected["euler"]
    prob_euler = ContourProblem(EulerKernel(), dom, [mkpatch()])
    vel_euler = zeros(SVector{2,Float64}, total_nodes(prob_euler))
    velocity!(vel_euler, prob_euler)
    @test all(isapprox.(vel_euler, expected_euler; rtol=1e-12, atol=1e-14))

    expected_qg = expected["qg"]
    prob_qg = ContourProblem(QGKernel(1.0), dom, [mkpatch()])
    vel_qg = zeros(SVector{2,Float64}, total_nodes(prob_qg))
    velocity!(vel_qg, prob_qg)
    @test all(isapprox.(vel_qg, expected_qg; rtol=1e-12, atol=1e-14))

    expected_sqg = expected["sqg"]
    prob_sqg = ContourProblem(SQGKernel(0.02), dom, [mkpatch()])
    vel_sqg = zeros(SVector{2,Float64}, total_nodes(prob_sqg))
    velocity!(vel_sqg, prob_sqg)
    @test all(isapprox.(vel_sqg, expected_sqg; rtol=1e-12, atol=1e-14))
end
