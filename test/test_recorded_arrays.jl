using Test, ContourDynamics, RecordedArrays

@testset "RecordedArrays Extension" begin
    contour = circular_patch(0.25, 16, 1.0)
    prob = ContourProblem(EulerKernel(), UnboundedDomain(), [contour])

    @test_throws ArgumentError recorded_diagnostics(prob; dt=0.0, nsteps=2)
    @test_throws ArgumentError recorded_diagnostics(prob; dt=Inf, nsteps=2)
    @test_throws ArgumentError recorded_diagnostics(prob; dt=0.01, nsteps=-1)
    @test_throws ArgumentError recorded_diagnostics(
        prob; dt=0.01, nsteps=2, record_every=0)

    contour32 = circular_patch(0.25, 16, 1.0; T=Float32)
    prob32 = ContourProblem(EulerKernel(), UnboundedDomain(), [contour32])
    @test eltype(prob32.contours[1].nodes[1]) === Float32
    @test_throws ArgumentError recorded_diagnostics(prob32; dt=1.0e100, nsteps=2)

    rec = recorded_diagnostics(prob; dt=0.01, nsteps=2, record_every=1)
    @test hasproperty(rec, :energy)
    @test hasproperty(rec, :clock)
    @test hasproperty(rec, :callback)
    rec.callback(prob, 0)
    rec.callback(prob, 1)
    entries = record(rec.energy)
    @test length(entries) == 2
    @test getts(entries[1]) == [0.0, 0.01]
    @test getts(entries[2]) == [0.01, 0.01]
    @test only(unique(getvs(entries[1]))) == energy(prob)
    @test only(unique(getvs(entries[2]))) == energy(prob)
end
