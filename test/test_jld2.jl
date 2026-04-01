using Test, ContourDynamics, StaticArrays, JLD2

@testset "JLD2 Extension" begin
    @testset "Single-Layer Round-Trip" begin
        c = circular_patch(1.0, 32, 2.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

        fname = tempname() * ".jld2"
        try
            save_snapshot(fname, prob, 0; dt=0.01)
            data = load_snapshot(fname, 0)

            @test data.step == 0
            @test data.time ≈ 0.0
            @test length(data.contours) == 1
            @test nnodes(data.contours[1]) == 32
            @test data.contours[1].pv ≈ 2.0

            # Node positions preserved exactly
            for i in 1:32
                @test data.contours[1].nodes[i] ≈ prob.contours[1].nodes[i]
            end

            # Diagnostics preserved
            @test data.diagnostics.circulation ≈ circulation(prob)
            @test data.diagnostics.enstrophy ≈ enstrophy(prob)
        finally
            rm(fname; force=true)
        end
    end

    @testset "Overwrite Existing Step" begin
        c = circular_patch(1.0, 16, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

        fname = tempname() * ".jld2"
        try
            # Save step 0 twice — should not crash
            save_snapshot(fname, prob, 0)
            save_snapshot(fname, prob, 0)

            data = load_snapshot(fname, 0)
            @test length(data.contours) == 1
        finally
            rm(fname; force=true)
        end
    end

    @testset "Spanning Contour Wrap Preserved" begin
        domain = PeriodicDomain(3.0)
        staircase = beta_staircase(1.0, domain, 4; nodes_per_contour=8)
        vortex = circular_patch(0.5, 16, 1.0)
        prob = ContourProblem(QGKernel(1.0), domain, vcat(staircase, [vortex]))

        fname = tempname() * ".jld2"
        try
            save_snapshot(fname, prob, 5; dt=0.1)
            data = load_snapshot(fname, 5)

            @test length(data.contours) == length(prob.contours)

            # Spanning contour wraps preserved
            for (orig, loaded) in zip(prob.contours, data.contours)
                @test loaded.wrap ≈ orig.wrap
                @test is_spanning(loaded) == is_spanning(orig)
            end
        finally
            rm(fname; force=true)
        end
    end

    @testset "Multiple Steps and load_simulation" begin
        c = circular_patch(1.0, 16, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

        fname = tempname() * ".jld2"
        try
            save_snapshot(fname, prob, 0; dt=0.01)
            save_snapshot(fname, prob, 10; dt=0.01)
            save_snapshot(fname, prob, 20; dt=0.01)

            sim = load_simulation(fname)
            @test length(sim) == 3
            @test sim[1].step == 0
            @test sim[2].step == 10
            @test sim[3].step == 20
        finally
            rm(fname; force=true)
        end
    end
end
