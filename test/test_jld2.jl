using Test, ContourDynamics, StaticArrays, JLD2

function _jld2_contour_equal(a::PVContour, b::PVContour)
    return a.nodes == b.nodes && a.pv == b.pv && a.wrap == b.wrap &&
           a.corners == b.corners
end

@testset "JLD2 Extension" begin
    @testset "Snapshot helpers use materialized output boundary" begin
        ext = Base.get_extension(ContourDynamics, :ContourDynamicsJLD2Ext)

        c = circular_patch(1.0, 16, 2.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        @test ext._snapshot_contours(prob) === materialize_contours(prob)

        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)
        kernel = MultiLayerQGKernel(Ld, coupling)
        layers = ([circular_patch(0.5, 12, 1.0)], [circular_patch(0.3, 8, -0.5)])
        mlprob = MultiLayerContourProblem(kernel, UnboundedDomain(), layers)
        @test ext._snapshot_layers(mlprob) === materialize_contours(mlprob)
    end

    @testset "Single-Layer Round-Trip" begin
        c = circular_patch(1.0, 32, 2.0)
        corners = falses(nnodes(c))
        corners[3] = true
        c = PVContour(c.nodes, c.pv, c.wrap, corners)
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
            @test corner_indices(data.contours[1]) == [3]

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

    @testset "Beta-Plane Problem Restart" begin
        for T in (Float32, Float64)
            beta = T(0.4)
            domain = PeriodicDomain(T(2.5), T(1.5))
            reference = beta_staircase(beta, domain, 4; nodes_per_contour=8)
            kernel = BetaPlaneQGKernel(beta, T(0.9), reference)
            live = deepcopy(reference)
            # Make the live staircase distinguishable from the frozen kernel
            # reference. A restart must not infer the latter from this state.
            live[1].nodes[2] += SVector{2,T}(zero(T), T(0.025))
            vortex = circular_patch(T(0.2), 12, T(0.7); cx=T(0.3), cy=T(0.4), T=T)
            prob = ContourProblem(kernel, domain, vcat(live, [vortex]))

            fname = tempname() * ".jld2"
            try
                save_snapshot(fname, prob, 7; dt=T(0.01), diagnostics=false)
                restarted = load_problem(fname, 7)

                @test restarted isa ContourProblem
                @test restarted.kernel isa BetaPlaneQGKernel{T}
                @test restarted.domain isa PeriodicDomain{T}
                @test restarted.kernel.beta === beta
                @test restarted.kernel.Ld === T(0.9)
                @test length(restarted.kernel.reference_contours) == length(reference)
                @test all(_jld2_contour_equal.(restarted.kernel.reference_contours,
                                                prob.kernel.reference_contours))
                @test all(_jld2_contour_equal.(restarted.contours, prob.contours))
                @test restarted.kernel.reference_contours[1].nodes !=
                      restarted.contours[1].nodes

                point = SVector{2,T}(T(0.13), T(-0.17))
                tol = T === Float32 ? 2e-5 : 2e-12
                @test velocity(restarted, point) ≈ velocity(prob, point) rtol=tol atol=tol

                original_stepper = RK4Stepper(T(0.0005), total_nodes(prob))
                restart_stepper = RK4Stepper(T(0.0005), total_nodes(restarted))
                timestep!(prob, original_stepper)
                timestep!(restarted, restart_stepper)
                @test all(zip(restarted.contours, prob.contours)) do (actual, expected)
                    all(isapprox.(actual.nodes, expected.nodes; rtol=tol, atol=tol))
                end

                jldopen(fname, "r") do f
                    mg = f["step_000007/metadata"]
                    @test mg["kernel_reference_contours"] == length(reference)
                    rg = mg["kernel_reference_geometry"]
                    @test rg["ncontours"] == length(reference)
                end
            finally
                rm(fname; force=true)
            end
        end
    end

    @testset "Legacy Beta-Plane Snapshot Remains State-Readable" begin
        domain = PeriodicDomain(2.0, 1.5)
        reference = beta_staircase(0.4, domain, 4; nodes_per_contour=8)
        prob = ContourProblem(BetaPlaneQGKernel(0.4, 1.0, reference),
                              domain, deepcopy(reference))
        fname = tempname() * ".jld2"
        try
            save_snapshot(fname, prob, 0; diagnostics=false)
            # Simulate the schema written before frozen reference geometry was
            # persisted. load_snapshot remains backward-compatible, while a
            # runnable problem cannot be recovered without that missing state.
            jldopen(fname, "r+") do f
                delete!(f["step_000000/metadata"], "kernel_reference_geometry")
            end

            data = load_snapshot(fname, 0)
            @test all(_jld2_contour_equal.(data.contours, prob.contours))
            err = try
                load_problem(fname, 0)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("older ContourDynamics version", sprint(showerror, err))
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

    @testset "Multi-Layer Round-Trip" begin
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)
        kernel = MultiLayerQGKernel(Ld, coupling)
        layers = ([circular_patch(0.5, 24, 1.0)], [circular_patch(0.3, 12, -0.5)])
        prob = MultiLayerContourProblem(kernel, UnboundedDomain(), layers)

        fname = tempname() * ".jld2"
        try
            save_snapshot(fname, prob, 3; dt=0.02)
            data = load_snapshot(fname, 3)

            @test data.step == 3
            @test data.time ≈ 0.06
            @test length(data.layers) == 2
            @test data.layers isa Tuple{Vector{PVContour{Float64}}, Vector{PVContour{Float64}}}
            @test nnodes(data.layers[1][1]) == nnodes(prob.layers[1][1])
            @test nnodes(data.layers[2][1]) == nnodes(prob.layers[2][1])
            @test data.layers[1][1].pv ≈ prob.layers[1][1].pv
            @test data.layers[2][1].pv ≈ prob.layers[2][1].pv
        finally
            rm(fname; force=true)
        end
    end
end
