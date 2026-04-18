using ContourDynamics
using StaticArrays
using Test

# Guard against double-include when run from runtests.jl
@isdefined(circular_patch) || include("test_utils.jl")

# Disable scalar indexing on GPU arrays to catch accidental cu_array[i] access.
# Only activates when CUDA is actually loaded.
try
    using CUDA
    CUDA.allowscalar(false)
catch
end

@testset "Device abstraction" begin
    @testset "ContourProblem defaults to CPU" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        @test prob.dev === CPU()
    end

    @testset "ContourProblem accepts dev keyword" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=CPU())
        @test prob.dev === CPU()
    end

    @testset "GPU() without CUDA gives helpful error" begin
        @test_throws ErrorException device_array(GPU())
    end

    @testset "GPU velocity! without CUDA gives helpful error" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=GPU())
        vel = zeros(SVector{2,Float64}, total_nodes(prob))
        @test_throws ErrorException velocity!(vel, prob)
    end

    @testset "GPU() accepts supported single-layer periodic and unbounded cases" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(QGKernel(1.0), UnboundedDomain(), [c]; dev=GPU())
        @test prob.dev === GPU()

        prob_periodic = ContourProblem(QGKernel(1.0), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
        @test prob_periodic.dev === GPU()

        prob_periodic_euler = ContourProblem(EulerKernel(), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
        @test prob_periodic_euler.dev === GPU()

        prob = ContourProblem(SQGKernel(0.01), UnboundedDomain(), [c]; dev=GPU())
        @test prob.dev === GPU()
        prob_periodic_sqg = ContourProblem(SQGKernel(0.01), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
        @test prob_periodic_sqg.dev === GPU()
    end

    @testset "GPU() accepts multi-layer QG problems" begin
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)
        c = circular_patch(0.5, 32, 1.0)
        prob = MultiLayerContourProblem(MultiLayerQGKernel(Ld, coupling), UnboundedDomain(),
                                        ([c], PVContour{Float64}[]); dev=GPU())
        @test prob.dev === GPU()
    end

    @testset "CPU device_array returns Array" begin
        @test device_array(CPU()) === Array
    end

    @testset "to_cpu is identity for Array" begin
        x = [1.0, 2.0, 3.0]
        @test to_cpu(x) === x
    end

    @testset "device_zeros CPU" begin
        z = device_zeros(CPU(), Float64, 5)
        @test z == zeros(5)
        @test z isa Vector{Float64}
    end

    @testset "Full evolve! with dev=CPU() matches existing behavior" begin
        c = circular_patch(0.5, 64, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=CPU())
        stepper = RK4Stepper(0.01, total_nodes(prob))
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)
        circ_before = circulation(prob)
        evolve!(prob, stepper, params; nsteps=10)
        circ_after = circulation(prob)
        @test isapprox(circ_before, circ_after; rtol=1e-6)
    end

    @testset "pack_segments round-trip" begin
        c1 = circular_patch(0.5, 16, 1.0)
        c2 = circular_patch(0.3, 8, -0.5)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
        seg = ContourDynamics.pack_segments(prob, CPU())
        @test length(seg.ax) == total_nodes(prob)
        @test length(seg.pv) == total_nodes(prob)
        # First segment of c1
        @test seg.ax[1] ≈ c1.nodes[1][1]
        @test seg.ay[1] ≈ c1.nodes[1][2]
        @test seg.bx[1] ≈ c1.nodes[2][1]
        @test seg.by[1] ≈ c1.nodes[2][2]
        @test seg.pv[1] ≈ c1.pv
    end

    @testset "KA Euler velocity matches direct CPU" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        N = total_nodes(prob)

        # CPU reference
        vel_cpu = zeros(SVector{2,Float64}, N)
        ContourDynamics._direct_velocity!(vel_cpu, prob)

        # KA CPU kernel path
        vel_ka_x = zeros(Float64, N)
        vel_ka_y = zeros(Float64, N)
        seg = ContourDynamics.pack_segments(prob, CPU())
        target_x = Float64[c.nodes[i][1] for c in prob.contours for i in 1:nnodes(c)]
        target_y = Float64[c.nodes[i][2] for c in prob.contours for i in 1:nnodes(c)]
        ContourDynamics._ka_euler_velocity!(vel_ka_x, vel_ka_y, target_x, target_y, seg, CPU())

        for i in 1:N
            @test isapprox(vel_ka_x[i], vel_cpu[i][1]; atol=1e-12)
            @test isapprox(vel_ka_y[i], vel_cpu[i][2]; atol=1e-12)
        end
    end

    @testset "CPU velocity! uses KA path without changing results" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=CPU())
        N = total_nodes(prob)

        vel_ref = zeros(SVector{2,Float64}, N)
        vel = similar(vel_ref)
        ContourDynamics._direct_velocity!(vel_ref, prob)
        velocity!(vel, prob)

        for i in 1:N
            @test isapprox(vel[i][1], vel_ref[i][1]; atol=1e-12)
            @test isapprox(vel[i][2], vel_ref[i][2]; atol=1e-12)
        end
    end

    @testset "Large GPU-tagged single-layer problems fall back to treecode policy" begin
        c = circular_patch(0.5, 1024, 1.0)
        prob_cpu = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=CPU())
        prob_gpu = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=GPU())
        N = total_nodes(prob_cpu)

        vel_cpu = zeros(SVector{2,Float64}, N)
        vel_gpu = similar(vel_cpu)
        velocity!(vel_cpu, prob_cpu)
        velocity!(vel_gpu, prob_gpu)

        for i in 1:N
            @test vel_gpu[i] == vel_cpu[i]
        end
    end

    @testset "Treecode direct leaf KA helper matches CPU loop" begin
        c1 = circular_patch(0.3, 64, 1.0)
        c2 = circular_patch(0.3, 64, -0.8)
        contours = [c1, c2]
        tree = ContourDynamics.build_fmm_tree(contours)
        plan = ContourDynamics._build_tree_eval_plan(tree, contours)
        li_idx = findfirst(!isempty, plan.direct_lists)
        @test li_idx !== nothing

        target_leaf_idx = tree.leaf_indices[li_idx]
        source_box_idx = first(plan.direct_lists[li_idx])
        N = total_nodes(ContourProblem(EulerKernel(), UnboundedDomain(), contours))
        vel_ref = zeros(SVector{2,Float64}, N)
        vel_ka = zeros(SVector{2,Float64}, N)
        target_box = tree.boxes[target_leaf_idx]
        flat_idxs = [plan.flat_indices[seg_idx] for seg_idx in target_box.segment_range]

        ContourDynamics._treecode_direct_to_leaf!(vel_ref, tree, target_leaf_idx, source_box_idx,
                                                  contours, plan, EulerKernel(), UnboundedDomain(), nothing)
        ContourDynamics._ka_treecode_direct_to_leaf!(vel_ka, tree, target_leaf_idx, source_box_idx,
                                                     contours, plan, EulerKernel(), UnboundedDomain(), CPU())

        for i in flat_idxs
            @test isapprox(vel_ka[i][1], vel_ref[i][1]; atol=1e-12, rtol=1e-12)
            @test isapprox(vel_ka[i][2], vel_ref[i][2]; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "Treecode direct leaf KA helper matches periodic QG CPU loop" begin
        clear_ewald_cache!()
        c1 = circular_patch(0.25, 48, 1.0)
        c2 = circular_patch(0.18, 48, -0.8)
        contours = [c1, c2]
        tree = ContourDynamics.build_fmm_tree(contours)
        plan = ContourDynamics._build_tree_eval_plan(tree, contours)
        li_idx = findfirst(!isempty, plan.direct_lists)
        @test li_idx !== nothing

        target_leaf_idx = tree.leaf_indices[li_idx]
        source_box_idx = first(plan.direct_lists[li_idx])
        prob = ContourProblem(QGKernel(1.1), PeriodicDomain(2.0, 2.0), contours)
        N = total_nodes(prob)
        ewald = ContourDynamics._prefetch_ewald(prob.domain, EulerKernel())
        vel_ref = zeros(SVector{2,Float64}, N)
        vel_ka = zeros(SVector{2,Float64}, N)
        target_box = tree.boxes[target_leaf_idx]
        flat_idxs = [plan.flat_indices[seg_idx] for seg_idx in target_box.segment_range]

        ContourDynamics._treecode_direct_to_leaf!(vel_ref, tree, target_leaf_idx, source_box_idx,
                                                  contours, plan, prob.kernel, prob.domain, ewald)
        ContourDynamics._ka_treecode_direct_to_leaf!(vel_ka, tree, target_leaf_idx, source_box_idx,
                                                     contours, plan, prob.kernel, prob.domain, CPU())

        for i in flat_idxs
            @test isapprox(vel_ka[i][1], vel_ref[i][1]; atol=1e-8, rtol=1e-8)
            @test isapprox(vel_ka[i][2], vel_ref[i][2]; atol=1e-8, rtol=1e-8)
        end
    end

    @testset "Treecode linearized leaf KA helper matches CPU loop" begin
        shift(nodes, dx, dy) = [SVector(p[1] + dx, p[2] + dy) for p in nodes]
        base = circular_patch(0.12, 48, 1.0)
        contours = [
            PVContour(shift(base.nodes, -0.6, -0.6), 1.0),
            PVContour(shift(base.nodes,  0.6, -0.6), -0.8),
            PVContour(shift(base.nodes, -0.6,  0.6), 0.7),
            PVContour(shift(base.nodes,  0.6,  0.6), -0.6),
        ]
        tree = ContourDynamics.build_fmm_tree(contours; max_per_leaf=8)
        plan = ContourDynamics._build_tree_eval_plan(tree, contours)
        li_idx = findfirst(!isempty, plan.approx_lists)
        @test li_idx !== nothing

        target_leaf_idx = tree.leaf_indices[li_idx]
        source_box_idx = first(plan.approx_lists[li_idx])
        N = total_nodes(ContourProblem(EulerKernel(), UnboundedDomain(), contours))
        vel_ref = zeros(SVector{2,Float64}, N)
        vel_ka = zeros(SVector{2,Float64}, N)
        target_box = tree.boxes[target_leaf_idx]
        flat_idxs = [plan.flat_indices[seg_idx] for seg_idx in target_box.segment_range]

        ContourDynamics._treecode_linearized_to_leaf!(vel_ref, tree, target_leaf_idx, source_box_idx,
                                                      contours, plan, EulerKernel(), UnboundedDomain(), nothing)
        ContourDynamics._treecode_linearized_to_leaf!(vel_ka, tree, target_leaf_idx, source_box_idx,
                                                      contours, plan, EulerKernel(), UnboundedDomain(), nothing, CPU())

        for i in flat_idxs
            @test isapprox(vel_ka[i][1], vel_ref[i][1]; atol=1e-12, rtol=1e-12)
            @test isapprox(vel_ka[i][2], vel_ref[i][2]; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "Treecode linearized leaf KA helper matches periodic QG CPU loop" begin
        clear_ewald_cache!()
        shift(nodes, dx, dy) = [SVector(p[1] + dx, p[2] + dy) for p in nodes]
        base = circular_patch(0.1, 48, 1.0)
        contours = [
            PVContour(shift(base.nodes, -0.7, -0.7), 1.0),
            PVContour(shift(base.nodes,  0.7, -0.7), -0.8),
            PVContour(shift(base.nodes, -0.7,  0.7), 0.7),
            PVContour(shift(base.nodes,  0.7,  0.7), -0.6),
        ]
        tree = ContourDynamics.build_fmm_tree(contours; max_per_leaf=8)
        plan = ContourDynamics._build_tree_eval_plan(tree, contours)
        li_idx = findfirst(!isempty, plan.approx_lists)
        @test li_idx !== nothing

        target_leaf_idx = tree.leaf_indices[li_idx]
        source_box_idx = first(plan.approx_lists[li_idx])
        prob = ContourProblem(QGKernel(1.1), PeriodicDomain(2.0, 2.0), contours)
        N = total_nodes(prob)
        ewald = ContourDynamics._prefetch_ewald(prob.domain, EulerKernel())
        vel_ref = zeros(SVector{2,Float64}, N)
        vel_ka = zeros(SVector{2,Float64}, N)
        target_box = tree.boxes[target_leaf_idx]
        flat_idxs = [plan.flat_indices[seg_idx] for seg_idx in target_box.segment_range]

        ContourDynamics._treecode_linearized_to_leaf!(vel_ref, tree, target_leaf_idx, source_box_idx,
                                                      contours, plan, prob.kernel, prob.domain, ewald)
        ContourDynamics._treecode_linearized_to_leaf!(vel_ka, tree, target_leaf_idx, source_box_idx,
                                                      contours, plan, prob.kernel, prob.domain, ewald, CPU())

        for i in flat_idxs
            @test isapprox(vel_ka[i][1], vel_ref[i][1]; atol=1e-8, rtol=1e-8)
            @test isapprox(vel_ka[i][2], vel_ref[i][2]; atol=1e-8, rtol=1e-8)
        end
    end

    @testset "Treecode batched direct leaf KA helper matches per-box accumulation" begin
        shift(nodes, dx, dy) = [SVector(p[1] + dx, p[2] + dy) for p in nodes]
        base = circular_patch(0.12, 48, 1.0)
        contours = [
            PVContour(shift(base.nodes, -0.6, -0.6), 1.0),
            PVContour(shift(base.nodes,  0.6, -0.6), -0.8),
            PVContour(shift(base.nodes, -0.6,  0.6), 0.7),
            PVContour(shift(base.nodes,  0.6,  0.6), -0.6),
        ]
        tree = ContourDynamics.build_fmm_tree(contours; max_per_leaf=8)
        plan = ContourDynamics._build_tree_eval_plan(tree, contours)
        li_idx = findfirst(li -> length(plan.direct_lists[li]) > 1, eachindex(plan.direct_lists))
        @test li_idx !== nothing

        target_leaf_idx = tree.leaf_indices[li_idx]
        N = total_nodes(ContourProblem(EulerKernel(), UnboundedDomain(), contours))
        vel_ref = zeros(SVector{2,Float64}, N)
        vel_ka = zeros(SVector{2,Float64}, N)
        target_box = tree.boxes[target_leaf_idx]
        flat_idxs = [plan.flat_indices[seg_idx] for seg_idx in target_box.segment_range]

        for source_box_idx in plan.direct_lists[li_idx]
            ContourDynamics._treecode_direct_to_leaf!(vel_ref, tree, target_leaf_idx, source_box_idx,
                                                      contours, plan, EulerKernel(), UnboundedDomain(), nothing)
        end
        ContourDynamics._ka_treecode_direct_lists_to_leaf!(vel_ka, tree, target_leaf_idx, plan.direct_lists[li_idx],
                                                           contours, plan, EulerKernel(), UnboundedDomain(), CPU())

        for i in flat_idxs
            @test isapprox(vel_ka[i][1], vel_ref[i][1]; atol=1e-12, rtol=1e-12)
            @test isapprox(vel_ka[i][2], vel_ref[i][2]; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "Treecode batched linearized leaf KA helper matches per-box accumulation" begin
        clear_ewald_cache!()
        shift(nodes, dx, dy) = [SVector(p[1] + dx, p[2] + dy) for p in nodes]
        base = circular_patch(0.1, 48, 1.0)
        contours = [
            PVContour(shift(base.nodes, -0.7, -0.7), 1.0),
            PVContour(shift(base.nodes,  0.7, -0.7), -0.8),
            PVContour(shift(base.nodes, -0.7,  0.7), 0.7),
            PVContour(shift(base.nodes,  0.7,  0.7), -0.6),
        ]
        tree = ContourDynamics.build_fmm_tree(contours; max_per_leaf=8)
        plan = ContourDynamics._build_tree_eval_plan(tree, contours)
        li_idx = findfirst(li -> length(plan.approx_lists[li]) > 1, eachindex(plan.approx_lists))
        @test li_idx !== nothing

        target_leaf_idx = tree.leaf_indices[li_idx]
        prob = ContourProblem(QGKernel(1.1), PeriodicDomain(2.0, 2.0), contours)
        N = total_nodes(prob)
        ewald = ContourDynamics._prefetch_ewald(prob.domain, EulerKernel())
        vel_ref = zeros(SVector{2,Float64}, N)
        vel_ka = zeros(SVector{2,Float64}, N)
        target_box = tree.boxes[target_leaf_idx]
        flat_idxs = [plan.flat_indices[seg_idx] for seg_idx in target_box.segment_range]

        for source_box_idx in plan.approx_lists[li_idx]
            ContourDynamics._treecode_linearized_to_leaf!(vel_ref, tree, target_leaf_idx, source_box_idx,
                                                          contours, plan, prob.kernel, prob.domain, ewald)
        end
        ContourDynamics._ka_treecode_linearized_lists_to_leaf!(vel_ka, tree, target_leaf_idx, plan.approx_lists[li_idx],
                                                               contours, plan, prob.kernel, prob.domain, CPU())

        for i in flat_idxs
            @test isapprox(vel_ka[i][1], vel_ref[i][1]; atol=1e-8, rtol=1e-8)
            @test isapprox(vel_ka[i][2], vel_ref[i][2]; atol=1e-8, rtol=1e-8)
        end
    end

    @testset "KA SQG velocity matches direct CPU" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(SQGKernel(0.02), UnboundedDomain(), [c])
        N = total_nodes(prob)

        vel_ref = zeros(SVector{2,Float64}, N)
        vel_ka = similar(vel_ref)
        ContourDynamics._direct_velocity!(vel_ref, prob)
        ContourDynamics._ka_velocity!(vel_ka, prob, CPU())

        for i in 1:N
            @test isapprox(vel_ka[i][1], vel_ref[i][1]; atol=1e-12)
            @test isapprox(vel_ka[i][2], vel_ref[i][2]; atol=1e-12)
        end
    end

    @testset "KA QG velocity matches direct CPU" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(QGKernel(1.25), UnboundedDomain(), [c])
        N = total_nodes(prob)

        vel_ref = zeros(SVector{2,Float64}, N)
        vel_ka = similar(vel_ref)
        ContourDynamics._direct_velocity!(vel_ref, prob)
        ContourDynamics._ka_velocity!(vel_ka, prob, CPU())

        for i in 1:N
            @test isapprox(vel_ka[i][1], vel_ref[i][1]; atol=1e-8, rtol=1e-8)
            @test isapprox(vel_ka[i][2], vel_ref[i][2]; atol=1e-8, rtol=1e-8)
        end
    end

    @testset "KA periodic Euler velocity matches direct CPU" begin
        clear_ewald_cache!()
        c = circular_patch(0.35, 24, 1.0)
        prob = ContourProblem(EulerKernel(), PeriodicDomain(2.0, 2.0), [c])
        N = total_nodes(prob)

        vel_ref = zeros(SVector{2,Float64}, N)
        vel_ka = similar(vel_ref)
        ContourDynamics._direct_velocity!(vel_ref, prob)
        ContourDynamics._ka_velocity!(vel_ka, prob, CPU())

        for i in 1:N
            @test isapprox(vel_ka[i][1], vel_ref[i][1]; atol=1e-12, rtol=1e-12)
            @test isapprox(vel_ka[i][2], vel_ref[i][2]; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "KA periodic QG velocity matches direct CPU" begin
        clear_ewald_cache!()
        c = circular_patch(0.35, 24, 1.0)
        prob = ContourProblem(QGKernel(1.1), PeriodicDomain(2.0, 2.0), [c])
        N = total_nodes(prob)

        vel_ref = zeros(SVector{2,Float64}, N)
        vel_ka = similar(vel_ref)
        ContourDynamics._direct_velocity!(vel_ref, prob)
        ContourDynamics._ka_velocity!(vel_ka, prob, CPU())

        for i in 1:N
            @test isapprox(vel_ka[i][1], vel_ref[i][1]; atol=1e-12, rtol=1e-12)
            @test isapprox(vel_ka[i][2], vel_ref[i][2]; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "KA periodic SQG velocity matches direct CPU" begin
        clear_ewald_cache!()
        c = circular_patch(0.35, 24, 1.0)
        prob = ContourProblem(SQGKernel(0.02), PeriodicDomain(2.0, 2.0), [c])
        N = total_nodes(prob)

        vel_ref = zeros(SVector{2,Float64}, N)
        vel_ka = similar(vel_ref)
        ContourDynamics._direct_velocity!(vel_ref, prob)
        ContourDynamics._ka_velocity!(vel_ka, prob, CPU())

        for i in 1:N
            @test isapprox(vel_ka[i][1], vel_ref[i][1]; atol=1e-12, rtol=1e-12)
            @test isapprox(vel_ka[i][2], vel_ref[i][2]; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "KA multi-layer velocity matches direct CPU" begin
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)
        c1 = circular_patch(0.35, 24, 1.0)
        c2 = circular_patch(0.2, 16, -0.5)
        prob = MultiLayerContourProblem(MultiLayerQGKernel(Ld, coupling), UnboundedDomain(),
                                        ([c1], [c2]))

        vel_ref = (zeros(SVector{2,Float64}, nnodes(c1)),
                   zeros(SVector{2,Float64}, nnodes(c2)))
        vel_ka = (similar(vel_ref[1]), similar(vel_ref[2]))

        ContourDynamics._direct_velocity!(vel_ref, prob)
        ContourDynamics._ka_multilayer_velocity!(vel_ka, prob, CPU())

        for i in eachindex(vel_ref[1])
            @test isapprox(vel_ka[1][i][1], vel_ref[1][i][1]; atol=1e-8, rtol=1e-8)
            @test isapprox(vel_ka[1][i][2], vel_ref[1][i][2]; atol=1e-8, rtol=1e-8)
        end
        for i in eachindex(vel_ref[2])
            @test isapprox(vel_ka[2][i][1], vel_ref[2][i][1]; atol=1e-8, rtol=1e-8)
            @test isapprox(vel_ka[2][i][2], vel_ref[2][i][2]; atol=1e-8, rtol=1e-8)
        end
    end

    @testset "Large GPU-tagged multi-layer problems fall back to treecode policy" begin
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)
        c1 = circular_patch(0.35, 600, 1.0)
        c2 = circular_patch(0.2, 600, -0.5)
        prob_cpu = MultiLayerContourProblem(MultiLayerQGKernel(Ld, coupling), UnboundedDomain(),
                                            ([c1], [c2]); dev=CPU())
        prob_gpu = MultiLayerContourProblem(MultiLayerQGKernel(Ld, coupling), UnboundedDomain(),
                                            ([c1], [c2]); dev=GPU())

        vel_cpu = (zeros(SVector{2,Float64}, nnodes(c1)),
                   zeros(SVector{2,Float64}, nnodes(c2)))
        vel_gpu = (similar(vel_cpu[1]), similar(vel_cpu[2]))

        velocity!(vel_cpu, prob_cpu)
        velocity!(vel_gpu, prob_gpu)

        for i in eachindex(vel_cpu[1])
            @test vel_gpu[1][i] == vel_cpu[1][i]
        end
        for i in eachindex(vel_cpu[2])
            @test vel_gpu[2][i] == vel_cpu[2][i]
        end
    end
end
