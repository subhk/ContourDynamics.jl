using Test, ContourDynamics, StaticArrays

extended = get(ENV, "CONTOURDYNAMICS_EXTENDED_TESTS", "false") == "true"

@testset "FMM" begin
    @testset "Quadtree Construction" begin
        c = circular_patch(1.0, 200, 1.0)
        tree = ContourDynamics.build_fmm_tree([c])

        @test length(tree.boxes) > 1
        @test length(tree.sorted_segments) == 200
        @test length(tree.leaf_indices) > 0
        @test tree.max_level > 0
        @test tree.max_level <= ContourDynamics._FMM_MAX_DEPTH

        for li in tree.leaf_indices
            box = tree.boxes[li]
            @test length(box.segment_range) <= ContourDynamics._FMM_MAX_PER_LEAF
        end

        total = sum(length(tree.boxes[li].segment_range) for li in tree.leaf_indices)
        @test total == 200

        has_interactions = any(!isempty(tree.interaction_lists[i]) for i in 1:length(tree.boxes))
        @test has_interactions

        for li in tree.leaf_indices
            @test li in tree.near_lists[li]
        end
    end

    @testset "Quadtree Bounding Box" begin
        c1 = circular_patch(0.5, 50, 1.0)
        c2_nodes = [SVector(5.0 + 0.5*cos(2*pi*i/50), 5.0 + 0.5*sin(2*pi*i/50)) for i in 0:49]
        c2 = PVContour(c2_nodes, 1.0)
        tree = ContourDynamics.build_fmm_tree([c1, c2])

        root = tree.boxes[1]
        @test root.center[1] > 0.0 && root.center[1] < 5.0
        @test root.center[2] > 0.0 && root.center[2] < 5.0
        @test root.half_width >= 2.5
    end

    @testset "Adaptive Proxy Guard" begin
        dense_nodes = [SVector(-1.0 + 0.15*cos(2*pi*i/127), -1.0 + 0.15*sin(2*pi*i/127)) for i in 0:127]
        sparse_nodes = [SVector(1.0 + 0.2*cos(2*pi*i/11), -1.0 + 0.2*sin(2*pi*i/11)) for i in 0:11]
        dense = PVContour(dense_nodes, 1.0)
        sparse = PVContour(sparse_nodes, -0.5)
        tree = ContourDynamics.build_fmm_tree([dense, sparse]; max_per_leaf=16)

        @test ContourDynamics._has_unhandled_coarse_leaf_interactions(tree)
    end

    @testset "Proxy Surfaces" begin
        @testset "Point Generation" begin
            center = SVector(0.0, 0.0)
            hw = 1.0
            pts = ContourDynamics._proxy_points(center, hw, 36)
            @test length(pts) == 36
            for pt in pts
                r = sqrt(pt[1]^2 + pt[2]^2)
                @test r ≈ 1.5 rtol=1e-14
            end
        end

        @testset "Kernel Matrix" begin
            targets = [SVector(3.0, 0.0), SVector(0.0, 3.0)]
            sources = [SVector(0.0, 0.0), SVector(1.0, 0.0)]
            K = ContourDynamics._build_kernel_matrix(EulerKernel(), UnboundedDomain(),
                                                      targets, sources)
            @test size(K) == (2, 2)
            @test K[1,1] ≈ -log(9.0) / (4*pi) rtol=1e-12
        end

        @testset "Level Operators" begin
            c = circular_patch(1.0, 128, 1.0)
            tree = ContourDynamics.build_fmm_tree([c])
            ops = ContourDynamics.precompute_level_operators(tree, EulerKernel())
            @test length(ops) == tree.max_level + 1
            p = ContourDynamics._FMM_PROXY_ORDER
            p_check = ContourDynamics._FMM_CHECK_ORDER
            for op in ops
                @test size(op.check_to_proxy_pinv) == (p, p_check)
            end
        end

        @testset "S2M Proxy Strengths" begin
            c = circular_patch(0.5, 64, 1.0)
            contours = [c]
            tree = ContourDynamics.build_fmm_tree(contours; max_per_leaf=100)
            ops = ContourDynamics.precompute_level_operators(tree, EulerKernel())

            p = ContourDynamics._FMM_PROXY_ORDER
            p_check = ContourDynamics._FMM_CHECK_ORDER
            plan = ContourDynamics._build_tree_eval_plan(tree, contours;
                                                         p, p_check,
                                                         include_proxy_geometry=true,
                                                         kernel=EulerKernel(),
                                                         domain=UnboundedDomain())
            proxy_data = [ContourDynamics.ProxyData(
                zeros(SVector{2,Float64}, p),
                SVector{2,Float64}[]) for _ in 1:length(tree.boxes)]

            ContourDynamics._s2m!(proxy_data, tree, contours, plan, EulerKernel(),
                                  UnboundedDomain(), ops, nothing; p, p_check)

            leaf = tree.leaf_indices[1]
            strengths = proxy_data[leaf].equiv_strengths
            @test length(strengths) == p
            @test all(s -> isfinite(s[1]) && isfinite(s[2]), strengths)
            @test any(!iszero, strengths)
        end
    end

    @testset "Translation Operators" begin
        @testset "M2L Operator Precomputation" begin
            c = circular_patch(1.0, 128, 1.0)
            tree = ContourDynamics.build_fmm_tree([c])
            ops = ContourDynamics.precompute_level_operators(tree, EulerKernel())
            m2l = ContourDynamics.precompute_m2l_operators(tree, EulerKernel(),
                                                            UnboundedDomain(), ops)
            @test length(m2l) == tree.max_level + 1
            for level_ops in m2l
                @test length(level_ops.operators) > 0
                @test length(level_ops.operators) <= 189
            end
        end
    end

    # Proxy FMM vs Direct — exercise the experimental implementation directly.
    @testset "Proxy FMM vs Direct Accuracy" begin
        @testset "Euler Unbounded" begin
            c = circular_patch(1.0, 128, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            N = total_nodes(prob)
            vel_direct = zeros(SVector{2,Float64}, N)
            vel_fmm = zeros(SVector{2,Float64}, N)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._experimental_fmm_velocity!(vel_fmm, prob)
            max_err = maximum(sqrt(sum((vel_fmm[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-10
        end

        @testset "Euler Periodic" begin
            c = circular_patch(1.0, 96, 1.0)
            prob = ContourProblem(EulerKernel(), PeriodicDomain(2pi, 2pi), [c])
            N = total_nodes(prob)
            vel_direct = zeros(SVector{2,Float64}, N)
            vel_fmm = zeros(SVector{2,Float64}, N)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._experimental_fmm_velocity!(vel_fmm, prob)
            max_err = maximum(sqrt(sum((vel_fmm[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-10
        end

        @testset "QG Unbounded" begin
            c = circular_patch(1.0, 128, 1.0)
            prob = ContourProblem(QGKernel(2.0), UnboundedDomain(), [c])
            N = total_nodes(prob)
            vel_direct = zeros(SVector{2,Float64}, N)
            vel_fmm = zeros(SVector{2,Float64}, N)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._experimental_fmm_velocity!(vel_fmm, prob)
            max_err = maximum(sqrt(sum((vel_fmm[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-10
        end

        @testset "QG Periodic" begin
            c = circular_patch(1.0, 96, 1.0)
            prob = ContourProblem(QGKernel(2.0), PeriodicDomain(2pi, 2pi), [c])
            N = total_nodes(prob)
            vel_direct = zeros(SVector{2,Float64}, N)
            vel_fmm = zeros(SVector{2,Float64}, N)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._experimental_fmm_velocity!(vel_fmm, prob)
            max_err = maximum(sqrt(sum((vel_fmm[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-10
        end

        @testset "SQG Periodic" begin
            c = circular_patch(1.0, 96, 1.0)
            prob = ContourProblem(SQGKernel(0.05), PeriodicDomain(2pi, 2pi), [c])
            N = total_nodes(prob)
            vel_direct = zeros(SVector{2,Float64}, N)
            vel_fmm = zeros(SVector{2,Float64}, N)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._experimental_fmm_velocity!(vel_fmm, prob)
            max_err = maximum(sqrt(sum((vel_fmm[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-10
        end

        @testset "Two-Layer QG Unbounded" begin
            Ld = SVector(1.0)
            F = 1.0 / (2 * Ld[1]^2)
            coupling = SMatrix{2,2}(-F, F, F, -F)
            kernel = MultiLayerQGKernel(Ld, coupling)
            c1 = circular_patch(0.5, 96, 1.0)
            c2_nodes = [SVector(2.0 + 0.5*cos(2*pi*i/96), 0.5*sin(2*pi*i/96)) for i in 0:95]
            c2 = PVContour(c2_nodes, 0.5)
            prob = MultiLayerContourProblem(kernel, UnboundedDomain(), ([c1], [c2]))
            vel_direct = ContourDynamics._make_vel_tuple(prob)
            vel_fmm = ContourDynamics._make_vel_tuple(prob)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._experimental_fmm_velocity!(vel_fmm, prob)
            for i in 1:2
                max_err = maximum(sqrt(sum((vel_fmm[i][j] - vel_direct[i][j]).^2)) /
                                  max(sqrt(sum(vel_direct[i][j].^2)), 1e-15) for j in eachindex(vel_direct[i]))
                @test max_err < 1e-10
            end
        end

        @testset "Two-Layer QG Periodic" begin
            Ld = SVector(1.0)
            F = 1.0 / (2 * Ld[1]^2)
            coupling = SMatrix{2,2}(-F, F, F, -F)
            kernel = MultiLayerQGKernel(Ld, coupling)
            c1 = circular_patch(0.5, 96, 1.0)
            c2_nodes = [SVector(2.0 + 0.5*cos(2*pi*i/96), 0.5*sin(2*pi*i/96)) for i in 0:95]
            c2 = PVContour(c2_nodes, 0.5)
            prob = MultiLayerContourProblem(kernel, PeriodicDomain(2pi, 2pi), ([c1], [c2]))
            vel_direct = ContourDynamics._make_vel_tuple(prob)
            vel_fmm = ContourDynamics._make_vel_tuple(prob)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._experimental_fmm_velocity!(vel_fmm, prob)
            for i in 1:2
                max_err = maximum(sqrt(sum((vel_fmm[i][j] - vel_direct[i][j]).^2)) /
                                  max(sqrt(sum(vel_direct[i][j].^2)), 1e-15) for j in eachindex(vel_direct[i]))
                @test max_err < 1e-10
            end
        end
    end

    @testset "Large-Problem Dispatcher" begin
        c = circular_patch(1.0, extended ? 1200 : 400, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        vel = zeros(SVector{2,Float64}, total_nodes(prob))
        expected = similar(vel)

        velocity!(vel, prob)

        if ContourDynamics._FMM_ACCELERATION_ENABLED
            ContourDynamics._fmm_velocity!(expected, prob)
        elseif total_nodes(prob) >= ContourDynamics._FMM_THRESHOLD
            ContourDynamics._treecode_velocity!(expected, prob)
        else
            ContourDynamics._direct_velocity!(expected, prob)
        end

        @test vel ≈ expected atol=1e-14 rtol=0
    end

    @testset "Production Treecode Accuracy" begin
        @testset "Euler Unbounded" begin
            N_tree = extended ? 600 : 200
            c = circular_patch(1.0, N_tree, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            N = total_nodes(prob)
            vel_direct = zeros(SVector{2,Float64}, N)
            vel_tree = similar(vel_direct)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._treecode_velocity!(vel_tree, prob)
            max_err = maximum(sqrt(sum((vel_tree[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 2e-3
        end

        @testset "QG Unbounded" begin
            N_tree = extended ? 600 : 200
            c = circular_patch(1.0, N_tree, 1.0)
            prob = ContourProblem(QGKernel(2.0), UnboundedDomain(), [c])
            N = total_nodes(prob)
            vel_direct = zeros(SVector{2,Float64}, N)
            vel_tree = similar(vel_direct)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._treecode_velocity!(vel_tree, prob)
            max_err = maximum(sqrt(sum((vel_tree[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 2e-3
        end

        @testset "SQG Unbounded" begin
            N_tree = extended ? 600 : 200
            c = circular_patch(1.0, N_tree, 1.0)
            prob = ContourProblem(SQGKernel(0.05), UnboundedDomain(), [c])
            N = total_nodes(prob)
            vel_direct = zeros(SVector{2,Float64}, N)
            vel_tree = similar(vel_direct)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._treecode_velocity!(vel_tree, prob)
            max_err = maximum(sqrt(sum((vel_tree[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-3
        end

        @testset "Two Patches" begin
            N_patch = extended ? 300 : 100
            tree_tol = extended ? 2e-3 : 5e-3
            c1 = circular_patch(0.5, N_patch, 1.0)
            c2_nodes = [SVector(3.0 + 0.5*cos(2*pi*i/N_patch), 0.5*sin(2*pi*i/N_patch)) for i in 0:(N_patch-1)]
            c2 = PVContour(c2_nodes, -0.5)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
            N = total_nodes(prob)
            vel_direct = zeros(SVector{2,Float64}, N)
            vel_tree = similar(vel_direct)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._treecode_velocity!(vel_tree, prob)
            max_err = maximum(sqrt(sum((vel_tree[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < tree_tol
        end

        @testset "Euler Periodic" begin
            domain = PeriodicDomain(Float64(pi), Float64(pi))
            N_per = extended ? 300 : 100
            c = circular_patch(0.5, N_per, 1.0)
            prob = ContourProblem(EulerKernel(), domain, [c])
            N = total_nodes(prob)
            vel_direct = zeros(SVector{2,Float64}, N)
            vel_tree = similar(vel_direct)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._treecode_velocity!(vel_tree, prob)
            max_err = maximum(sqrt(sum((vel_tree[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-12
        end
    end

    @testset "Conservation Baseline" begin
        @testset "Kirchhoff Ellipse" begin
            e = elliptical_patch(2.0, 1.0, 128, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [e])
            stepper = RK4Stepper(0.01, total_nodes(prob))
            params = SurgeryParams(0.005, 0.02, 0.3, 1e-6, 50)

            circ0 = circulation(prob)
            area0 = vortex_area(prob.contours[1])

            evolve!(prob, stepper, params; nsteps=20)

            @test abs(circulation(prob) - circ0) / abs(circ0) < 1e-6
            @test abs(vortex_area(prob.contours[1]) - area0) / abs(area0) < 1e-4
        end
    end

    @testset "Periodic Accelerator Paths" begin
        @testset "Euler Periodic" begin
            domain = PeriodicDomain(Float64(pi), Float64(pi))
            N_per = extended ? 300 : 100
            c = circular_patch(0.5, N_per, 1.0)
            prob = ContourProblem(EulerKernel(), domain, [c])
            N = total_nodes(prob)

            vel_direct = zeros(SVector{2,Float64}, N)
            vel_fmm = zeros(SVector{2,Float64}, N)

            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._fmm_velocity!(vel_fmm, prob)

            max_err = maximum(sqrt(sum((vel_fmm[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            if ContourDynamics._FMM_ACCELERATION_ENABLED
                @test max_err < 1e-8
            else
                @test max_err == 0.0
            end
        end

        @testset "QG Periodic" begin
            domain = PeriodicDomain(Float64(pi), Float64(pi))
            N_per = extended ? 300 : 100
            c = circular_patch(0.5, N_per, 1.0)
            prob = ContourProblem(QGKernel(1.0), domain, [c])
            N = total_nodes(prob)
            vel_direct = zeros(SVector{2,Float64}, N)
            vel_tree = similar(vel_direct)
            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._treecode_velocity!(vel_tree, prob)
            max_err = maximum(sqrt(sum((vel_tree[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-8
        end
    end

    @testset "Multi-Layer Treecode" begin
        @testset "Two-Layer QG" begin
            Ld = SVector(1.0)
            F = 1.0 / (2 * Ld[1]^2)
            coupling = SMatrix{2,2}(-F, F, F, -F)
            kernel = MultiLayerQGKernel(Ld, coupling)

            N_ml = extended ? 300 : 100
            c1 = circular_patch(0.5, N_ml, 1.0)
            c2_nodes = [SVector(2.0 + 0.5*cos(2*pi*i/N_ml), 0.5*sin(2*pi*i/N_ml)) for i in 0:(N_ml-1)]
            c2 = PVContour(c2_nodes, 0.5)
            layers = ([c1], [c2])
            prob = MultiLayerContourProblem(kernel, UnboundedDomain(), layers)

            vel_direct = ContourDynamics._make_vel_tuple(prob)
            vel_tree = ContourDynamics._make_vel_tuple(prob)

            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._treecode_velocity!(vel_tree, prob)

            for i in 1:2
                for j in eachindex(vel_direct[i])
                    d = sqrt(sum((vel_tree[i][j] - vel_direct[i][j]).^2))
                    ref = max(sqrt(sum(vel_direct[i][j].^2)), 1e-15)
                    @test d / ref < 5e-3
                end
            end
        end
    end
end
