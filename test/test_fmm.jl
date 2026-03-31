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
            c = circular_patch(1.0, 200, 1.0)
            tree = ContourDynamics.build_fmm_tree([c])
            ops = ContourDynamics.precompute_level_operators(tree, EulerKernel())
            @test length(ops) == tree.max_level + 1
            p = ContourDynamics._FMM_PROXY_ORDER
            p_check = ContourDynamics._FMM_CHECK_ORDER
            for op in ops
                @test size(op.check_to_proxy_pinv) == (p, p_check)
            end
        end

        @testset "S2M Far-Field Accuracy" begin
            c = circular_patch(0.5, 64, 1.0)
            contours = [c]
            tree = ContourDynamics.build_fmm_tree(contours; max_per_leaf=100)
            ops = ContourDynamics.precompute_level_operators(tree, EulerKernel())

            p = ContourDynamics._FMM_PROXY_ORDER
            proxy_data = [ContourDynamics.ProxyData(
                zeros(SVector{2,Float64}, p),
                zeros(SVector{2,Float64}, p)) for _ in 1:length(tree.boxes)]

            ContourDynamics._s2m!(proxy_data, tree, contours, EulerKernel(),
                                  UnboundedDomain(), ops, nothing)

            far_pt = SVector(5.0, 5.0)
            leaf = tree.leaf_indices[1]
            box = tree.boxes[leaf]
            proxy_pts = ContourDynamics._proxy_points(box.center, box.half_width, p)
            v_proxy = zero(SVector{2,Float64})
            for k in 1:p
                G = ContourDynamics._kernel_value(EulerKernel(), UnboundedDomain(),
                                                  far_pt, proxy_pts[k])
                v_proxy = v_proxy + G * proxy_data[leaf].equiv_strengths[k]
            end

            prob = ContourProblem(EulerKernel(), UnboundedDomain(), contours)
            v_direct = velocity(prob, far_pt)

            @test v_proxy ≈ v_direct rtol=1e-10
        end
    end

    @testset "Translation Operators" begin
        @testset "M2L Operator Precomputation" begin
            c = circular_patch(1.0, 300, 1.0)
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

    @testset "FMM vs Direct Accuracy" begin
        @testset "Euler Unbounded" begin
            c = circular_patch(1.0, 300, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            N = total_nodes(prob)

            vel_direct = zeros(SVector{2,Float64}, N)
            vel_fmm = zeros(SVector{2,Float64}, N)

            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._fmm_velocity!(vel_fmm, prob)

            max_err = maximum(sqrt(sum((vel_fmm[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-10
        end

        @testset "QG Unbounded" begin
            c = circular_patch(1.0, 300, 1.0)
            prob = ContourProblem(QGKernel(2.0), UnboundedDomain(), [c])
            N = total_nodes(prob)

            vel_direct = zeros(SVector{2,Float64}, N)
            vel_fmm = zeros(SVector{2,Float64}, N)

            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._fmm_velocity!(vel_fmm, prob)

            max_err = maximum(sqrt(sum((vel_fmm[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-10
        end

        @testset "SQG Unbounded" begin
            c = circular_patch(1.0, 300, 1.0)
            prob = ContourProblem(SQGKernel(0.05), UnboundedDomain(), [c])
            N = total_nodes(prob)

            vel_direct = zeros(SVector{2,Float64}, N)
            vel_fmm = zeros(SVector{2,Float64}, N)

            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._fmm_velocity!(vel_fmm, prob)

            max_err = maximum(sqrt(sum((vel_fmm[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-10
        end

        @testset "Two Patches" begin
            c1 = circular_patch(0.5, 150, 1.0)
            c2_nodes = [SVector(3.0 + 0.5*cos(2*pi*i/150), 0.5*sin(2*pi*i/150)) for i in 0:149]
            c2 = PVContour(c2_nodes, -0.5)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
            N = total_nodes(prob)

            vel_direct = zeros(SVector{2,Float64}, N)
            vel_fmm = zeros(SVector{2,Float64}, N)

            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._fmm_velocity!(vel_fmm, prob)

            max_err = maximum(sqrt(sum((vel_fmm[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-10
        end

        @testset "Auto-Switch Activates" begin
            c = circular_patch(1.0, 300, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            vel = zeros(SVector{2,Float64}, total_nodes(prob))
            velocity!(vel, prob)
            @test any(v -> sqrt(v[1]^2 + v[2]^2) > 0.1, vel)
        end
    end

    @testset "FMM Conservation" begin
        @testset "Kirchhoff Ellipse" begin
            e = elliptical_patch(2.0, 1.0, 300, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [e])
            stepper = RK4Stepper(0.01, total_nodes(prob))
            params = SurgeryParams(0.005, 0.02, 0.3, 1e-6, 50)

            circ0 = circulation(prob)
            area0 = vortex_area(prob.contours[1])

            evolve!(prob, stepper, params; nsteps=100)

            @test abs(circulation(prob) - circ0) / abs(circ0) < 1e-6
            @test abs(vortex_area(prob.contours[1]) - area0) / abs(area0) < 1e-4
        end
    end

    @testset "FMM Periodic" begin
        @testset "Euler Periodic" begin
            domain = PeriodicDomain(Float64(pi), Float64(pi))
            c = circular_patch(0.5, 300, 1.0)
            prob = ContourProblem(EulerKernel(), domain, [c])
            N = total_nodes(prob)

            vel_direct = zeros(SVector{2,Float64}, N)
            vel_fmm = zeros(SVector{2,Float64}, N)

            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._fmm_velocity!(vel_fmm, prob)

            max_err = maximum(sqrt(sum((vel_fmm[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-8
        end

        @testset "QG Periodic" begin
            domain = PeriodicDomain(Float64(pi), Float64(pi))
            c = circular_patch(0.5, 300, 1.0)
            prob = ContourProblem(QGKernel(1.0), domain, [c])
            N = total_nodes(prob)

            vel_direct = zeros(SVector{2,Float64}, N)
            vel_fmm = zeros(SVector{2,Float64}, N)

            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._fmm_velocity!(vel_fmm, prob)

            max_err = maximum(sqrt(sum((vel_fmm[i] - vel_direct[i]).^2)) /
                              max(sqrt(sum(vel_direct[i].^2)), 1e-15) for i in 1:N)
            @test max_err < 1e-8
        end
    end

    @testset "FMM Multi-Layer" begin
        @testset "Two-Layer QG" begin
            Ld = SVector(1.0)
            F = 1.0 / (2 * Ld[1]^2)
            coupling = SMatrix{2,2}(-F, F, F, -F)
            kernel = MultiLayerQGKernel(Ld, coupling)

            c1 = circular_patch(0.5, 150, 1.0)
            c2_nodes = [SVector(2.0 + 0.5*cos(2*pi*i/150), 0.5*sin(2*pi*i/150)) for i in 0:149]
            c2 = PVContour(c2_nodes, 0.5)
            layers = ([c1], [c2])
            prob = MultiLayerContourProblem(kernel, UnboundedDomain(), layers)

            vel_direct = ContourDynamics._make_vel_tuple(prob)
            vel_fmm = ContourDynamics._make_vel_tuple(prob)

            ContourDynamics._direct_velocity!(vel_direct, prob)
            ContourDynamics._fmm_velocity!(vel_fmm, prob)

            for i in 1:2
                for j in eachindex(vel_direct[i])
                    @test vel_fmm[i][j] ≈ vel_direct[i][j] rtol=1e-9
                end
            end
        end
    end
end
