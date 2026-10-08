using Test, ContourDynamics, StaticArrays, LinearAlgebra

# Storage/workspace testsets run first, matching the historical test_groups.jl
# order (test_storage_workspace.jl preceded test_core.jl).

@testset "Storage ownership" begin
    input = [circular_patch(1., 16, 1.)]
    prob = ContourProblem(EulerKernel(), UnboundedDomain(), input)
    @test contours(prob) === input
    @test prob.contours === input
    @test materialize_contours(prob) === input # compatibility accessor
    saved = snapshot_contours(prob)
    original = saved[1].nodes[1]
    input[1].nodes[1] = SVector(2., 0.)
    input[1].corners[1] = true
    @test saved[1].nodes[1] == original
    @test !saved[1].corners[1]
    @test contours(prob)[1].nodes[1] == SVector(2., 0.)

    state = DeviceContourState(saved, CPU())
    storage = ContourDynamics._DeviceContourStorage(state)
    @test_throws ErrorException ContourDynamics._borrow_contours(storage)
    before = ContourDynamics._snapshot_storage(storage)
    state.x[1] += 1
    after = ContourDynamics._snapshot_storage(storage)
    @test after[1].nodes[1][1] == before[1].nodes[1][1] + 1
    @test before[1].nodes[1] == original

    kernel = MultiLayerQGKernel(SVector(1.), SMatrix{2,2}(-.5, .5, .5, -.5))
    multi = MultiLayerContourProblem(kernel, UnboundedDomain(), (saved, deepcopy(saved)))
    copy_layers = snapshot_contours(multi)
    contours(multi)[2][1].nodes[1] += SVector(1., 0.)
    @test copy_layers[2][1].nodes[1] == original
    wrapped = Problem(prob, RK4Stepper(.01, total_nodes(prob)), nothing)
    @test snapshot_contours(wrapped)[1].nodes == contours(prob)[1].nodes
end

@testset "Explicit computational workspace" begin
    ws = ExecutionWorkspace()
    prob = Problem(contours=[circular_patch(1., 16, 1.)], dt=.01, workspace=ws)
    other = Problem(contours=[circular_patch(1., 12, 1.)], dt=.01)
    @test execution_workspace(prob) === ws
    @test execution_workspace(other) !== ws
    @test prob.contour_problem.velocity_scratch === ws.cpu

    state = DeviceContourState(snapshot_contours(prob), CPU())
    vel = zeros(SVector{2,Float64}, total_nodes(prob))
    ContourDynamics._ka_velocity_from_state!(vel, state, EulerKernel(), UnboundedDomain(), CPU(); workspace=ws)
    @test !isempty(ws.buffers)
    buffer = ContourDynamics._get_state_workspace(CPU(), Float64, length(vel); workspace=ws)
    other_buffer = ContourDynamics._get_state_workspace(CPU(), Float64, 12; workspace=execution_workspace(other))
    @test ContourDynamics._get_state_workspace(CPU(), Float64, length(vel); workspace=ws) === buffer
    @test buffer !== other_buffer
    ContourDynamics._rk4_state_step!(state, EulerKernel(), UnboundedDomain(), prob.stepper, CPU(); workspace=ws)
    @test ContourDynamics._get_state_workspace(CPU(), Float64, length(vel); workspace=ws) === buffer
    clear_state_workspace_cache!(prob)
    @test isempty(ws.buffers)
    @test !isempty(execution_workspace(other).buffers)
    @test prob.contour_problem.velocity_scratch === ws.cpu

    # Reusing a workspace sequentially across different models must invalidate
    # modal transforms even when both problems have the same number of layers.
    layers = ([circular_patch(.3, 12, 1.)], [circular_patch(.2, 12, -.5; cx=.6)])
    k1 = MultiLayerQGKernel(SVector(1.), SMatrix{2,2}(-.5, .5, .5, -.5))
    k2 = MultiLayerQGKernel(SVector(1.), SMatrix{2,2}(-.75, .25, .75, -.25))
    p1 = MultiLayerContourProblem(k1, UnboundedDomain(), deepcopy(layers); workspace=ws)
    p2 = MultiLayerContourProblem(k2, UnboundedDomain(), deepcopy(layers); workspace=ws)
    reference = MultiLayerContourProblem(k2, UnboundedDomain(), deepcopy(layers))
    velocity(p1, SVector(.1, .2))
    @test all(isapprox.(velocity(p2, SVector(.1, .2)), velocity(reference, SVector(.1, .2)); rtol=1e-12))
end


@testset "ContourDynamics.jl" begin
    @testset "Core Types" begin
        # PVContour construction
        c = circular_patch(1.0, 64, 1.0)
        @test nnodes(c) == 64
        @test c.pv == 1.0
        @test !any(c.corners)
        @test isempty(corner_indices(c))
        corners = falses(nnodes(c))
        corners[1] = true
        c_corner = PVContour(c.nodes, c.pv, c.wrap, corners)
        @test is_corner(c_corner, 1)
        @test corner_indices(c_corner) == [1]

        # EulerKernel
        k = EulerKernel()
        @test k isa AbstractKernel

        # QGKernel validation
        @test_throws ArgumentError QGKernel(-1.0)
        qg = QGKernel(2.5)
        @test qg.Ld == 2.5

        # Domains
        d = UnboundedDomain()
        @test d isa AbstractDomain
        pd = PeriodicDomain(Float64(π), Float64(π))
        @test pd.Lx == Float64(π)
        @test_throws MethodError PeriodicDomain(1, 1)  # Int not allowed
        @test_throws ArgumentError PeriodicDomain(-1.0, 1.0)

        # ContourProblem
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        @test total_nodes(prob) == 64

        # SurgeryParams validation
        sp = SurgeryParams(0.001, 0.005, 0.1, 1e-6, 10)
        @test sp.δ == 0.001
        @test sp.μ == 0.005
        @test sp.Δ_max == 0.1
        @test sp.delta == sp.δ
        @test sp.mu == sp.μ
        @test sp.Delta_max == sp.Δ_max
        @test sp.n_surgery == 10
        @test_throws ArgumentError SurgeryParams(0.01, 0.005, 0.003, 1e-6, 10)  # Δ_max < μ

        # RK4Stepper construction
        rk = RK4Stepper(0.01, 64)
        @test rk.dt == 0.01
        @test length(rk.k1) == 64
    end

    @testset "Euler Kernel" begin
        c = circular_patch(1.0, 64, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        vel = zeros(SVector{2, Float64}, total_nodes(prob))
        velocity!(vel, prob)

        # Velocity at each node: tangential, magnitude = pv*R/2 = 0.5
        expected_speed = 0.5
        for i in 1:nnodes(c)
            speed = sqrt(vel[i][1]^2 + vel[i][2]^2)
            @test speed ≈ expected_speed rtol=0.02
        end

        # Check tangential direction: velocity perpendicular to position (dot product ≈ 0)
        for i in 1:nnodes(c)
            pos = c.nodes[i]
            @test abs(vel[i][1]*pos[1] + vel[i][2]*pos[2]) < 0.01
        end
    end

    @testset "QG Kernel" begin
        N = 32
        c = circular_patch(1.0, N, 1.0)
        prob_euler = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        prob_qg = ContourProblem(QGKernel(100.0), UnboundedDomain(), [c])

        vel_euler = zeros(SVector{2, Float64}, N)
        vel_qg = zeros(SVector{2, Float64}, N)
        velocity!(vel_euler, prob_euler)
        velocity!(vel_qg, prob_qg)

        # Large Ld → QG approaches Euler
        for i in 1:N
            @test vel_qg[i] ≈ vel_euler[i] rtol=0.1
        end

        # Small Ld → QG velocity weaker than Euler
        prob_qg_small = ContourProblem(QGKernel(0.5), UnboundedDomain(), [c])
        vel_qg_small = zeros(SVector{2, Float64}, N)
        velocity!(vel_qg_small, prob_qg_small)

        euler_speed = sqrt(vel_euler[1][1]^2 + vel_euler[1][2]^2)
        qg_speed = sqrt(vel_qg_small[1][1]^2 + vel_qg_small[1][2]^2)
        @test qg_speed < euler_speed
    end

    @testset "Per-Contour Diagnostics" begin
        c = circular_patch(1.0, 64, 1.0)
        @test vortex_area(c) ≈ π rtol=5e-3

        cx = centroid(c)
        @test abs(cx[1]) < 1e-10
        @test abs(cx[2]) < 1e-10

        e = elliptical_patch(2.0, 1.0, 64, 1.0)
        @test vortex_area(e) ≈ 2π rtol=5e-3

        ratio, angle = ellipse_moments(e)
        @test ratio ≈ 2.0 rtol=0.05
        @test abs(angle) < 0.1

        # Rotated ellipse: Jxy ≠ 0, catches incorrect product-of-inertia formula
        θ = π / 4  # 45 degrees
        e_rot = rotated_elliptical_patch(2.0, 1.0, 128, 1.0, θ)
        ratio_rot, angle_rot = ellipse_moments(e_rot)
        @test ratio_rot ≈ 2.0 rtol=0.05
        @test angle_rot ≈ θ atol=0.05

        # 30-degree rotation
        θ30 = π / 6
        e_rot30 = rotated_elliptical_patch(2.0, 1.0, 128, 1.0, θ30)
        ratio30, angle30 = ellipse_moments(e_rot30)
        @test ratio30 ≈ 2.0 rtol=0.05
        @test angle30 ≈ θ30 atol=0.05
    end

    @testset "Problem-Level Diagnostics" begin
        c = circular_patch(1.0, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

        @test circulation(prob) ≈ π rtol=0.01
        @test enstrophy(prob) ≈ π / 2 rtol=0.01

        E = energy(prob)
        @test E > 0
        @test isfinite(E)

        L = angular_momentum(prob)
        @test L ≈ π / 2 rtol=0.02
    end

    @testset "Energy double sums fold symmetric pairs" begin
        # Energy visits each unordered segment and contour pair once. Compare
        # with ordered double sums: a copy of a contour is not identical to it,
        # so pairing with the copy runs the full rectangular loop.
        Φ = rv -> ContourDynamics._euler_energy_potential_scalar(rv[1]^2 + rv[2]^2)
        pair(ci, cj) = ContourDynamics._energy_contour_pair(ci, cj, Φ)
        for n in (3, 4, 7, 10, 65, 66)   # odd and even; 65+ runs the threaded loop
            c = elliptical_patch(0.8, 0.5, n, 1.0)
            @test pair(c, c) ≈ pair(c, deepcopy(c)) rtol=1e-13
        end

        cs = [elliptical_patch(0.8, 0.5, 31, 1.0), circular_patch(0.3, 20, -0.6; cx=1.5),
              circular_patch(0.2, 12, 0.4; cy=-1.4)]
        ordered = ContourDynamics._normalize_energy(
            sum(ci.pv * cj.pv * pair(ci, deepcopy(cj)) for ci in cs, cj in cs))
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), cs)
        @test energy(prob) ≈ ordered rtol=1e-13
        @test ContourDynamics._ka_energy(prob, CPU()) ≈ ordered rtol=1e-13

        F = 0.5
        kernel = MultiLayerQGKernel(SVector(1.0), SMatrix{2,2,Float64}(-F, F, F, -F))
        layers = (cs[1:2], cs[3:3])
        raw = 0.0
        for (mode, λ) in pairs(kernel.eigenvalues)
            raw += ContourDynamics._dispatch_qg_mode(kernel, λ) do mode_kernel
                s = 0.0
                for la in 1:2, lb in 1:2, ci in layers[la], cj in layers[lb]
                    w = kernel.physical_to_modal[mode, la] * kernel.physical_to_modal[mode, lb]
                    s += w * ci.pv * cj.pv * ContourDynamics._modal_pair_energy(
                        ci, deepcopy(cj), mode_kernel, UnboundedDomain(), nothing, zeros(64))
                end
                s
            end
        end
        ml = MultiLayerContourProblem(kernel, UnboundedDomain(), layers)
        @test energy(ml) ≈ ContourDynamics._normalize_energy(raw) rtol=1e-12
        states = map(layer -> DeviceContourState(layer, CPU()), layers)
        @test ContourDynamics._ka_multilayer_energy_from_states(
            states, kernel, UnboundedDomain(), CPU()) ≈ energy(ml) rtol=1e-12
    end

    @testset "Node Management" begin
        nodes = SVector{2, Float64}[
            SVector(0.0, 0.0), SVector(0.001, 0.0), SVector(0.002, 0.0),
            SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0),
        ]
        c = PVContour(nodes, 1.0)
        params = SurgeryParams(0.002, 0.01, 0.05, 1e-6, 10)

        c_new = remesh(c, params)

        for i in 1:nnodes(c_new)
            j = mod1(i + 1, nnodes(c_new))
            d = c_new.nodes[j] - c_new.nodes[i]
            spacing = sqrt(d[1]^2 + d[2]^2)
            @test spacing >= params.μ * 0.9
            @test spacing <= params.Δ_max * 1.1
        end

        @test vortex_area(c_new) ≈ vortex_area(c) rtol=0.1
    end


    @testset "Time Steppers" begin
        @testset "RK4 single step" begin
            c = circular_patch(1.0, 64, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            stepper = RK4Stepper(0.01, total_nodes(prob))

            area_before = vortex_area(prob.contours[1])
            timestep!(prob, stepper)
            area_after = vortex_area(prob.contours[1])

            @test area_after ≈ area_before rtol=1e-8
        end

        @testset "evolve! with callbacks" begin
            c = circular_patch(1.0, 64, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            stepper = RK4Stepper(0.01, total_nodes(prob))
            params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)

            initial_area = vortex_area(prob.contours[1])
            areas = Float64[]
            cb = (p, step) -> push!(areas, vortex_area(p.contours[1]))

            evolve!(prob, stepper, params; nsteps=10, callbacks=[cb])
            @test length(areas) == 11  # step 0 (initial) + steps 1-10
            @test all(a -> abs(a - initial_area) / abs(initial_area) < 1e-4, areas)
        end
    end




    @testset "Periodic Domain (Ewald)" begin
        clear_ewald_cache!()
        c = circular_patch(0.1, 16, 1.0)
        prob_unbounded = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        prob_periodic = ContourProblem(EulerKernel(), PeriodicDomain(10.0, 10.0), [c])

        vel_u = zeros(SVector{2, Float64}, 16)
        vel_p = zeros(SVector{2, Float64}, 16)
        velocity!(vel_u, prob_unbounded)
        velocity!(vel_p, prob_periodic)

        for i in 1:16
            @test vel_p[i] ≈ vel_u[i] rtol=0.15
        end
    end

    @testset "Periodic contour wrapping preserves geometry" begin
        domain = PeriodicDomain(1.0, 1.0)
        R, N = 0.3, 64
        cx = 0.85
        nodes = [SVector(cx + R * cos(2π * k / N), R * sin(2π * k / N)) for k in 0:(N - 1)]
        c = PVContour(nodes, 1.0)
        prob = ContourProblem(EulerKernel(), domain, [c])

        A0 = vortex_area(c)
        ctr0 = centroid(c)
        rel0 = [node - ctr0 for node in c.nodes]

        wrap_nodes!(prob)

        wrapped = prob.contours[1]
        A1 = vortex_area(wrapped)
        ctr1 = centroid(wrapped)
        rel1 = [node - ctr1 for node in wrapped.nodes]

        @test A1 ≈ A0 rtol=1e-12 atol=1e-12
        @test ctr1 ≈ ContourDynamics.wrap_node(ctr0, domain) rtol=1e-12 atol=1e-12
        @test all(rel1[i] ≈ rel0[i] for i in eachindex(rel0))
    end

    @testset "Periodic shift recenters straddling contour" begin
        # Contour stored straddling the x = L seam with nonzero area. Its raw
        # area-weighted centroid lands (wrongly) inside the domain, so the naive
        # `iszero(ref)` fallback never fires and a zero shift leaves it
        # straddling. The minimum-image reference is just outside the domain
        # (x ≈ 1), so the correct lattice shift is nonzero.
        domain = PeriodicDomain(1.0, 1.0)
        nodes = SVector{2,Float64}[
            SVector(0.9, 0.0), SVector(-0.9, 0.4), SVector(-0.9, -0.2),
        ]
        c = PVContour(nodes, 1.0)

        shift = ContourDynamics.contour_periodic_shift(c, domain)
        @test shift[1] ≈ -2.0 atol=1e-12
        @test shift[2] ≈ 0.0 atol=1e-12

        # After shifting, the minimum-image centroid lies inside [-L, L).
        shifted = [n + shift for n in nodes]
        p0 = shifted[1]
        acc = sum(shifted) do p
            d = p - p0
            SVector(d[1] - 2 * round(d[1] / 2), d[2] - 2 * round(d[2] / 2))
        end
        ref = p0 + acc / length(shifted)
        @test -1.0 <= ref[1] < 1.0
        @test -1.0 <= ref[2] < 1.0
    end





    @testset "Multi-Layer QG" begin
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)

        kernel = MultiLayerQGKernel(Ld, coupling)
        @test nlayers(kernel) == 2

        @test_throws ArgumentError MultiLayerQGKernel(SVector(3.0), coupling)

        c1 = circular_patch(1.0, 64, 1.0)
        c2 = circular_patch(0.5, 64, -1.0)
        domain = UnboundedDomain()
        prob = MultiLayerContourProblem(kernel, domain, ([c1], [c2]))

        @test nlayers(prob) == 2
        @test total_nodes(prob) == 128

        vel = (zeros(SVector{2, Float64}, 64), zeros(SVector{2, Float64}, 64))
        velocity!(vel, prob)

        @test all(v -> all(isfinite, v), vel[1])
        @test all(v -> all(isfinite, v), vel[2])
        @test any(v -> sqrt(v[1]^2 + v[2]^2) > 1e-10, vel[1])

        # Changing the deformation radius through a consistent coupling matrix
        # should change the multilayer velocity field.
        Ld2 = SVector(2.0)
        F2 = 1.0 / (2 * Ld2[1]^2)
        kernel2 = MultiLayerQGKernel(Ld2, SMatrix{2,2}(-F2, F2, F2, -F2))
        prob2 = MultiLayerContourProblem(kernel2, domain, ([c1], [c2]))
        vel2 = (zeros(SVector{2, Float64}, 64), zeros(SVector{2, Float64}, 64))
        velocity!(vel2, prob2)
        @test maximum(sqrt((vel[1][i][1] - vel2[1][i][1])^2 + (vel[1][i][2] - vel2[1][i][2])^2) for i in 1:64) > 1e-6
    end






    @testset "Spanning Contours & Beta Staircase" begin
        T = Float64
        domain = PeriodicDomain(T(3.0))

        # beta_staircase creates paper-style mid-step spanning contours
        staircase = beta_staircase(T(1.0), domain, 6; nodes_per_contour=16)
        @test length(staircase) == 6
        @test [c.nodes[1][2] for c in staircase] ≈ T[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

        # Each contour is spanning with correct wrap
        for c in staircase
            @test is_spanning(c)
            @test c.wrap == SVector{2,T}(6.0, 0.0)
            @test nnodes(c) == 16
        end

        # PV jump = beta * dy
        dy = 2 * 3.0 / 6
        @test staircase[1].pv ≈ 1.0 * dy

        # Spanning contours have zero area (skip in diagnostics)
        @test vortex_area(staircase[1]) == zero(T)

        # next_node wraps correctly for spanning contours
        c = staircase[1]
        last_node = c.nodes[end]
        wrapped = next_node(c, nnodes(c))
        @test wrapped ≈ c.nodes[1] + c.wrap

        # Velocity computation works with spanning + closed contours mixed
        vortex = PVContour([SVector{2,T}(0.3*cos(2π*k/16), 0.5 + 0.3*sin(2π*k/16)) for k in 0:15], T(2π))
        all_contours = vcat(staircase, [vortex])
        kernel = QGKernel(T(1.0))
        prob = ContourProblem(kernel, domain, all_contours)
        vel = zeros(SVector{2,T}, total_nodes(prob))
        velocity!(vel, prob)
        @test all(v -> all(isfinite, v), vel)

        # Surgery skips spanning contours in reconnection and filament removal
        params = SurgeryParams(T(0.01), T(0.05), T(0.5), T(1e-4), 10)
        surgery!(prob, params)
        # All spanning contours should survive surgery
        n_spanning = count(is_spanning, prob.contours)
        @test n_spanning == 6

        # Remesh preserves wrap
        remeshed = remesh(staircase[1], params)
        @test is_spanning(remeshed)
        @test remeshed.wrap == staircase[1].wrap
    end


end


@testset "Contour constructor invariants" begin
    for T in (Float32, Float64)
        c = circular_patch(1, 64, 1; T=T)
        for flags in (falses(1), [false], falses(65), fill(false, 65))
            @test_throws DimensionMismatch PVContour(c.nodes, c.pv, c.wrap, flags)
            @test_throws DimensionMismatch PVContour{T}(c.nodes, c.pv, c.wrap, flags)
        end
        for flags in (falses(64), fill(false, 64))
            flags[3] = true
            for ctor in (PVContour, PVContour{T})
                valid = ctor(c.nodes, 1, c.wrap, flags)
                @test corner_indices(valid) == [3]
                @test valid.pv === one(T)
            end
        end
    end
end

@testset "Curvature is independent of coordinate units" begin
    for T in (Float32, Float64), radius in (1, 0.01, 1e-5)
        c = circular_patch(radius, 64, 1; T=T)
        R = T(radius)
        tolerance = T === Float32 ? T(2e-4) : T(2e-12)
        @test all(k -> isapprox(k * R, one(T); rtol=tolerance),
                  ContourDynamics._signed_node_curvatures(c))
        state = DeviceContourState([c], CPU())
        segments = ContourDynamics._state_segment_data(state, CPU())
        @test all(k -> isapprox(k * R, one(T); rtol=tolerance), segments.ka)
        @test all(k -> isapprox(k * R, one(T); rtol=tolerance), segments.kb)
        path_curvatures = ContourDynamics._signed_path_curvatures(c.nodes, c.corners)
        @test all(k -> isapprox(k * R, one(T); rtol=tolerance), path_curvatures[2:end-1])
        reversed = PVContour(reverse(c.nodes), c.pv)
        @test all(k -> isapprox(k * R, -one(T); rtol=tolerance),
                  ContourDynamics._signed_node_curvatures(reversed))

        # At the smallest Float32 radius, the velocity kernel's separate
        # absolute-distance cutoffs dominate; test curvature there directly.
        if T === Float64 || radius >= 0.01
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            @test velocity(prob, c.nodes[1])[2] / R ≈ T(0.5) atol=T(1e-5)
        end
    end
    # Repeated vertices and a vanishing closing chord remain safe degeneracies.
    for points in ((SVector(0., 0.), SVector(0., 0.), SVector(1., 0.)),
                   (SVector(0., 0.), SVector(1., 0.), SVector(0., 0.)))
        c = PVContour(collect(points), 1.)
        @test all(iszero, ContourDynamics._signed_node_curvatures(c))
    end
end

@testset "Resolved weak multilayer modes" begin
    coupling64 = SMatrix{3,3,Float64}([-1 1 0; 1 -1.01 .01; 0 .01 -.01])
    radii64 = SVector{2,Float64}(1 ./ sqrt.(abs.(eigvals(Symmetric(Matrix(coupling64)))[1:2])))
    for T in (Float32, Float64)
        coupling = T.(coupling64)
        kernel = MultiLayerQGKernel(T.(radii64), coupling)
        @test count(λ -> ContourDynamics._is_barotropic_mode(kernel, λ), kernel.eigenvalues) == 1
        # Compare modal inversion to a direct physical-layer solve, including
        # the weak mode whose deformation radius is much larger than the first.
        k2 = T(0.01)
        modal_inverse = kernel.modal_to_physical *
                        Diagonal(inv.(k2 .- kernel.eigenvalues)) * kernel.physical_to_modal
        @test modal_inverse ≈ inv(k2 * I - coupling) rtol=(T === Float32 ? 1e-4 : 1e-12)
    end
end


@testset "Polygon geometry stability" begin
    base_nodes = SVector{2,Float64}[
        SVector(0.0, 0.0),
        SVector(2.0, 0.0),
        SVector(1.0, 1.0),
        SVector(0.0, 1.0),
    ]
    expected_area = 1.5
    expected_centroid = SVector(7 / 9, 4 / 9)
    expected_ellipse_moments = ellipse_moments(PVContour(base_nodes, 1.0))

    @testset "large coordinate translation" begin
        shift = SVector(1.0e8, -1.0e8)
        shifted_nodes = [p + shift for p in base_nodes]
        shifted = PVContour(shifted_nodes, 1.0)

        @test vortex_area(shifted) ≈ expected_area rtol=0 atol=10eps(Float64)
        @test centroid(shifted) - shift ≈ expected_centroid rtol=0 atol=2e-8
        shifted_ratio, shifted_angle = ellipse_moments(shifted)
        @test shifted_ratio ≈ expected_ellipse_moments[1] rtol=1e-7
        @test shifted_angle ≈ expected_ellipse_moments[2] rtol=1e-7

        # Remeshing's private preservation helpers use the same polygon
        # moments and must therefore be translation-stable as well.
        @test ContourDynamics._raw_polygon_area(shifted_nodes) ≈
              expected_area rtol=0 atol=10eps(Float64)
        @test ContourDynamics._raw_polygon_centroid(shifted_nodes) - shift ≈
              expected_centroid rtol=0 atol=2e-8

        state = DeviceContourState([shifted], CPU())
        state_area, state_moment = ContourDynamics._state_area_moment(state, CPU())
        @test only(to_cpu(state_area)) ≈ expected_area rtol=0 atol=10eps(Float64)

        expected_moment = 5 / 3 +
                          2 * shift[1] * (expected_area * expected_centroid[1]) +
                          2 * shift[2] * (expected_area * expected_centroid[2]) +
                          sum(abs2, shift) * expected_area
        @test only(to_cpu(state_moment)) ≈ expected_moment rtol=10eps(Float64)
        @test ContourDynamics._second_moment_r2(shifted) ≈
              expected_moment rtol=10eps(Float64)

        flat = ContourDynamics._pack_flat_topology([shifted], CPU())
        @test ContourDynamics._flat_closed_area2(
            flat.x, flat.y, flat.wrapx, flat.wrapy,
            flat.offsets, flat.lengths, 1) ≈
              2 * expected_area rtol=0 atol=10eps(Float64)

        filament_params = SurgeryParams(0.001, 0.01, 0.5, 1.25, 10)
        @test ContourDynamics._device_filament_flags(
            [PVContour(base_nodes, 1.0), shifted], filament_params, CPU()) ==
              [false, false]

        remesh_params = SurgeryParams(0.001, 0.1, 0.8, 1e-8, 10)
        device_remeshed = only(ContourDynamics._device_remesh_contours(
            [shifted], remesh_params, CPU()))
        @test vortex_area(device_remeshed) ≈ expected_area rtol=0 atol=2e-8

        corner_distorted = copy(shifted_nodes)
        corner_distorted[2] = shift + 0.9 * base_nodes[2]
        corner_distorted[4] = shift + 0.9 * base_nodes[4]
        corners = BitVector((true, false, true, false))
        fixed = copy(corner_distorted[corners])
        ContourDynamics._preserve_closed_area_fixed_corners!(
            corner_distorted, corners, expected_area)
        @test corner_distorted[corners] == fixed
        @test ContourDynamics._raw_polygon_area(corner_distorted) ≈
              expected_area rtol=0 atol=2e-8
    end

    @testset "periodic wrapping uses a stable translated centroid" begin
        shift = SVector(1.0e8, -1.0e8)
        shifted_nodes = [point + shift for point in base_nodes]
        domain = PeriodicDomain(10.0, 10.0)

        refx, refy = ContourDynamics._unwrapped_centroid_core(
            i -> Tuple(shifted_nodes[i]), length(shifted_nodes), 20.0, 20.0)
        @test SVector(refx, refy) - shift ≈ expected_centroid rtol=0 atol=2e-8

        shifted = PVContour(shifted_nodes, 1.0)
        cpu_prob = ContourProblem(EulerKernel(), domain, [deepcopy(shifted)])
        device_state = DeviceContourState([deepcopy(shifted)], CPU())
        wrap_nodes!(cpu_prob)
        ContourDynamics._wrap_state_nodes!(device_state, domain, CPU())

        wrapped_cpu = only(cpu_prob.contours)
        wrapped_device = only(materialize_contours(device_state))
        @test centroid(wrapped_cpu) ≈ expected_centroid rtol=0 atol=2e-8
        @test wrapped_device.nodes == wrapped_cpu.nodes
    end

    @testset "small but nondegenerate polygon" begin
        scale = 1.0e-9
        small_nodes = [scale * p for p in base_nodes]
        small = PVContour(small_nodes, 1.0)

        @test vortex_area(small) ≈ scale^2 * expected_area rtol=10eps(Float64)
        @test centroid(small) ≈ scale * expected_centroid rtol=10eps(Float64)
        @test ContourDynamics._raw_polygon_centroid(small_nodes) ≈
              scale * expected_centroid rtol=10eps(Float64)

        target_area = scale^2 * expected_area

        uniformly_distorted = [0.9 * p for p in small_nodes]
        ContourDynamics._preserve_closed_area!(uniformly_distorted, target_area)
        @test ContourDynamics._raw_polygon_area(uniformly_distorted) ≈
              target_area rtol=100eps(Float64)

        corner_distorted = copy(small_nodes)
        corner_distorted[2] *= 0.9
        corner_distorted[4] *= 0.9
        corners = BitVector((true, false, true, false))
        fixed = copy(corner_distorted[corners])
        ContourDynamics._preserve_closed_area_fixed_corners!(
            corner_distorted, corners, target_area)
        @test corner_distorted[corners] == fixed
        @test ContourDynamics._raw_polygon_area(corner_distorted) ≈
              target_area rtol=100eps(Float64)

        remesh_params = SurgeryParams(1e-12, 1e-10, 8e-10, 1e-30, 10)
        device_remeshed = only(ContourDynamics._device_remesh_contours(
            [small], remesh_params, CPU()))
        @test vortex_area(device_remeshed) ≈ target_area rtol=1e-10

        corner_flags = BitVector((true, false, true, false))
        cornered = PVContour(copy(small_nodes), 1.0,
                             zero(SVector{2,Float64}), corner_flags)
        device_cornered = only(ContourDynamics._device_remesh_contours(
            [cornered], remesh_params, CPU()))
        @test vortex_area(device_cornered) ≈ target_area rtol=1e-10
        for fixed_corner in small_nodes[corner_flags]
            @test any(==(fixed_corner), device_cornered.nodes)
        end
    end
end


@testset "Shape Helpers" begin
    @testset "circular_patch" begin
        c = circular_patch(0.5, 32, 2π)
        @test c isa PVContour{Float64}
        @test nnodes(c) == 32
        @test c.pv == 2π

        # Nodes lie on circle of radius 0.5
        for i in 1:nnodes(c)
            r = sqrt(c.nodes[i][1]^2 + c.nodes[i][2]^2)
            @test r ≈ 0.5 atol=1e-12
        end

        # Center offset
        c2 = circular_patch(1.0, 16, 1.0; cx=2.0, cy=3.0)
        center = sum(c2.nodes) / nnodes(c2)
        @test center[1] ≈ 2.0 atol=1e-10
        @test center[2] ≈ 3.0 atol=1e-10

        # Float32
        c32 = circular_patch(0.5, 16, 1.0; T=Float32)
        @test c32 isa PVContour{Float32}

        # Numeric args auto-promoted
        c_int = circular_patch(1, 16, 1)
        @test c_int isa PVContour{Float64}
    end

    @testset "elliptical_patch" begin
        e = elliptical_patch(2.0, 1.0, 64, 1.0)
        @test nnodes(e) == 64
        @test e.pv == 1.0

        # Area ≈ π*a*b = 2π
        @test vortex_area(e) ≈ 2π rtol=0.01

        # With rotation
        e_rot = elliptical_patch(2.0, 1.0, 64, 1.0; θ=π/4)
        @test vortex_area(e_rot) ≈ 2π rtol=0.01
        _, angle = ellipse_moments(e_rot)
        @test angle ≈ π/4 atol=0.1

        # Center offset
        e2 = elliptical_patch(1.0, 0.5, 32, 1.0; cx=1.0, cy=-1.0)
        center = sum(e2.nodes) / nnodes(e2)
        @test center[1] ≈ 1.0 atol=1e-10
        @test center[2] ≈ -1.0 atol=1e-10
    end

    @testset "rankine_vortex" begin
        v = rankine_vortex(1.0, 64, 2π)
        @test v isa Vector{PVContour{Float64}}
        @test length(v) == 1
        @test v[1].pv ≈ 2π / (π * 1.0^2)  # Γ / (π R²)
        @test nnodes(v[1]) == 64

        # Nodes on unit circle
        for i in 1:nnodes(v[1])
            r = sqrt(v[1].nodes[i][1]^2 + v[1].nodes[i][2]^2)
            @test r ≈ 1.0 atol=1e-12
        end

        # Center offset
        v2 = rankine_vortex(0.5, 32, 1.0; cx=1.0, cy=2.0)
        center = sum(v2[1].nodes) / nnodes(v2[1])
        @test center[1] ≈ 1.0 atol=1e-10
        @test center[2] ≈ 2.0 atol=1e-10
    end
end
