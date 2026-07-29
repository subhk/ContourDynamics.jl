using ContourDynamics
using StaticArrays
using Test

# Guard against double-include when run from runtests.jl
@isdefined(circular_patch) || include("test_utils.jl")
@isdefined(_full_rewrite_output_layout) || include("device_rewrite_layout_oracle.jl")

function _assert_device_layout_matches_host(contours, plan)
    host = _full_rewrite_output_layout(contours, plan, CPU())
    dev = ContourDynamics._device_full_rewrite_output_layout(contours, plan, CPU())
    @test dev.total_nodes == host.total_nodes
    for name in (:offsets, :lengths, :op_index, :source_contour, :part,
                 :pv, :wrapx, :wrapy, :out_node_contour)
        @test to_cpu(getproperty(dev, name)) == to_cpu(getproperty(host, name))
    end
end

# Disable scalar indexing on GPU arrays to catch accidental cu_array[i] access.
# Only activates when CUDA is actually loaded.
const _TEST_CUDA_LOADED = Ref(false)
try
    using CUDA
    CUDA.allowscalar(false)
    _TEST_CUDA_LOADED[] = true
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
        if _TEST_CUDA_LOADED[]
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=GPU())
            vel = zeros(SVector{2,Float64}, total_nodes(prob))
            velocity!(vel, prob)
            @test all(v -> all(isfinite, v), vel)
        else
            @test_throws ErrorException ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=GPU())
        end
    end

    @testset "GPU surgery! without CUDA gives helpful error" begin
        c = circular_patch(0.5, 32, 1.0)
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)
        if _TEST_CUDA_LOADED[]
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=GPU())
            surgery!(prob, params)
            @test materialize_contours(prob) isa Vector{PVContour{Float64}}
        else
            @test_throws ErrorException ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=GPU())
        end
    end

    @testset "GPU() accepts supported single-layer periodic and unbounded cases" begin
        c = circular_patch(0.5, 32, 1.0)
        if _TEST_CUDA_LOADED[]
            prob = ContourProblem(QGKernel(1.0), UnboundedDomain(), [c]; dev=GPU())
            @test prob.dev === GPU()
            @test prob.device_state isa ContourDynamics.DeviceContourState

            prob_periodic = ContourProblem(QGKernel(1.0), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
            @test prob_periodic.dev === GPU()
            @test prob_periodic.device_state isa ContourDynamics.DeviceContourState

            prob_periodic_euler = ContourProblem(EulerKernel(), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
            @test prob_periodic_euler.dev === GPU()
            @test prob_periodic_euler.device_state isa ContourDynamics.DeviceContourState

            prob = ContourProblem(SQGKernel(0.01), UnboundedDomain(), [c]; dev=GPU())
            @test prob.dev === GPU()
            @test prob.device_state isa ContourDynamics.DeviceContourState
            prob_periodic_sqg = ContourProblem(SQGKernel(0.01), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
            @test prob_periodic_sqg.dev === GPU()
            @test prob_periodic_sqg.device_state isa ContourDynamics.DeviceContourState
        else
            @test_throws ErrorException ContourProblem(QGKernel(1.0), UnboundedDomain(), [c]; dev=GPU())
            @test_throws ErrorException ContourProblem(QGKernel(1.0), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
            @test_throws ErrorException ContourProblem(EulerKernel(), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
            @test_throws ErrorException ContourProblem(SQGKernel(0.01), UnboundedDomain(), [c]; dev=GPU())
            @test_throws ErrorException ContourProblem(SQGKernel(0.01), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
        end
    end

    @testset "GPU() accepts multi-layer QG problems" begin
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)
        c = circular_patch(0.5, 32, 1.0)
        if _TEST_CUDA_LOADED[]
            prob = MultiLayerContourProblem(MultiLayerQGKernel(Ld, coupling), UnboundedDomain(),
                                            ([c], PVContour{Float64}[]); dev=GPU())
            @test prob.dev === GPU()
            @test prob.device_state isa Tuple
            @test all(s -> s isa ContourDynamics.DeviceContourState, prob.device_state)
        else
            @test_throws ErrorException MultiLayerContourProblem(MultiLayerQGKernel(Ld, coupling), UnboundedDomain(),
                                                                 ([c], PVContour{Float64}[]); dev=GPU())
        end
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

    @testset "DeviceContourState packs and materializes flat topology" begin
        c1 = circular_patch(0.5, 8, 1.25)
        c2_nodes = [SVector(0.1, 0.2), SVector(0.7, 0.2),
                    SVector(0.7, 0.8), SVector(0.1, 0.8)]
        c2 = PVContour(c2_nodes, -0.5, SVector(1.0, 0.0),
                       Bool[true, false, true, false])
        source = [c1, c2]
        state = ContourDynamics.DeviceContourState(source, CPU())

        expected_x = Float64[p[1] for c in source for p in c.nodes]
        expected_y = Float64[p[2] for c in source for p in c.nodes]
        @test to_cpu(state.x) == expected_x
        @test to_cpu(state.y) == expected_y
        @test to_cpu(state.pv) == [c1.pv, c2.pv]
        @test to_cpu(state.wrapx) == [c1.wrap[1], c2.wrap[1]]
        @test to_cpu(state.wrapy) == [c1.wrap[2], c2.wrap[2]]
        @test to_cpu(state.offsets) == [1, nnodes(c1) + 1]
        @test to_cpu(state.lengths) == [nnodes(c1), nnodes(c2)]
        @test to_cpu(state.contour_of_node) == vcat(fill(1, nnodes(c1)), fill(2, nnodes(c2)))
        @test to_cpu(state.local_index) == vcat(collect(1:nnodes(c1)), collect(1:nnodes(c2)))
        @test to_cpu(state.corners) == UInt8[c.corners[i] ? 1 : 0 for c in source for i in 1:nnodes(c)]

        materialized = ContourDynamics.materialize_contours(state)
        @test length(materialized) == length(source)
        @test all(zip(materialized, source)) do (actual, expected)
            actual.nodes == expected.nodes &&
                actual.pv == expected.pv &&
                actual.wrap == expected.wrap &&
                actual.corners == expected.corners
        end
    end

    @testset "GPU contour access requires explicit materialization" begin
        c = circular_patch(0.5, 12, 1.0)
        if _TEST_CUDA_LOADED[]
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [deepcopy(c)]; dev=GPU())
            @test prob.device_state isa ContourDynamics.DeviceContourState
            @test !(prob.device_state.x isa Vector)
            @test_throws ErrorException contours(prob)

            before = materialize_contours(prob)
            prob.contours[1].nodes[1] = SVector(99.0, 99.0)
            after = materialize_contours(prob)
            @test after[1].nodes[1] == before[1].nodes[1]
            @test after[1].nodes[1] == c.nodes[1]
        else
            @test_throws ErrorException ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=GPU())
        end
    end

    @testset "DeviceContourState builds velocity segments on backend" begin
        c1 = circular_patch(0.5, 12, 1.0)
        c2 = PVContour([SVector(0.1, 0.0), SVector(0.5, 0.2),
                        SVector(0.4, 0.8), SVector(0.0, 0.7)],
                       -0.25, SVector(1.0, 0.0),
                       Bool[true, false, false, true])
        contours_in = [c1, c2]
        state = DeviceContourState(contours_in, CPU())
        seg = ContourDynamics._state_segment_data(state, CPU())
        packed = ContourDynamics.pack_segments(ContourProblem(EulerKernel(), UnboundedDomain(), contours_in), CPU())

        for name in (:ax, :ay, :bx, :by, :pv, :ka, :kb)
            @test to_cpu(getproperty(seg, name)) ≈ to_cpu(getproperty(packed, name))
        end
    end

    @testset "DeviceContourState velocity ignores stale host contour shadow" begin
        c1 = circular_patch(0.45, 16, 1.0)
        c2 = PVContour([p + SVector(0.9, -0.2) for p in circular_patch(0.2, 10, -0.5).nodes], -0.5)
        original = [c1, c2]
        ref_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(original); dev=CPU())
        ref = zeros(SVector{2,Float64}, total_nodes(ref_prob))
        ContourDynamics._direct_velocity!(ref, ref_prob)

        state_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(original); dev=CPU())
        state = DeviceContourState(deepcopy(original), CPU())
        state_prob.contours[1].nodes[1] = SVector(99.0, 99.0)

        vel = zeros(SVector{2,Float64}, length(ref))
        ContourDynamics._ka_velocity_from_state!(vel, state, state_prob.kernel,
                                                 state_prob.domain, CPU())
        @test all(eachindex(ref)) do i
            isapprox(vel[i][1], ref[i][1]; rtol=1e-10, atol=1e-10) &&
                isapprox(vel[i][2], ref[i][2]; rtol=1e-10, atol=1e-10)
        end
    end

    @testset "DeviceContourState point velocity uses authoritative coordinates" begin
        contours_in = [circular_patch(0.5, 24, 1.0)]
        state = DeviceContourState(deepcopy(contours_in), CPU())
        state.x .+= 0.75
        x = SVector(0.1, 0.2)

        current = ContourProblem(EulerKernel(), UnboundedDomain(),
                                 materialize_contours(state))
        stale = ContourProblem(EulerKernel(), UnboundedDomain(), contours_in)
        expected = velocity(current, x)
        actual = ContourDynamics._velocity_at_state(state, EulerKernel(),
                                                    UnboundedDomain(), x)

        @test actual ≈ expected rtol=1e-12 atol=1e-12
        @test !isapprox(actual, velocity(stale, x); rtol=1e-8, atol=1e-8)

        F = 0.5
        kernel = MultiLayerQGKernel(SVector(1 / sqrt(2F)),
                                    SMatrix{2,2,Float64}(-F, F, F, -F))
        layers = ([circular_patch(0.35, 16, 1.0)],
                  [circular_patch(0.25, 12, -0.5; cx=0.4)])
        states = ntuple(i -> DeviceContourState(deepcopy(layers[i]), CPU()), 2)
        states[1].x .-= 0.6
        current_layers = ntuple(i -> materialize_contours(states[i]), 2)
        current_multi = MultiLayerContourProblem(kernel, UnboundedDomain(), current_layers)
        expected_multi = velocity(current_multi, x)
        actual_multi = ContourDynamics._velocity_at_state(states, kernel,
                                                          UnboundedDomain(), x)

        @test all(isapprox.(actual_multi, expected_multi; rtol=1e-12, atol=1e-12))
    end

    @testset "DeviceContourState RK4 step matches CPU contour RK4" begin
        contours_in = [
            circular_patch(0.35, 18, 1.0),
            PVContour([p + SVector(0.7, -0.25) for p in circular_patch(0.18, 12, -0.4).nodes], -0.4),
        ]
        cpu_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours_in); dev=CPU())
        state = DeviceContourState(deepcopy(contours_in), CPU())
        dt = 0.002
        cpu_stepper = RK4Stepper(dt, total_nodes(cpu_prob); dev=CPU())
        state_stepper = RK4Stepper(dt, total_nodes(cpu_prob); dev=CPU())

        timestep!(cpu_prob, cpu_stepper)
        ContourDynamics._rk4_state_step!(state, EulerKernel(), UnboundedDomain(),
                                         state_stepper, CPU())
        state_contours = materialize_contours(state)

        @test all(zip(state_contours, cpu_prob.contours)) do (actual, expected)
            all(isapprox.(actual.nodes, expected.nodes; rtol=1e-10, atol=1e-10))
        end
    end

    @testset "DeviceContourState leapfrog steps match CPU contour leapfrog" begin
        contours_in = [circular_patch(0.35, 18, 1.0)]
        cpu_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours_in); dev=CPU())
        state = DeviceContourState(deepcopy(contours_in), CPU())
        dt = 0.002
        cpu_stepper = LeapfrogStepper(dt, total_nodes(cpu_prob); dev=CPU(), ra_coeff=0.05)
        state_stepper = LeapfrogStepper(dt, total_nodes(cpu_prob); dev=CPU(), ra_coeff=0.05)

        for _ in 1:2
            timestep!(cpu_prob, cpu_stepper)
            ContourDynamics._leapfrog_state_step!(state, EulerKernel(), UnboundedDomain(),
                                                  state_stepper, CPU())
        end
        state_contours = materialize_contours(state)

        @test state_stepper.initialized
        @test all(zip(state_contours, cpu_prob.contours)) do (actual, expected)
            all(isapprox.(actual.nodes, expected.nodes; rtol=1e-10, atol=1e-10))
        end
    end

    @testset "DeviceContourState periodic wrapping matches CPU contour wrapping" begin
        domain = PeriodicDomain(1.0, 1.0)
        shifted = PVContour([p + SVector(2.2, -2.1) for p in circular_patch(0.2, 12, 1.0).nodes], 1.0)
        spanning = PVContour([SVector(-1.0, -0.5), SVector(1.0, -0.5),
                              SVector(1.0, 0.5), SVector(-1.0, 0.5)],
                             -0.25, SVector(2.0, 0.0))
        cpu_prob = ContourProblem(EulerKernel(), domain, deepcopy([shifted, spanning]); dev=CPU())
        state = DeviceContourState(deepcopy([shifted, spanning]), CPU())

        wrap_nodes!(cpu_prob)
        ContourDynamics._wrap_state_nodes!(state, domain, CPU())
        state_contours = materialize_contours(state)

        @test all(zip(state_contours, cpu_prob.contours)) do (actual, expected)
            all(isapprox.(actual.nodes, expected.nodes; rtol=1e-12, atol=1e-12))
        end
    end

    @testset "Stepper buffers honor selected device" begin
        rk_cpu = RK4Stepper(0.01, 8; dev=CPU())
        lf_cpu = LeapfrogStepper(0.01, 8; dev=CPU())
        @test rk_cpu.k1 isa Vector{SVector{2,Float64}}
        @test lf_cpu.vel_buf isa Vector{SVector{2,Float64}}

        if _TEST_CUDA_LOADED[]
            rk_gpu = RK4Stepper(0.01, 8; dev=GPU())
            lf_gpu = LeapfrogStepper(0.01, 8; dev=GPU())
            @test !(rk_gpu.k1 isa Vector)
            @test !(rk_gpu.nodes_buf isa Vector)
            @test !(lf_gpu.vel_buf isa Vector)
            @test !(lf_gpu.nodes_buf isa Vector)
        else
            @test_throws ErrorException RK4Stepper(0.01, 8; dev=GPU())
            @test_throws ErrorException LeapfrogStepper(0.01, 8; dev=GPU())
        end
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

    @testset "CPU velocity! agrees with the KA evaluator" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=CPU())
        N = total_nodes(prob)

        # CPU velocity! uses the allocation-free direct evaluator; it must still
        # match the KA workspace path (reserved for GPU) to machine precision.
        vel_ka = zeros(SVector{2,Float64}, N)
        vel = similar(vel_ka)
        ContourDynamics._ka_velocity!(vel_ka, prob, prob.dev)
        velocity!(vel, prob)

        for i in 1:N
            @test isapprox(vel[i][1], vel_ka[i][1]; atol=1e-12)
            @test isapprox(vel[i][2], vel_ka[i][2]; atol=1e-12)
        end
    end

    @testset "Device surgery filament flags match Dritschel cleanup predicates" begin
        tiny = PVContour([
            SVector(0.0, 0.0), SVector(0.001, 0.0), SVector(0.0005, 0.001)
        ], 1.0)
        normal = circular_patch(1.0, 64, 1.0)
        corner_nodes = SVector{2,Float64}[
            SVector(0.0, 0.0),
            SVector(1.0, 0.0),
            SVector(1.0, 1.0),
            SVector(0.0, 1.0),
        ]
        corner_flags = falses(length(corner_nodes))
        corner_flags[1] = true
        too_few_corner = PVContour(corner_nodes, 1.0, zero(SVector{2,Float64}), corner_flags)
        params = SurgeryParams(0.001, 0.005, 0.1, 1e-4, 10)

        flags = ContourDynamics._device_filament_flags([normal, tiny, too_few_corner], params, CPU())

        @test flags == [false, true, true]

        contours = [normal, tiny, too_few_corner]
        ContourDynamics._device_remove_filaments!(contours, params, CPU())
        @test length(contours) == 1
        @test contours[1].nodes == normal.nodes
    end

    @testset "DeviceContourState filament removal rewrites authoritative state" begin
        tiny = PVContour([
            SVector(0.0, 0.0), SVector(0.001, 0.0), SVector(0.0005, 0.001)
        ], 1.0)
        normal = circular_patch(1.0, 64, 1.0)
        params = SurgeryParams(0.001, 0.005, 0.1, 1e-4, 10)
        state = DeviceContourState([normal, tiny], CPU())

        ContourDynamics._device_remove_filaments!(state, params, CPU())
        actual = materialize_contours(state)

        @test length(actual) == 1
        @test actual[1].nodes == normal.nodes
        @test actual[1].pv == normal.pv
    end

    @testset "DeviceContourState corner promotion matches CPU corner rules" begin
        nodes = SVector{2,Float64}[
            SVector(0.0, 0.0),
            SVector(1.0, 0.0),
            SVector(2.0, 0.2),
            SVector(0.3, 0.8),
            SVector(1.1, -0.15),
        ]
        corners = falses(length(nodes))
        corners[2] = true
        contours_in = [PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)]
        expected = deepcopy(contours_in)
        ContourDynamics._demote_obtuse_corners!(expected)
        ContourDynamics._promote_high_curvature_corners!(expected, 1.0)
        state = DeviceContourState(deepcopy(contours_in), CPU())

        ContourDynamics._demote_obtuse_corners!(state, CPU())
        ContourDynamics._promote_high_curvature_corners!(state, 1.0, CPU())
        actual = materialize_contours(state)

        @test actual[1].corners == expected[1].corners
    end

    @testset "Device closed remesh matches CPU weighted remesh" begin
        contours = [
            elliptical_patch(1.0, 0.6, 40, 1.0),
            circular_patch(0.35, 24, 0.5),
        ]
        params = SurgeryParams(0.005, 0.04, 0.16, 1e-6, 10)

        density_sources = copy(contours)
        cpu_remeshed = [
            remesh(c, params; _density_sources=density_sources)
            for c in contours
        ]
        device_remeshed = ContourDynamics._device_remesh_contours(contours, params, CPU())

        @test device_remeshed !== nothing
        @test nnodes.(device_remeshed) == nnodes.(cpu_remeshed)
        @test getproperty.(device_remeshed, :pv) == getproperty.(cpu_remeshed, :pv)
        @test getproperty.(device_remeshed, :wrap) == getproperty.(cpu_remeshed, :wrap)
        @test all(zip(device_remeshed, cpu_remeshed)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-10, rtol=1e-10))
        end
        @test all(c -> isempty(corner_indices(c)), device_remeshed)

        cornered = deepcopy(contours[1])
        cornered.corners[1] = true
        cornered.corners[21] = true
        cpu_cornered = remesh(cornered, params; _density_sources=[cornered])
        device_cornered = only(ContourDynamics._device_remesh_contours([cornered], params, CPU()))
        @test nnodes(device_cornered) == nnodes(cpu_cornered)
        @test corner_indices(device_cornered) == corner_indices(cpu_cornered)
        @test all(isapprox.(device_cornered.nodes, cpu_cornered.nodes; atol=1e-10, rtol=1e-10))
    end

    @testset "DeviceContourState remesh rewrites authoritative state" begin
        contours_in = [
            elliptical_patch(0.9, 0.45, 30, 1.0),
            PVContour([p + SVector(1.2, -0.3) for p in circular_patch(0.25, 18, -0.5).nodes], -0.5),
        ]
        params = SurgeryParams(0.005, 0.04, 0.16, 1e-6, 10)
        expected = ContourDynamics._device_remesh_contours(deepcopy(contours_in), params, CPU())
        state = DeviceContourState(deepcopy(contours_in), CPU())

        @test expected !== nothing
        ContourDynamics._device_remesh_state!(state, params, CPU())
        actual = materialize_contours(state)

        @test nnodes.(actual) == nnodes.(expected)
        @test getproperty.(actual, :pv) == getproperty.(expected, :pv)
        @test all(zip(actual, expected)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12)) &&
                dev_c.corners == cpu_c.corners
        end
    end

    @testset "DeviceContourState rewrite preserves unchanged contours" begin
        function state_rectangle_patch(xmin, xmax, ymin, ymax, nside, pv)
            nodes = SVector{2,Float64}[]
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmin + (xmax - xmin) * k / nside, ymin))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmax, ymin + (ymax - ymin) * k / nside))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmax - (xmax - xmin) * k / nside, ymax))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmin, ymax - (ymax - ymin) * k / nside))
            end
            return PVContour(nodes, pv)
        end

        δ = 0.02
        contours_in = [
            state_rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            state_rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
            PVContour([p + SVector(4.0, 0.0) for p in circular_patch(0.2, 16, 0.5).nodes], 0.5),
        ]
        candidates = ContourDynamics._device_admissible_close_segment_buffer(
            contours_in, δ, UnboundedDomain(), CPU())
        selected = ContourDynamics._device_select_reconnection_pair_buffer(
            contours_in, candidates, CPU())
        expected = ContourDynamics._device_rewrite_contours(contours_in, selected, CPU())
        state = DeviceContourState(deepcopy(contours_in), CPU())

        ContourDynamics._device_rewrite_state!(state, selected, CPU())
        actual = materialize_contours(state)

        @test nnodes.(actual) == nnodes.(expected)
        @test getproperty.(actual, :pv) == getproperty.(expected, :pv)
        @test actual[end].nodes == contours_in[end].nodes
        @test all(zip(actual, expected)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12)) &&
                dev_c.corners == cpu_c.corners
        end
    end

    @testset "DeviceContourState reconnect mutates state not host contours" begin
        function reconnect_rectangle_patch(xmin, xmax, ymin, ymax, nside, pv)
            nodes = SVector{2,Float64}[]
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmin + (xmax - xmin) * k / nside, ymin))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmax, ymin + (ymax - ymin) * k / nside))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmax - (xmax - xmin) * k / nside, ymax))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmin, ymax - (ymax - ymin) * k / nside))
            end
            return PVContour(nodes, pv)
        end

        δ = 0.02
        contours_in = [
            reconnect_rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            reconnect_rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
        ]
        candidates = ContourDynamics._device_admissible_close_segment_buffer(
            contours_in, δ, UnboundedDomain(), CPU())
        expected = deepcopy(contours_in)
        @test ContourDynamics._device_reconnect!(expected, candidates, CPU())
        state_shadow = deepcopy(contours_in)
        state = DeviceContourState(state_shadow, CPU())

        @test ContourDynamics._device_reconnect!(state, candidates, CPU())
        actual = materialize_contours(state)

        @test all(zip(state_shadow, contours_in)) do (shadow_c, source_c)
            shadow_c.nodes == source_c.nodes &&
                shadow_c.pv == source_c.pv &&
                shadow_c.wrap == source_c.wrap &&
                shadow_c.corners == source_c.corners
        end
        @test nnodes.(actual) == nnodes.(expected)
        @test all(zip(actual, expected)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12)) &&
                dev_c.corners == cpu_c.corners
        end
    end

    @testset "Device close-pair candidates match CPU surgery on simple merge" begin
        function rectangle_patch(xmin, xmax, ymin, ymax, nside, pv)
            nodes = SVector{2,Float64}[]
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmin + (xmax - xmin) * k / nside, ymin))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmax, ymin + (ymax - ymin) * k / nside))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmax - (xmax - xmin) * k / nside, ymax))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmin, ymax - (ymax - ymin) * k / nside))
            end
            return PVContour(nodes, pv)
        end

        δ = 0.02
        contours_same = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
        ]
        idx = ContourDynamics.build_spatial_index(contours_same, δ)
        cpu_pairs = Set(ContourDynamics.find_close_segments(contours_same, idx, δ))
        buffer = ContourDynamics._device_close_pair_candidate_buffer(contours_same, δ, CPU())
        buffered_pairs = Set(ContourDynamics._unpack_close_pair_candidates(buffer))
        dev_pairs = Set(ContourDynamics._device_close_pair_candidates(contours_same, δ, CPU()))
        admissible_buffer = ContourDynamics._device_admissible_close_segment_buffer(
            contours_same, δ, UnboundedDomain(), CPU())
        admissible_pairs = Set(ContourDynamics._unpack_close_pair_candidates(admissible_buffer))

        @test !isempty(cpu_pairs)
        @test cpu_pairs == dev_pairs
        @test buffered_pairs == dev_pairs
        @test admissible_pairs == dev_pairs
        @test length(to_cpu(buffer.ci)) == length(dev_pairs)

        contours_different_pv = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 2.0),
        ]
        @test isempty(ContourDynamics._device_close_pair_candidates(contours_different_pv, δ, CPU()))
    end

    @testset "Device close-pair admissibility honors interior vorticity" begin
        δ = 0.05
        outer = circular_patch(1.0, 96, 1.0)
        inner = circular_patch(0.98, 96, 1.0)
        contours = [outer, inner]

        raw_pairs = ContourDynamics._device_close_pair_candidates(contours, δ, CPU())
        admissible = ContourDynamics._device_admissible_close_segments(
            contours, δ, UnboundedDomain(), CPU())
        admissible_buffer = ContourDynamics._device_admissible_close_segment_buffer(
            contours, δ, UnboundedDomain(), CPU())

        @test !isempty(raw_pairs)
        @test isempty(admissible)
        @test isempty(ContourDynamics._unpack_close_pair_candidates(admissible_buffer))
    end

    @testset "Device reconnection planner matches CPU independent-pair selection" begin
        function rectangle_patch(xmin, xmax, ymin, ymax, nside, pv)
            nodes = SVector{2,Float64}[]
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmin + (xmax - xmin) * k / nside, ymin))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmax, ymin + (ymax - ymin) * k / nside))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmax - (xmax - xmin) * k / nside, ymax))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmin, ymax - (ymax - ymin) * k / nside))
            end
            return PVContour(nodes, pv)
        end

        δ = 0.02
        contours = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(3.0, 4.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(4.012, 5.0, 0.0, 1.0, 6, 1.0),
        ]
        idx = ContourDynamics.build_spatial_index(contours, δ)
        close_pairs = ContourDynamics.find_close_segments(contours, idx, δ)

        cpu_selected = ContourDynamics._select_reconnection_pairs(contours, close_pairs)
        device_selected = ContourDynamics._device_select_reconnection_pairs(contours, close_pairs, CPU())
        candidate_buffer = ContourDynamics._device_close_pair_candidate_buffer(contours, δ, CPU())
        buffer_pairs = ContourDynamics._unpack_close_pair_candidates(candidate_buffer)
        buffer_selected = ContourDynamics._device_select_reconnection_pairs(contours, candidate_buffer, CPU())
        selected_buffer = ContourDynamics._device_select_reconnection_pair_buffer(
            contours, candidate_buffer, CPU())
        selected_buffer_pairs = ContourDynamics._unpack_close_pair_candidates(selected_buffer)
        plan = ContourDynamics._device_reconnection_plan(contours, close_pairs, CPU())
        buffer_plan = ContourDynamics._device_reconnection_plan(contours, candidate_buffer, CPU())
        rewrite = ContourDynamics._device_topology_rewrite_plan(contours, device_selected, CPU())
        buffer_rewrite = ContourDynamics._device_topology_rewrite_plan(contours, selected_buffer, CPU())
        _assert_device_layout_matches_host(contours, rewrite)
        materialized = ContourDynamics._unpack_rewrite_outputs(
            ContourDynamics._device_materialize_rewrite_outputs(contours, device_selected, CPU()))
        expected_merge_lengths = [
            nnodes(contours[pair[1]]) + nnodes(contours[pair[3]]) + 1
            for pair in device_selected
        ]

        @test !isempty(close_pairs)
        @test Set(buffer_pairs) == Set(close_pairs)
        @test device_selected == cpu_selected
        @test Set(buffer_selected) == Set(device_selected)
        @test Set(selected_buffer_pairs) == Set(device_selected)
        @test Set(to_cpu(plan.op)) == Set(UInt8[2])
        @test Set(to_cpu(buffer_plan.op)) == Set(UInt8[2])
        @test all(isfinite, to_cpu(plan.distance2))
        @test all(isfinite, to_cpu(buffer_plan.distance2))
        @test all(to_cpu(rewrite.op) .== UInt8(2))
        @test to_cpu(buffer_rewrite.op) == to_cpu(rewrite.op)
        @test to_cpu(buffer_rewrite.out_len1) == to_cpu(rewrite.out_len1)
        @test all(to_cpu(rewrite.valid) .== UInt8(1))
        @test to_cpu(rewrite.out_len1) == expected_merge_lengths
        @test all(isfinite, to_cpu(rewrite.stitch_x))
        @test all(isfinite, to_cpu(rewrite.stitch_y))
        @test sort(nnodes.(materialized)) == sort(expected_merge_lengths)
        @test all(c -> length(corner_indices(c)) >= 2, materialized)

        cpu_contours = deepcopy(contours)
        ContourDynamics.reconnect!(cpu_contours, device_selected)
        once_contours = deepcopy(contours)
        @test ContourDynamics._device_reconnect_once!(once_contours, δ, UnboundedDomain(), CPU())
        full_materialized = ContourDynamics._device_rewrite_contours(contours, device_selected, CPU())
        buffer_materialized = ContourDynamics._device_rewrite_contours(contours, selected_buffer, CPU())
        @test nnodes.(full_materialized) == nnodes.(cpu_contours)
        @test nnodes.(buffer_materialized) == nnodes.(cpu_contours)
        @test nnodes.(once_contours) == nnodes.(cpu_contours)
        @test getproperty.(full_materialized, :pv) == getproperty.(cpu_contours, :pv)
        @test all(zip(full_materialized, cpu_contours)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12))
        end
        @test all(zip(buffer_materialized, cpu_contours)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12))
        end
        @test all(zip(once_contours, cpu_contours)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12))
        end
    end

    @testset "Device topology rewrite matches CPU split output geometry" begin
        gap = 0.002
        δ = 0.01
        N_half = 30
        nodes = SVector{2,Float64}[]
        for k in 0:N_half
            θ = -π/2 + π * k / N_half
            push!(nodes, SVector(1.0 + cos(θ), sin(θ)))
        end
        for k in 1:5
            x = 1.0 - k * 2.0 / 6
            push!(nodes, SVector(x, gap))
        end
        for k in 0:N_half
            θ = π/2 + π * k / N_half
            push!(nodes, SVector(-1.0 + cos(θ), sin(θ)))
        end
        for k in 1:5
            x = -1.0 + k * 2.0 / 6
            push!(nodes, SVector(x, -gap))
        end

        contours = [PVContour(nodes, 1.0)]
        idx = ContourDynamics.build_spatial_index(contours, δ)
        close_pairs = ContourDynamics.find_close_segments(contours, idx, δ)
        selected = ContourDynamics._select_reconnection_pairs(contours, close_pairs)
        rewrite = ContourDynamics._device_topology_rewrite_plan(contours, selected, CPU())
        _assert_device_layout_matches_host(contours, rewrite)
        materialized = ContourDynamics._unpack_rewrite_outputs(
            ContourDynamics._device_materialize_rewrite_outputs(contours, selected, CPU()))

        cpu_contours = deepcopy(contours)
        ContourDynamics.reconnect!(cpu_contours, selected)
        cpu_lengths = sort(nnodes.(cpu_contours))
        device_lengths = sort([to_cpu(rewrite.out_len1)[1], to_cpu(rewrite.out_len2)[1]])
        materialized_lengths = sort(nnodes.(materialized))

        @test length(selected) == 1
        @test to_cpu(rewrite.op) == UInt8[1]
        @test to_cpu(rewrite.valid) == UInt8[1]
        @test to_cpu(rewrite.out_count) == [2]
        @test device_lengths == cpu_lengths
        @test materialized_lengths == cpu_lengths
        @test all(zip(materialized, cpu_contours)) do (dev_c, cpu_c)
            isapprox(vortex_area(dev_c), vortex_area(cpu_c); atol=1e-12, rtol=1e-12)
        end
        @test all(c -> all(p -> all(isfinite, p), c.nodes), materialized)
        @test all(c -> !isempty(corner_indices(c)), materialized)

        full_materialized = ContourDynamics._device_rewrite_contours(contours, selected, CPU())
        @test nnodes.(full_materialized) == nnodes.(cpu_contours)
        @test all(zip(full_materialized, cpu_contours)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12))
        end
    end

    @testset "Device full topology rewrite preserves unchanged contours" begin
        function rectangle_patch(xmin, xmax, ymin, ymax, nside, pv)
            nodes = SVector{2,Float64}[]
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmin + (xmax - xmin) * k / nside, ymin))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmax, ymin + (ymax - ymin) * k / nside))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmax - (xmax - xmin) * k / nside, ymax))
            end
            for k in 0:(nside - 1)
                push!(nodes, SVector(xmin, ymax - (ymax - ymin) * k / nside))
            end
            return PVContour(nodes, pv)
        end

        δ = 0.02
        contours = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
            circular_patch(0.2, 16, 0.5),
        ]
        contours[3] = PVContour([p + SVector(4.0, 0.0) for p in contours[3].nodes],
                                contours[3].pv, contours[3].wrap, contours[3].corners)

        idx = ContourDynamics.build_spatial_index(contours, δ)
        close_pairs = ContourDynamics.find_close_segments(contours, idx, δ)
        selected = ContourDynamics._select_reconnection_pairs(contours, close_pairs)
        rewrite = ContourDynamics._device_topology_rewrite_plan(contours, selected, CPU())
        _assert_device_layout_matches_host(contours, rewrite)

        cpu_contours = deepcopy(contours)
        ContourDynamics.reconnect!(cpu_contours, selected)
        full_materialized = ContourDynamics._device_rewrite_contours(contours, selected, CPU())

        @test length(full_materialized) == length(cpu_contours) == 2
        @test nnodes.(full_materialized) == nnodes.(cpu_contours)
        @test full_materialized[2].nodes == contours[3].nodes
        @test all(zip(full_materialized, cpu_contours)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12))
        end
    end

    @testset "KA single-layer energy matches scalar diagnostics on CPU backend" begin
        c = circular_patch(0.25, 8, 1.0)
        probs = (
            ContourProblem(EulerKernel(), UnboundedDomain(), [c]),
            ContourProblem(QGKernel(1.0), UnboundedDomain(), [c]),
            ContourProblem(SQGKernel(0.02), UnboundedDomain(), [c]),
            ContourProblem(EulerKernel(), PeriodicDomain(2.0, 2.0), [c]),
            ContourProblem(QGKernel(1.0), PeriodicDomain(2.0, 2.0), [c]),
            ContourProblem(SQGKernel(0.02), PeriodicDomain(2.0, 2.0), [c]),
        )

        for prob in probs
            E_scalar = energy(prob)
            E_ka = ContourDynamics._ka_energy(prob, CPU())
            @test isfinite(E_ka)
            @test E_ka ≈ E_scalar rtol=1e-7 atol=1e-10
        end
    end

    @testset "DeviceContourState geometry diagnostics match CPU diagnostics" begin
        closed1 = circular_patch(0.35, 24, 1.0)
        closed2 = PVContour([p + SVector(0.8, -0.25) for p in circular_patch(0.18, 16, -0.4).nodes], -0.4)
        spanning = PVContour([SVector(-1.0, -0.2), SVector(1.0, -0.2),
                              SVector(1.0, 0.2), SVector(-1.0, 0.2)],
                             0.25, SVector(2.0, 0.0))
        contours_in = [closed1, closed2, spanning]
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours_in); dev=CPU())
        state = DeviceContourState(deepcopy(contours_in), CPU())

        @test to_cpu(ContourDynamics._state_vortex_area(state, CPU())) ≈ vortex_area(prob)
        @test ContourDynamics._state_circulation(state, CPU()) ≈ circulation(prob)
        @test ContourDynamics._state_enstrophy(state, CPU()) ≈ enstrophy(prob)
        @test ContourDynamics._state_angular_momentum(state, CPU()) ≈ angular_momentum(prob)
    end

    @testset "DeviceContourState energy diagnostics match KA CPU diagnostics" begin
        contours_in = [
            circular_patch(0.35, 20, 1.0),
            PVContour([p + SVector(0.8, -0.25) for p in circular_patch(0.18, 14, -0.4).nodes], -0.4),
        ]
        domains = (UnboundedDomain(), PeriodicDomain(2.0, 2.0))
        kernels = (EulerKernel(), QGKernel(1.25), SQGKernel(0.02))
        for domain in domains, kernel in kernels
            clear_ewald_cache!()
            prob = ContourProblem(kernel, domain, deepcopy(contours_in); dev=CPU())
            state = DeviceContourState(deepcopy(contours_in), CPU())
            @test ContourDynamics._ka_energy_from_state(state, kernel, domain, CPU()) ≈
                  ContourDynamics._ka_energy(prob, CPU()) rtol=1e-7 atol=1e-10
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
        let _states = (ContourDynamics.DeviceContourState(prob.layers[1], CPU()),
                       ContourDynamics.DeviceContourState(prob.layers[2], CPU()))
            _flat = zeros(SVector{2,Float64}, nnodes(c1) + nnodes(c2))
            ContourDynamics._ka_multilayer_velocity_from_states!(_flat, _states, prob.kernel, prob.domain, CPU())
            vel_ka[1] .= _flat[1:nnodes(c1)]
            vel_ka[2] .= _flat[nnodes(c1)+1:end]
        end

        for i in eachindex(vel_ref[1])
            @test isapprox(vel_ka[1][i][1], vel_ref[1][i][1]; atol=1e-8, rtol=1e-8)
            @test isapprox(vel_ka[1][i][2], vel_ref[1][i][2]; atol=1e-8, rtol=1e-8)
        end
        for i in eachindex(vel_ref[2])
            @test isapprox(vel_ka[2][i][1], vel_ref[2][i][1]; atol=1e-8, rtol=1e-8)
            @test isapprox(vel_ka[2][i][2], vel_ref[2][i][2]; atol=1e-8, rtol=1e-8)
        end
    end

end

@testset "Multi-layer device state" begin
    @testset "Scaled segment pack multiplies PV only" begin
        c = circular_patch(0.4, 12, 2.0)
        state = ContourDynamics.DeviceContourState([c], CPU())
        seg1 = ContourDynamics._state_segment_data(state, CPU())
        n = ContourDynamics._device_state_nnodes(state)

        ax = zeros(n); ay = zeros(n); bx = zeros(n); by = zeros(n)
        pv = zeros(n); ka = zeros(n); kb = zeros(n)
        ContourDynamics.@_ka_launch CPU() n ContourDynamics._state_segment_data_kernel!(
            ax, ay, bx, by, pv, ka, kb,
            state.x, state.y, state.pv, state.wrapx, state.wrapy,
            state.offsets, state.lengths, state.corners,
            state.contour_of_node, state.local_index, 0.5, n)

        @test pv ≈ 0.5 .* seg1.pv
        @test ax == seg1.ax && ay == seg1.ay && bx == seg1.bx && by == seg1.by
        @test ka == seg1.ka && kb == seg1.kb
    end

    @testset "Layer state ranges partition flat index space" begin
        c1 = circular_patch(0.5, 24, 1.0)
        c2 = circular_patch(0.3, 16, -0.7)
        states = (ContourDynamics.DeviceContourState([c1], CPU()),
                  ContourDynamics.DeviceContourState([c2], CPU()))
        ranges = ContourDynamics._layer_state_ranges(states)
        @test ranges isa NTuple{2, UnitRange{Int}}  # tuple, not Vector: allocation-free per-step path
        @test ranges == (1:24, 25:40)

        # A layer with 0 nodes produces an empty range and the next layer starts at the same cursor.
        states3 = (ContourDynamics.DeviceContourState(PVContour{Float64}[], CPU()),
                   ContourDynamics.DeviceContourState([circular_patch(0.2, 8, 1.0)], CPU()))
        ranges3 = ContourDynamics._layer_state_ranges(states3)
        @test ranges3[1] == 1:0
        @test ranges3[2] == 1:8
    end

    @testset "Multi-layer output validation uses authoritative state sizes" begin
        layers = (
            [circular_patch(0.3, 8, 1.0)],
            [circular_patch(0.2, 12, -0.5; cx=0.4)],
        )
        states = ntuple(i -> DeviceContourState(layers[i], CPU()), 2)
        vel = (zeros(SVector{2,Float64}, 8), zeros(SVector{2,Float64}, 12))

        @test ContourDynamics._validate_multilayer_state_velocity_buffer!(vel, states) === vel
        undersized = (zeros(SVector{2,Float64}, 7), vel[2])
        @test_throws DimensionMismatch ContourDynamics._validate_multilayer_state_velocity_buffer!(
            undersized, states)
    end

    @testset "State-based multi-layer velocity matches CPU modal velocity (unbounded)" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        ml_layers = (
            [circular_patch(0.5, 24, 1.0)],
            [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 16, -0.7).nodes], -0.7)],
        )

        cpu_prob = MultiLayerContourProblem(ml_kernel, UnboundedDomain(), deepcopy(ml_layers))
        vel_t = (zeros(SVector{2,Float64}, 24), zeros(SVector{2,Float64}, 16))
        velocity!(vel_t, cpu_prob)
        expected = vcat(vel_t[1], vel_t[2])

        states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                  ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()))
        flat = zeros(SVector{2,Float64}, 40)
        ContourDynamics._ka_multilayer_velocity_from_states!(flat, states, ml_kernel,
                                                             UnboundedDomain(), CPU())
        @test all(isapprox.(flat, expected; rtol=1e-8, atol=1e-10))
    end

    @testset "State-based multi-layer velocity matches CPU modal velocity (periodic)" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        ml_layers = (
            [circular_patch(0.5, 24, 1.0)],
            [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 16, -0.7).nodes], -0.7)],
        )
        domain = PeriodicDomain(4.0, 4.0)
        setup_ewald_cache!(domain, EulerKernel())

        cpu_prob = MultiLayerContourProblem(ml_kernel, domain, deepcopy(ml_layers))
        vel_t = (zeros(SVector{2,Float64}, 24), zeros(SVector{2,Float64}, 16))
        velocity!(vel_t, cpu_prob)
        expected = vcat(vel_t[1], vel_t[2])

        states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                  ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()))
        flat = zeros(SVector{2,Float64}, 40)
        ContourDynamics._ka_multilayer_velocity_from_states!(flat, states, ml_kernel,
                                                             domain, CPU())
        @test all(isapprox.(flat, expected; rtol=1e-8, atol=1e-10))
    end

    @testset "State-based 3-layer velocity matches CPU modal velocity (asymmetric P)" begin
        # Distinct tridiagonal couplings give an orthogonal but NON-symmetric
        # eigenvector matrix, so P != P_inv and a swapped/transposed projection
        # would fail this test (the 2-layer equal-F fixture cannot catch that).
        #
        # Coupling matrix is the tridiagonal nearest-neighbor form
        #   C = [-F1  F1   0 ]
        #       [ F1 -(F1+F2)  F2]
        #       [  0   F2  -F2]
        # with F1 != F2.  F1, F2 are chosen so that the two non-zero
        # eigenvalues are exactly -1 and -4, consistent with Ld = [1.0, 0.5].
        # (F1+F2 = 2.5, F1*F2 = 4/3 → F = (2.5 ± √(11/12)) / 2)
        let F1 = (2.5 + sqrt(11.0/12.0)) / 2,
            F2 = (2.5 - sqrt(11.0/12.0)) / 2
            ml_Ld      = SVector(1.0, 0.5)
            ml_coupling = SMatrix{3,3,Float64}(
                -F1,  F1,       0,
                 F1, -(F1+F2), F2,
                  0,  F2,      -F2)
            ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
            P = ml_kernel.eigenvectors
            @test !(P ≈ ml_kernel.eigenvectors_inv)  # fixture sanity: orientation distinguishable

            ml_layers = (
                [circular_patch(0.5, 24, 1.0)],
                [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 16, -0.7).nodes], -0.7)],
                [PVContour([p + SVector(-0.4, 0.5) for p in circular_patch(0.25, 12, 0.9).nodes], 0.9)],
            )

            cpu_prob = MultiLayerContourProblem(ml_kernel, UnboundedDomain(), deepcopy(ml_layers))
            vel_t = (zeros(SVector{2,Float64}, 24), zeros(SVector{2,Float64}, 16), zeros(SVector{2,Float64}, 12))
            velocity!(vel_t, cpu_prob)
            expected = vcat(vel_t[1], vel_t[2], vel_t[3])

            states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                      ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()),
                      ContourDynamics.DeviceContourState(deepcopy(ml_layers[3]), CPU()))
            flat = zeros(SVector{2,Float64}, 52)
            ContourDynamics._ka_multilayer_velocity_from_states!(flat, states, ml_kernel,
                                                                 UnboundedDomain(), CPU())
            @test all(isapprox.(flat, expected; rtol=1e-8, atol=1e-10))
        end
    end

    @testset "State-based multi-layer energy matches CPU modal energy" begin
        F = 0.5
        kernel = MultiLayerQGKernel(
            SVector(1.0), SMatrix{2,2,Float64}(-F, F, F, -F))
        layers = (
            [circular_patch(0.3, 16, 1.0)],
            [circular_patch(0.2, 12, -0.5; cx=0.4)],
        )
        states = ntuple(i -> DeviceContourState(layers[i], CPU()), 2)

        for domain in (UnboundedDomain(), PeriodicDomain(2.0, 2.0))
            clear_ewald_cache!()
            prob = MultiLayerContourProblem(kernel, domain, deepcopy(layers))
            actual = ContourDynamics._ka_multilayer_energy_from_states(
                states, kernel, domain, CPU())
            @test actual ≈ energy(prob) rtol=1e-7 atol=1e-10
        end

        F1 = (2.5 + sqrt(11.0 / 12.0)) / 2
        F2 = (2.5 - sqrt(11.0 / 12.0)) / 2
        kernel3 = MultiLayerQGKernel(
            SVector(1.0, 0.5),
            SMatrix{3,3,Float64}(
                -F1, F1, 0,
                F1, -(F1 + F2), F2,
                0, F2, -F2))
        layers3 = (
            [circular_patch(0.25, 10, 1.0)],
            [circular_patch(0.2, 8, -0.6; cx=0.4)],
            [circular_patch(0.15, 8, 0.8; cx=-0.3, cy=0.35)],
        )
        states3 = ntuple(i -> DeviceContourState(layers3[i], CPU()), 3)
        prob3 = MultiLayerContourProblem(kernel3, UnboundedDomain(), deepcopy(layers3))
        @test ContourDynamics._ka_multilayer_energy_from_states(
            states3, kernel3, UnboundedDomain(), CPU()) ≈ energy(prob3) rtol=1e-7 atol=1e-10
    end

    @testset "Multi-layer state RK4 matches CPU multi-layer RK4" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        ml_layers = (
            [circular_patch(0.5, 24, 1.0)],
            [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 16, -0.7).nodes], -0.7)],
        )
        dt = 0.002

        cpu_prob = MultiLayerContourProblem(ml_kernel, UnboundedDomain(), deepcopy(ml_layers))
        cpu_stepper = RK4Stepper(dt, total_nodes(cpu_prob))
        states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                  ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()))
        state_stepper = RK4Stepper(dt, 40; dev=CPU())

        for _ in 1:3
            timestep!(cpu_prob, cpu_stepper)
            ContourDynamics._rk4_multilayer_state_step!(states, ml_kernel,
                                                        UnboundedDomain(), state_stepper, CPU())
        end

        for ℓ in 1:2
            actual = materialize_contours(states[ℓ])
            expected = cpu_prob.layers[ℓ]
            @test all(zip(actual, expected)) do (a, e)
                all(isapprox.(a.nodes, e.nodes; rtol=1e-8, atol=1e-10))
            end
        end
    end

    @testset "Multi-layer state leapfrog matches CPU multi-layer leapfrog" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        ml_layers = (
            [circular_patch(0.5, 24, 1.0)],
            [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 16, -0.7).nodes], -0.7)],
        )
        dt = 0.002

        cpu_prob = MultiLayerContourProblem(ml_kernel, UnboundedDomain(), deepcopy(ml_layers))
        cpu_stepper = LeapfrogStepper(dt, total_nodes(cpu_prob); ra_coeff=0.05)
        states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                  ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()))
        state_stepper = LeapfrogStepper(dt, 40; dev=CPU(), ra_coeff=0.05)

        # 3 steps covers the bootstrap half-step plus two regular leapfrog steps.
        for _ in 1:3
            timestep!(cpu_prob, cpu_stepper)
            ContourDynamics._leapfrog_multilayer_state_step!(states, ml_kernel,
                                                             UnboundedDomain(), state_stepper, CPU())
        end

        for ℓ in 1:2
            actual = materialize_contours(states[ℓ])
            expected = cpu_prob.layers[ℓ]
            @test all(zip(actual, expected)) do (a, e)
                all(isapprox.(a.nodes, e.nodes; rtol=1e-8, atol=1e-10))
            end
        end
    end

    @testset "State-based multi-layer velocity ignores stale host contours" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        layer1 = [circular_patch(0.5, 24, 1.0)]
        layer2 = [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 16, -0.7).nodes], -0.7)]

        states = (ContourDynamics.DeviceContourState(layer1, CPU()),
                  ContourDynamics.DeviceContourState(layer2, CPU()))
        flat_before = zeros(SVector{2,Float64}, 40)
        ContourDynamics._ka_multilayer_velocity_from_states!(flat_before, states, ml_kernel,
                                                             UnboundedDomain(), CPU())

        # Mutate the host contours the states were built from.
        # DeviceContourState copies node data into its own arrays, so the state
        # is immune to host-side mutations — flat_after must equal flat_before.
        for i in eachindex(layer1[1].nodes)
            layer1[1].nodes[i] += SVector(10.0, 10.0)
        end

        flat_after = zeros(SVector{2,Float64}, 40)
        ContourDynamics._ka_multilayer_velocity_from_states!(flat_after, states, ml_kernel,
                                                             UnboundedDomain(), CPU())
        @test flat_after == flat_before
    end

    @testset "Multi-layer state periodic wrap matches CPU wrap" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        domain = PeriodicDomain(2.0, 2.0)
        # Patches whose centroids sit outside the box, so wrapping moves them.
        ml_layers = (
            [PVContour([p + SVector(2.5, 0.0) for p in circular_patch(0.3, 16, 1.0).nodes], 1.0)],
            [PVContour([p + SVector(0.0, -2.5) for p in circular_patch(0.2, 12, -0.5).nodes], -0.5)],
        )

        cpu_prob = MultiLayerContourProblem(ml_kernel, domain, deepcopy(ml_layers))
        wrap_nodes!(cpu_prob)
        # Guard against double-no-op: wrapping must actually displace both layers.
        @test abs(cpu_prob.layers[1][1].nodes[1][1] - ml_layers[1][1].nodes[1][1]) > 0.5
        @test abs(cpu_prob.layers[2][1].nodes[1][2] - ml_layers[2][1].nodes[1][2]) > 0.5

        states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                  ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()))
        for s in states
            ContourDynamics._wrap_state_nodes!(s, domain, CPU())
        end

        for ℓ in 1:2
            actual = materialize_contours(states[ℓ])
            expected = cpu_prob.layers[ℓ]
            @test all(zip(actual, expected)) do (a, e)
                all(isapprox.(a.nodes, e.nodes; rtol=1e-12, atol=1e-12))
            end
        end
    end

    @testset "Multi-layer device surgery matches CPU multi-layer surgery (unbounded)" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        # Layer 1: a healthy patch plus a tiny filament that surgery must remove.
        # Layer 2: a single healthy patch.
        tiny = PVContour([SVector(2.0, 0.0), SVector(2.0 + 1e-6, 0.0), SVector(2.0, 1e-6)], 1.0)
        ml_layers = (
            [circular_patch(0.5, 32, 1.0), tiny],
            [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 24, -0.7).nodes], -0.7)],
        )
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)

        cpu_prob = MultiLayerContourProblem(ml_kernel, UnboundedDomain(), deepcopy(ml_layers))
        surgery!(cpu_prob, params)

        states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                  ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()))
        ContourDynamics._device_multilayer_surgery!(states, params, UnboundedDomain(), CPU())

        for ℓ in 1:2
            actual = materialize_contours(states[ℓ])
            expected = cpu_prob.layers[ℓ]
            @test length(actual) == length(expected)
            @test all(zip(actual, expected)) do (a, e)
                length(a.nodes) == length(e.nodes) &&
                    all(isapprox.(a.nodes, e.nodes; rtol=1e-8, atol=1e-10))
            end
        end
    end

    @testset "Periodic multi-layer GPU surgery still errors" begin
        if _TEST_CUDA_LOADED[]
            ml_Ld = SVector(1.0)
            ml_F = 1.0 / (2 * ml_Ld[1]^2)
            ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
            prob = MultiLayerContourProblem(MultiLayerQGKernel(ml_Ld, ml_coupling),
                                            PeriodicDomain(4.0, 4.0),
                                            ([circular_patch(0.3, 16, 1.0)], PVContour{Float64}[]);
                                            dev=GPU())
            @test_throws ArgumentError surgery!(prob, SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100))
        else
            # Without CUDA the GPU periodic method is unreachable at runtime,
            # but we can verify it exists in the method table.
            @test any(methods(surgery!)) do m
                occursin("PeriodicDomain", string(m.sig)) &&
                    occursin("GPU", string(m.sig)) &&
                    occursin("MultiLayer", string(m.sig))
            end
        end
    end
end
