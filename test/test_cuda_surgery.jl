using Test
using ContourDynamics
using JLD2
using StaticArrays

@isdefined(circular_patch) || include("test_utils.jl")

function _cuda_available()
    try
        @eval using CUDA
        return CUDA.functional()
    catch
        return false
    end
end

# Precision-dependent tolerances. Float64 values are the historical ones; the
# Float32 run only checks that the device path agrees to single precision.
_cuda_tol(::Type{Float64}) = (atol=1e-8, rtol=1e-8)
_cuda_tol(::Type{Float32}) = (atol=1e-4, rtol=1e-4)
_cuda_diag_tol(::Type{Float64}) = (rtol=1e-10, atol=1e-10)
_cuda_diag_tol(::Type{Float32}) = (rtol=1e-4, atol=1e-5)
_cuda_energy_tol(::Type{Float64}) = (rtol=1e-7, atol=1e-10)
_cuda_energy_tol(::Type{Float32}) = (rtol=1e-4, atol=1e-5)

function _test_cuda_velocity_and_energy(kernel, domain; T::Type=Float64,
                                        atol=_cuda_tol(T).atol, rtol=_cuda_tol(T).rtol)
    clear_ewald_cache!()
    c1 = circular_patch(0.35, 24, 1.0; T=T)
    c2 = PVContour([p + SVector{2,T}(0.9, -0.35)
                    for p in circular_patch(0.18, 16, -0.4; T=T).nodes], T(-0.4))
    cpu_prob = ContourProblem(kernel, domain, [c1, c2]; dev=CPU())
    gpu_prob = ContourProblem(kernel, domain, deepcopy([c1, c2]); dev=GPU())
    n = total_nodes(cpu_prob)
    diag_tol = _cuda_diag_tol(T)
    energy_tol = _cuda_energy_tol(T)

    vel_ref = zeros(SVector{2,T}, n)
    ContourDynamics._direct_velocity!(vel_ref, cpu_prob)

    dev_vel = device_zeros(GPU(), SVector{2,T}, n)
    velocity!(dev_vel, gpu_prob)
    @test !(dev_vel isa Vector)
    vel_gpu = to_cpu(dev_vel)

    @test all(eachindex(vel_ref)) do i
        isapprox(vel_gpu[i][1], vel_ref[i][1]; atol, rtol) &&
            isapprox(vel_gpu[i][2], vel_ref[i][2]; atol, rtol)
    end
    point = SVector{2,T}(0.13, -0.17)
    @test velocity(gpu_prob, point) ≈ velocity(cpu_prob, point) atol=atol rtol=rtol

    # Batched probe: one pack and one launch over all targets must agree with
    # the CPU reference and with the per-point device probes.
    points = [point, SVector{2,T}(-0.42, 0.33), SVector{2,T}(0.61, 0.05),
              SVector{2,T}(-0.05, -0.58), SVector{2,T}(0.9, 0.9)]
    batched = velocity(gpu_prob, points)
    @test batched isa Vector{SVector{2,T}}
    @test length(batched) == length(points)
    @test all(isapprox.(batched, velocity(cpu_prob, points); atol, rtol))
    @test all(isapprox.(batched, [velocity(gpu_prob, x) for x in points]; atol, rtol))
    @test isempty(velocity(gpu_prob, SVector{2,T}[]))

    stale_shadow = deepcopy(gpu_prob.contours)
    gpu_prob.contours[1].nodes[1] = SVector{2,T}(99.0, 99.0)
    mixed_point = T === Float64 ? SVector{2,Float32}(point) : SVector{2,Float64}(point)
    @test velocity(gpu_prob, mixed_point) ≈ velocity(cpu_prob, mixed_point) atol=atol rtol=rtol
    @test velocity(gpu_prob, [mixed_point]) isa Vector{SVector{2,T}}
    @test all(isapprox.(velocity(gpu_prob, [mixed_point]),
                        velocity(cpu_prob, [mixed_point]); atol, rtol))
    @test energy(gpu_prob) ≈ energy(cpu_prob) rtol=energy_tol.rtol atol=energy_tol.atol
    @test circulation(gpu_prob) ≈ circulation(cpu_prob) rtol=diag_tol.rtol atol=diag_tol.atol
    @test enstrophy(gpu_prob) ≈ enstrophy(cpu_prob) rtol=diag_tol.rtol atol=diag_tol.atol
    @test angular_momentum(gpu_prob) ≈ angular_momentum(cpu_prob) rtol=diag_tol.rtol atol=diag_tol.atol
    @test vortex_area(gpu_prob) ≈ vortex_area(cpu_prob) rtol=diag_tol.rtol atol=diag_tol.atol

    fname = tempname() * ".jld2"
    try
        save_snapshot(fname, gpu_prob, 0; diagnostics=false)
        data = load_snapshot(fname, 0)
        materialized = materialize_contours(gpu_prob)
        @test data.contours[1].nodes[1] ≈ materialized[1].nodes[1]
        @test data.contours[1].nodes[1] != gpu_prob.contours[1].nodes[1]
    finally
        rm(fname; force=true)
    end

    gpu_prob.contours[1] = stale_shadow[1]
end

function _test_cuda_multilayer_paths(domain)
    clear_ewald_cache!()
    F = 0.5
    kernel = MultiLayerQGKernel(
        SVector(1.0), SMatrix{2,2,Float64}(-F, F, F, -F))
    c1 = circular_patch(0.35, 24, 1.0)
    c2 = circular_patch(0.2, 16, -0.5; cx=0.4)
    layers = ([c1], [c2])
    cpu_prob = MultiLayerContourProblem(kernel, domain, deepcopy(layers); dev=CPU())
    gpu_prob = MultiLayerContourProblem(kernel, domain, deepcopy(layers); dev=GPU())

    vel_ref = (zeros(SVector{2,Float64}, nnodes(c1)),
               zeros(SVector{2,Float64}, nnodes(c2)))
    vel_gpu = (similar(vel_ref[1]), similar(vel_ref[2]))
    velocity!(vel_ref, cpu_prob)
    velocity!(vel_gpu, gpu_prob)
    for layer in 1:2, i in eachindex(vel_ref[layer])
        @test vel_gpu[layer][i][1] ≈ vel_ref[layer][i][1] rtol=1e-8 atol=1e-8
        @test vel_gpu[layer][i][2] ≈ vel_ref[layer][i][2] rtol=1e-8 atol=1e-8
    end

    gpu_prob.layers[1][1].nodes[1] = SVector(99.0, 99.0)
    point = SVector(0.1, -0.15)
    @test all(isapprox.(velocity(gpu_prob, point), velocity(cpu_prob, point);
                        rtol=1e-8, atol=1e-8))
    mixed_point = SVector{2,Float32}(point)
    @test all(isapprox.(velocity(gpu_prob, mixed_point),
                        velocity(cpu_prob, mixed_point);
                        rtol=1e-8, atol=1e-8))
    points = [point, SVector(-0.42, 0.33), SVector(0.61, 0.05)]
    batched = velocity(gpu_prob, points)
    @test batched isa Vector{NTuple{2,SVector{2,Float64}}}
    @test all(zip(batched, velocity(cpu_prob, points))) do (a, b)
        all(isapprox.(a, b; rtol=1e-8, atol=1e-8))
    end
    @test circulation(gpu_prob) ≈ circulation(cpu_prob) rtol=1e-10 atol=1e-10
    @test enstrophy(gpu_prob) ≈ enstrophy(cpu_prob) rtol=1e-10 atol=1e-10
    @test angular_momentum(gpu_prob) ≈ angular_momentum(cpu_prob) rtol=1e-10 atol=1e-10
    @test vortex_area(gpu_prob) == vortex_area(cpu_prob)
    @test energy(gpu_prob) ≈ energy(cpu_prob) rtol=1e-7 atol=1e-10

    cpu_stepper = RK4Stepper(0.001, total_nodes(cpu_prob); dev=CPU())
    gpu_stepper = RK4Stepper(0.001, total_nodes(gpu_prob); dev=GPU())
    timestep!(cpu_prob, cpu_stepper)
    timestep!(gpu_prob, gpu_stepper)
    for layer in 1:2
        @test all(zip(materialize_contours(gpu_prob)[layer], cpu_prob.layers[layer])) do (a, b)
            all(isapprox.(a.nodes, b.nodes; rtol=1e-8, atol=1e-8))
        end
    end

    if domain isa PeriodicDomain
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)
        surgery!(cpu_prob, params)
        surgery!(gpu_prob, params)
        gpu_layers = materialize_contours(gpu_prob)
        for layer in 1:2
            @test nnodes.(gpu_layers[layer]) == nnodes.(cpu_prob.layers[layer])
            @test all(zip(gpu_layers[layer], cpu_prob.layers[layer])) do (a, b)
                all(isapprox.(a.nodes, b.nodes; rtol=1e-10, atol=1e-10))
            end
        end
    end
end

@testset "CUDA surgery backend" begin
    if !_cuda_available()
        @test true
    else
        CUDA.allowscalar(false)

        @testset "CUDA single-layer velocity and energy match CPU references ($T)" for T in (Float32, Float64)
            for kernel in (EulerKernel(), QGKernel(T(1.25)), SQGKernel(T(0.02)))
                _test_cuda_velocity_and_energy(kernel, UnboundedDomain(); T=T)
                _test_cuda_velocity_and_energy(kernel, PeriodicDomain(T(2), T(2)); T=T)
            end
        end

        @testset "CUDA single-layer timestepping uses device state" begin
            contours = [
                circular_patch(0.35, 20, 1.0),
                PVContour([p + SVector(0.8, -0.2) for p in circular_patch(0.16, 12, -0.4).nodes], -0.4),
            ]
            cpu_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours); dev=CPU())
            gpu_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours); dev=GPU())
            cpu_rk = RK4Stepper(0.002, total_nodes(cpu_prob); dev=CPU())
            gpu_rk = RK4Stepper(0.002, total_nodes(gpu_prob); dev=GPU())
            stale_first = gpu_prob.contours[1].nodes[1]

            # Device topology is authoritative for GPU problems. Deliberately
            # desynchronize the host shadow so this catches dispatch falling
            # through to the generic host-contour implementation.
            device_n = length(gpu_prob.device_state.x)
            pop!(gpu_prob.contours[1].nodes)
            @test total_nodes(gpu_prob) == device_n
            @test_throws ErrorException contours(gpu_prob)

            timestep!(cpu_prob, cpu_rk)
            timestep!(gpu_prob, gpu_rk)
            gpu_contours = materialize_contours(gpu_prob)
            @test !(gpu_rk.k1 isa Vector)
            @test gpu_prob.contours[1].nodes[1] == stale_first
            @test all(zip(gpu_contours, cpu_prob.contours)) do (gpu_c, cpu_c)
                all(isapprox.(gpu_c.nodes, cpu_c.nodes; rtol=1e-8, atol=1e-8))
            end
            point = SVector(0.1, -0.15)
            @test velocity(gpu_prob, point) ≈ velocity(cpu_prob, point) rtol=1e-8 atol=1e-8

            # A device topology rewrite may change its flat node count between
            # timesteps. Post-surgery handling must resize CuArray work buffers
            # from their actual size, even if the caller's count is stale.
            stale_gpu_rk = RK4Stepper(0.002, total_nodes(gpu_prob) - 1; dev=GPU())
            ContourDynamics._handle_post_surgery!(
                gpu_prob, stale_gpu_rk, total_nodes(gpu_prob))
            @test length(stale_gpu_rk.k1) == total_nodes(gpu_prob)
            @test length(stale_gpu_rk.nodes_buf) == total_nodes(gpu_prob)
        end

        @testset "CUDA multi-layer paths match CPU references" begin
            _test_cuda_multilayer_paths(UnboundedDomain())
            _test_cuda_multilayer_paths(PeriodicDomain(2.0, 2.0))
        end

        @testset "CUDA periodic surgery stays device-resident across the seam" begin
            domain = PeriodicDomain(2.0, 2.0)
            contours = [
                rectangle_patch(1.2, 1.99, -0.5, 0.5, 8, 1.0),
                rectangle_patch(-1.99, -1.2, -0.5, 0.5, 8, 1.0),
            ]
            params = SurgeryParams(0.03, 0.12, 0.25, 1e-8, 10)
            cpu_prob = ContourProblem(
                EulerKernel(), domain, deepcopy(contours); dev=CPU())
            gpu_prob = ContourProblem(
                EulerKernel(), domain, deepcopy(contours); dev=GPU())
            stale_shadow = deepcopy(gpu_prob.contours)

            surgery!(cpu_prob, params)
            surgery!(gpu_prob, params)
            actual = materialize_contours(gpu_prob)

            @test all(zip(gpu_prob.contours, stale_shadow)) do (actual_shadow, stale)
                actual_shadow.nodes == stale.nodes &&
                    actual_shadow.pv == stale.pv &&
                    actual_shadow.wrap == stale.wrap &&
                    actual_shadow.corners == stale.corners
            end
            @test nnodes.(actual) == nnodes.(cpu_prob.contours)
            @test all(zip(actual, cpu_prob.contours)) do (a, b)
                a.pv == b.pv && a.wrap == b.wrap && a.corners == b.corners &&
                    all(isapprox.(a.nodes, b.nodes; rtol=1e-8, atol=1e-10))
            end
        end

        @testset "CUDA evolve! with surgery in the loop matches CPU" begin
            # Transcription of "Full evolve! with dev=CPU()" (test_device_state.jl)
            # run on both backends: surgery every step, so remesh, filament
            # cleanup and the post-surgery stepper resize all run inside the
            # loop. Positions are compared through node counts, sorted areas
            # and circulation at a loose tolerance.
            contours = [
                circular_patch(0.5, 64, 1.0),
                PVContour([p + SVector(0.9, -0.3) for p in circular_patch(0.2, 32, -0.6).nodes], -0.6),
            ]
            params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 1)
            cpu_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours); dev=CPU())
            gpu_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours); dev=GPU())
            cpu_stepper = RK4Stepper(0.01, total_nodes(cpu_prob); dev=CPU())
            gpu_stepper = RK4Stepper(0.01, total_nodes(gpu_prob); dev=GPU())
            circ_before = circulation(gpu_prob)

            evolve!(cpu_prob, cpu_stepper, params; nsteps=10)
            evolve!(gpu_prob, gpu_stepper, params; nsteps=10)
            gpu_contours = materialize_contours(gpu_prob)

            @test !(gpu_stepper.k1 isa Vector)
            @test length(gpu_stepper.k1) == total_nodes(gpu_prob)
            @test length(gpu_contours) == length(cpu_prob.contours)
            @test total_nodes(gpu_prob) == total_nodes(cpu_prob)
            @test nnodes.(gpu_contours) == nnodes.(cpu_prob.contours)
            @test all(isapprox.(sort(vortex_area.(gpu_contours)),
                                sort(vortex_area.(cpu_prob.contours)); rtol=1e-6))
            @test circulation(gpu_prob) ≈ circulation(cpu_prob) rtol=1e-6
            @test circulation(gpu_prob) ≈ circ_before rtol=1e-6
        end

        @testset "CUDA periodic single-layer stepping wraps like CPU" begin
            # Transcription of the CPU-backend RK4 and periodic wrapping parity
            # tests in test_device_state.jl, iterated for three steps so that
            # wrapped coordinates feed the next velocity evaluation.
            clear_ewald_cache!()
            domain = PeriodicDomain(2.0, 2.0)
            # The first patch starts just past the x seam, so the first
            # wrap_nodes! translates it as a whole across the box.
            contours = [
                circular_patch(0.3, 24, 1.0; cx=2.05),
                circular_patch(0.2, 16, -0.5; cx=-0.5, cy=0.4),
            ]
            cpu_prob = ContourProblem(EulerKernel(), domain, deepcopy(contours); dev=CPU())
            gpu_prob = ContourProblem(EulerKernel(), domain, deepcopy(contours); dev=GPU())
            cpu_stepper = RK4Stepper(0.002, total_nodes(cpu_prob); dev=CPU())
            gpu_stepper = RK4Stepper(0.002, total_nodes(gpu_prob); dev=GPU())

            for _ in 1:3
                timestep!(cpu_prob, cpu_stepper)
                wrap_nodes!(cpu_prob)
                timestep!(gpu_prob, gpu_stepper)
                wrap_nodes!(gpu_prob)
                gpu_contours = materialize_contours(gpu_prob)
                @test nnodes.(gpu_contours) == nnodes.(cpu_prob.contours)
                @test all(zip(gpu_contours, cpu_prob.contours)) do (a, b)
                    a.wrap == b.wrap &&
                        all(isapprox.(a.nodes, b.nodes; rtol=1e-8, atol=1e-8))
                end
            end
            # The wrapped device state is what the probe sees.
            point = SVector(0.1, -0.15)
            @test velocity(gpu_prob, point) ≈ velocity(cpu_prob, point) rtol=1e-8 atol=1e-8
        end

        @testset "CUDA single-layer QG and SQG surgery match CPU" begin
            # Same style as the multi-layer surgery test: full surgery! on both
            # backends, compared contour by contour. The periodic QG case reuses
            # the cross-seam rectangles; the unbounded SQG case reuses the
            # adjacent-square merge used by the planner/rewrite tests below.
            cases = (
                (QGKernel(1.25), PeriodicDomain(2.0, 2.0),
                 [rectangle_patch(1.2, 1.99, -0.5, 0.5, 8, 1.0),
                  rectangle_patch(-1.99, -1.2, -0.5, 0.5, 8, 1.0)],
                 SurgeryParams(0.03, 0.12, 0.25, 1e-8, 10)),
                (SQGKernel(0.02), UnboundedDomain(),
                 [rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
                  rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0)],
                 SurgeryParams(0.01, 0.04, 0.16, 1e-6, 10)),
            )
            for (kernel, domain, contours, params) in cases
                clear_ewald_cache!()
                cpu_prob = ContourProblem(kernel, domain, deepcopy(contours); dev=CPU())
                gpu_prob = ContourProblem(kernel, domain, deepcopy(contours); dev=GPU())
                stale_shadow = deepcopy(gpu_prob.contours)

                surgery!(cpu_prob, params)
                surgery!(gpu_prob, params)
                actual = materialize_contours(gpu_prob)

                @test all(zip(gpu_prob.contours, stale_shadow)) do (actual_shadow, stale)
                    actual_shadow.nodes == stale.nodes && actual_shadow.pv == stale.pv
                end
                @test length(actual) == length(cpu_prob.contours)
                @test nnodes.(actual) == nnodes.(cpu_prob.contours)
                @test all(zip(actual, cpu_prob.contours)) do (a, b)
                    a.pv == b.pv && a.wrap == b.wrap && a.corners == b.corners &&
                        all(isapprox.(a.nodes, b.nodes; rtol=1e-8, atol=1e-10))
                end
                # Post-surgery device topology still feeds the velocity path.
                point = SVector(0.1, -0.15)
                @test velocity(gpu_prob, point) ≈ velocity(cpu_prob, point) rtol=1e-8 atol=1e-8
            end
        end

        @testset "CUDA surgery removes a tiny filament on both backends" begin
            # Transcription of the unbounded multi-layer surgery test's layer 1
            # (a healthy patch plus a three-node filament) as a single-layer
            # problem: the filament must be dropped by surgery! on both backends.
            tiny = PVContour([SVector(2.0, 0.0), SVector(2.0 + 1e-6, 0.0), SVector(2.0, 1e-6)], 1.0)
            contours = [circular_patch(0.5, 32, 1.0), tiny]
            params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)
            cpu_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours); dev=CPU())
            gpu_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours); dev=GPU())
            @test length(materialize_contours(gpu_prob)) == 2

            surgery!(cpu_prob, params)
            surgery!(gpu_prob, params)
            actual = materialize_contours(gpu_prob)

            @test length(cpu_prob.contours) == 1
            @test length(actual) == 1
            @test total_nodes(gpu_prob) == total_nodes(cpu_prob)
            @test all(zip(actual, cpu_prob.contours)) do (a, e)
                a.pv == e.pv && length(a.nodes) == length(e.nodes) &&
                    all(isapprox.(a.nodes, e.nodes; rtol=1e-8, atol=1e-10))
            end
        end

        @testset "CUDA fixed-corner remesh of a zero-length span matches CPU" begin
            # Transcription of the CPU-backend fixed-corner remesh test in
            # test_device_surgery.jl.
            base = circular_patch(1.0, 40, 1.0)
            nodes = copy(base.nodes)
            insert!(nodes, 11, nodes[10])
            corners = falses(length(nodes))
            corners[10] = true
            corners[11] = true
            corners[31] = true
            c = PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)
            params = SurgeryParams(0.005, 0.04, 0.16, 1e-6, 10)

            cpu_c = remesh(c, params; _density_sources=[c])
            gpu_c = only(ContourDynamics._device_remesh_contours([c], params, GPU()))
            @test nnodes(gpu_c) == nnodes(cpu_c)
            @test gpu_c.corners == cpu_c.corners
            @test all(isapprox.(gpu_c.nodes, cpu_c.nodes; atol=1e-10, rtol=1e-10))
        end

        @testset "CUDA beta-plane velocity matches CPU reference" begin
            domain = PeriodicDomain(2.0, 2.0)
            reference = beta_staircase(0.4, domain, 4; nodes_per_contour=8)
            kernel = BetaPlaneQGKernel(0.4, 1.0, reference)
            live = vcat(deepcopy(reference), [circular_patch(0.25, 16, 2π; cy=0.5)])
            cpu_prob = ContourProblem(kernel, domain, deepcopy(live); dev=CPU())
            gpu_prob = ContourProblem(kernel, domain, deepcopy(live); dev=GPU())
            expected = zeros(SVector{2,Float64}, total_nodes(cpu_prob))
            actual = device_zeros(GPU(), SVector{2,Float64}, total_nodes(gpu_prob))

            velocity!(expected, cpu_prob)
            velocity!(actual, gpu_prob)
            @test all(isapprox.(to_cpu(actual), expected; rtol=1e-8, atol=1e-8))
            flat = ContourDynamics._flat_topology(gpu_prob.device_state, GPU())
            eligible = ContourDynamics._device_eligible_surgery_segment_indices(
                flat, GPU())
            @test length(eligible) == nnodes(live[end])
            point = SVector(0.13, -0.17)
            @test velocity(gpu_prob, point) ≈ velocity(cpu_prob, point) rtol=1e-8 atol=1e-8
            mixed_point = SVector{2,Float32}(point)
            @test velocity(gpu_prob, mixed_point) ≈
                  velocity(cpu_prob, mixed_point) rtol=1e-8 atol=1e-8

            cpu_stepper = RK4Stepper(0.001, total_nodes(cpu_prob); dev=CPU())
            gpu_stepper = RK4Stepper(0.001, total_nodes(gpu_prob); dev=GPU())
            timestep!(cpu_prob, cpu_stepper)
            timestep!(gpu_prob, gpu_stepper)
            @test all(zip(materialize_contours(gpu_prob), cpu_prob.contours)) do (a, b)
                all(isapprox.(a.nodes, b.nodes; rtol=1e-8, atol=1e-8))
            end

            fname = tempname() * ".jld2"
            try
                save_snapshot(fname, gpu_prob, 1; diagnostics=false)
                restarted = load_problem(fname, 1; dev=GPU())
                @test restarted.kernel isa BetaPlaneQGKernel
                @test all(zip(restarted.kernel.reference_contours,
                              gpu_prob.kernel.reference_contours)) do (a, b)
                    a.nodes == b.nodes && a.pv == b.pv &&
                        a.wrap == b.wrap && a.corners == b.corners
                end
                @test all(zip(materialize_contours(restarted),
                              materialize_contours(gpu_prob))) do (a, b)
                    a.nodes == b.nodes && a.pv == b.pv &&
                        a.wrap == b.wrap && a.corners == b.corners
                end
                @test velocity(restarted, point) ≈ velocity(gpu_prob, point) rtol=1e-8 atol=1e-8
            finally
                rm(fname; force=true)
            end
        end

        δ = 0.02
        contours = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(3.0, 4.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(4.012, 5.0, 0.0, 1.0, 6, 1.0),
        ]

        cpu_candidates = ContourDynamics._device_admissible_close_segment_buffer(
            contours, δ, UnboundedDomain(), CPU())
        gpu_candidates = ContourDynamics._device_admissible_close_segment_buffer(
            contours, δ, UnboundedDomain(), GPU())
        @test Set(ContourDynamics._unpack_close_pair_candidates(gpu_candidates)) ==
              Set(ContourDynamics._unpack_close_pair_candidates(cpu_candidates))

        cpu_selected = ContourDynamics._device_select_reconnection_pair_buffer(
            contours, cpu_candidates, CPU())
        gpu_selected = ContourDynamics._device_select_reconnection_pair_buffer(
            contours, gpu_candidates, GPU())
        @test Set(ContourDynamics._unpack_close_pair_candidates(gpu_selected)) ==
              Set(ContourDynamics._unpack_close_pair_candidates(cpu_selected))

        cpu_rewritten = ContourDynamics._device_rewrite_contours(contours, cpu_selected, CPU())
        gpu_rewritten = ContourDynamics._device_rewrite_contours(contours, gpu_selected, GPU())
        @test nnodes.(gpu_rewritten) == nnodes.(cpu_rewritten)
        @test all(zip(gpu_rewritten, cpu_rewritten)) do (gpu_c, cpu_c)
            all(isapprox.(gpu_c.nodes, cpu_c.nodes; atol=1e-11, rtol=1e-11))
        end

        remesh_contours = [
            elliptical_patch(1.0, 0.6, 40, 1.0),
            circular_patch(0.35, 24, 0.5),
        ]
        params = SurgeryParams(0.005, 0.04, 0.16, 1e-6, 10)
        cpu_remeshed = ContourDynamics._device_remesh_contours(remesh_contours, params, CPU())
        gpu_remeshed = ContourDynamics._device_remesh_contours(remesh_contours, params, GPU())
        @test nnodes.(gpu_remeshed) == nnodes.(cpu_remeshed)
        @test all(zip(gpu_remeshed, cpu_remeshed)) do (gpu_c, cpu_c)
            all(isapprox.(gpu_c.nodes, cpu_c.nodes; atol=1e-10, rtol=1e-10))
        end

        cpu_prob = Problem(; contours=deepcopy(contours), dt=0.01, dev=CPU())
        gpu_prob = Problem(; contours=deepcopy(contours), dt=0.01, dev=GPU())
        stale_gpu_shadow = deepcopy(gpu_prob.contour_problem.contours)
        surgery!(cpu_prob, SurgeryParams(0.005, 0.04, 0.16, 1e-6, 10))
        surgery!(gpu_prob, SurgeryParams(0.005, 0.04, 0.16, 1e-6, 10))
        gpu_contours = materialize_contours(gpu_prob)
        @test all(zip(gpu_prob.contour_problem.contours, stale_gpu_shadow)) do (actual, stale)
            actual.nodes == stale.nodes &&
                actual.pv == stale.pv &&
                actual.wrap == stale.wrap &&
                actual.corners == stale.corners
        end
        @test nnodes.(gpu_contours) == nnodes.(cpu_prob.contours)
        @test getproperty.(gpu_contours, :pv) == getproperty.(cpu_prob.contours, :pv)
    end
end
