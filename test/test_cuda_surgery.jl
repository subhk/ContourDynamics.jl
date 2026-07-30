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

function _cuda_rectangle_patch(xmin, xmax, ymin, ymax, nside, pv)
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

function _test_cuda_velocity_and_energy(kernel, domain; atol=1e-8, rtol=1e-8)
    clear_ewald_cache!()
    c1 = circular_patch(0.35, 24, 1.0)
    c2 = PVContour([p + SVector(0.9, -0.35) for p in circular_patch(0.18, 16, -0.4).nodes], -0.4)
    cpu_prob = ContourProblem(kernel, domain, [c1, c2]; dev=CPU())
    gpu_prob = ContourProblem(kernel, domain, deepcopy([c1, c2]); dev=GPU())
    n = total_nodes(cpu_prob)

    vel_ref = zeros(SVector{2,Float64}, n)
    ContourDynamics._direct_velocity!(vel_ref, cpu_prob)

    dev_vel = device_zeros(GPU(), SVector{2,Float64}, n)
    velocity!(dev_vel, gpu_prob)
    @test !(dev_vel isa Vector)
    vel_gpu = to_cpu(dev_vel)

    @test all(eachindex(vel_ref)) do i
        isapprox(vel_gpu[i][1], vel_ref[i][1]; atol, rtol) &&
            isapprox(vel_gpu[i][2], vel_ref[i][2]; atol, rtol)
    end
    point = SVector(0.13, -0.17)
    @test velocity(gpu_prob, point) ≈ velocity(cpu_prob, point) atol=atol rtol=rtol
    stale_shadow = deepcopy(gpu_prob.contours)
    gpu_prob.contours[1].nodes[1] = SVector(99.0, 99.0)
    mixed_point = SVector{2,Float32}(point)
    @test velocity(gpu_prob, mixed_point) ≈ velocity(cpu_prob, mixed_point) atol=atol rtol=rtol
    @test energy(gpu_prob) ≈ energy(cpu_prob) rtol=1e-7 atol=1e-10
    @test circulation(gpu_prob) ≈ circulation(cpu_prob) rtol=1e-10 atol=1e-10
    @test enstrophy(gpu_prob) ≈ enstrophy(cpu_prob) rtol=1e-10 atol=1e-10
    @test angular_momentum(gpu_prob) ≈ angular_momentum(cpu_prob) rtol=1e-10 atol=1e-10
    @test vortex_area(gpu_prob) ≈ vortex_area(cpu_prob) rtol=1e-10 atol=1e-10

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

        @testset "CUDA single-layer velocity and energy match CPU references" begin
            for kernel in (EulerKernel(), QGKernel(1.25), SQGKernel(0.02))
                _test_cuda_velocity_and_energy(kernel, UnboundedDomain())
                _test_cuda_velocity_and_energy(kernel, PeriodicDomain(2.0, 2.0))
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

            cpu_lf = LeapfrogStepper(0.002, total_nodes(cpu_prob); dev=CPU())
            gpu_prob_lf = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours); dev=GPU())
            gpu_lf = LeapfrogStepper(0.002, total_nodes(gpu_prob_lf); dev=GPU())
            cpu_prob_lf = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours); dev=CPU())
            for _ in 1:2
                timestep!(cpu_prob_lf, cpu_lf)
                timestep!(gpu_prob_lf, gpu_lf)
            end
            gpu_lf_contours = materialize_contours(gpu_prob_lf)
            @test gpu_lf.initialized
            @test all(zip(gpu_lf_contours, cpu_prob_lf.contours)) do (gpu_c, cpu_c)
                all(isapprox.(gpu_c.nodes, cpu_c.nodes; rtol=1e-8, atol=1e-8))
            end
        end

        @testset "CUDA multi-layer paths match CPU references" begin
            _test_cuda_multilayer_paths(UnboundedDomain())
            _test_cuda_multilayer_paths(PeriodicDomain(2.0, 2.0))
        end

        @testset "CUDA periodic surgery stays device-resident across the seam" begin
            domain = PeriodicDomain(2.0, 2.0)
            contours = [
                _cuda_rectangle_patch(1.2, 1.99, -0.5, 0.5, 8, 1.0),
                _cuda_rectangle_patch(-1.99, -1.2, -0.5, 0.5, 8, 1.0),
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
        end

        δ = 0.02
        contours = [
            _cuda_rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            _cuda_rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
            _cuda_rectangle_patch(3.0, 4.0, 0.0, 1.0, 6, 1.0),
            _cuda_rectangle_patch(4.012, 5.0, 0.0, 1.0, 6, 1.0),
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
