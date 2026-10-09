using ContourDynamics
using StaticArrays
using Test
using Logging

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

@testset "Device surgery" begin
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

    @testset "Device remeshing is independent of coordinate units" begin
        for T in (Float32, Float64)
            c = elliptical_patch(2, 1, 96, 1; T=T)
            params = SurgeryParams(T(0.01), T(0.05), T(0.5), T(1e-8), 5)
            baseline = only(ContourDynamics._device_remesh_contours([c], params, CPU()))
            tolerance = T === Float32 ? T(2e-4) : T(1e-10)
            for scale in (T(0.01), T(100))
                scaled = PVContour([scale * x for x in c.nodes], c.pv)
                scaled_params = SurgeryParams(params.δ * scale, params.μ * scale,
                    params.Δ_max * scale, params.area_min * scale^2, params.n_surgery)
                actual = only(ContourDynamics._device_remesh_contours([scaled], scaled_params, CPU()))
                host = remesh(scaled, scaled_params)

                @test nnodes(actual) == nnodes(baseline) == nnodes(host)
                @test actual.nodes ./ scale ≈ baseline.nodes rtol=tolerance
                @test actual.nodes ≈ host.nodes rtol=tolerance
            end
        end
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
        δ = 0.02
        contours_in = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
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
        δ = 0.02
        contours_in = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
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

        # Raw candidates include right-angle contacts at the squares' corners,
        # where the two parts' far sides lie in different fluid; admissibility
        # drops those on both backends and keeps every facing-side contact.
        facing = Set(p for p in dev_pairs if 7 <= p[2] <= 12 && 19 <= p[4] <= 24)
        @test !isempty(cpu_pairs)
        @test cpu_pairs == admissible_pairs
        @test issubset(cpu_pairs, dev_pairs)
        @test issubset(facing, cpu_pairs)
        @test buffered_pairs == dev_pairs
        @test length(to_cpu(buffer.ci)) == length(dev_pairs)

        contours_different_pv = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 2.0),
        ]
        @test isempty(ContourDynamics._device_close_pair_candidates(contours_different_pv, δ, CPU()))

        spanning_nodes = [SVector(cospi(2k / 256), sinpi(2k / 256))
                          for k in 0:255]
        spanning = PVContour(spanning_nodes, 1.0, SVector(4.0, 0.0))
        with_spanning = vcat(contours_same, [spanning])
        flat = ContourDynamics._pack_flat_topology(with_spanning, CPU())
        eligible = ContourDynamics._device_eligible_surgery_segment_indices(
            flat, CPU())
        @test length(eligible) == sum(nnodes, contours_same)
        @test Set(ContourDynamics._device_close_pair_candidates(
            with_spanning, δ, CPU())) == dev_pairs
    end

    @testset "Triangular pair index enumerates every unordered pair once" begin
        for n in (0, 1, 2, 3, 7, 64, 1500)
            npairs = ContourDynamics._triangular_pair_count(n)
            @test npairs == n * (n - 1) ÷ 2
            seen = Set{Tuple{Int,Int}}()
            prev = (0, 0)
            in_range = true
            ordered = true
            for p in 1:npairs
                a, b = ContourDynamics._triangular_pair(p)
                in_range &= 1 <= a < b <= n
                # Same order as the former full-square sweep: b outer, a inner.
                ordered &= (b, a) > prev
                prev = (b, a)
                push!(seen, (a, b))
            end
            @test in_range
            @test ordered
            @test length(seen) == npairs
        end
        # Spot-check far beyond the Float32-exact range of the square root.
        for p in (2^31 - 1, 2^31, 10^12 + 7, 2^41 + 12345)
            a, b = ContourDynamics._triangular_pair(p)
            @test 1 <= a < b
            @test (b - 1) * (b - 2) ÷ 2 + a == p
        end
    end

    @testset "Chunked pair scan matches CPU reference across chunk boundary" begin
        # 3000 eligible segments → npairs = 3000·2999/2 ≈ 4.5e6 >
        # _PAIR_SCAN_CHUNK, so the candidate sweep spans several chunks;
        # results must match the CPU spatial-index reference exactly.
        δ = 0.05
        contours = [circular_patch(1.0, 1500, 1.0),
                    circular_patch(1.0, 1500, 1.0; cx=2.03)]
        @test ContourDynamics._triangular_pair_count(3000) > ContourDynamics._PAIR_SCAN_CHUNK
        idx = ContourDynamics.build_spatial_index(contours, δ)
        cpu_pairs = Set(ContourDynamics.find_close_segments(contours, idx, δ))
        buffer = ContourDynamics._device_close_pair_candidate_buffer(contours, δ, CPU())
        buffered = Set(ContourDynamics._unpack_close_pair_candidates(buffer))
        @test !isempty(cpu_pairs)
        @test cpu_pairs == buffered
    end

    @testset "Device close-pair admissibility uses periodic minimum images" begin
        domain = PeriodicDomain(2.0, 2.0)
        δ = 0.03
        contours_in = [
            rectangle_patch(1.2, 1.99, -0.5, 0.5, 6, 1.0),
            rectangle_patch(-1.99, -1.2, -0.5, 0.5, 6, 1.0 + 1e-8),
        ]
        index = ContourDynamics.build_spatial_index(contours_in, δ, domain)
        expected = ContourDynamics.find_close_segments(contours_in, index, δ, domain)
        state = DeviceContourState(deepcopy(contours_in), CPU())

        actual = ContourDynamics._device_admissible_close_segment_buffer(
            state, δ, domain, CPU())
        unbounded = ContourDynamics._device_admissible_close_segment_buffer(
            state, δ, UnboundedDomain(), CPU())

        @test !isempty(expected)
        @test Set(ContourDynamics._unpack_close_pair_candidates(actual)) == Set(expected)
        @test isempty(ContourDynamics._unpack_close_pair_candidates(unbounded))

        expected_merge = deepcopy(contours_in)
        ContourDynamics.reconnect!(expected_merge, expected, domain)
        rewrite_state = DeviceContourState(deepcopy(contours_in), CPU())
        selected = ContourDynamics._device_select_reconnection_pair_buffer(
            rewrite_state, actual, domain, CPU())
        ContourDynamics._device_rewrite_state!(
            rewrite_state, selected, domain, CPU())
        actual_merge = materialize_contours(rewrite_state)
        @test length(actual_merge) == length(expected_merge)
        @test all(zip(actual_merge, expected_merge)) do (a, b)
            a.pv == b.pv && a.wrap == b.wrap && a.corners == b.corners &&
                all(isapprox.(a.nodes, b.nodes; rtol=1e-12, atol=1e-12))
        end

        nested = [
            PVContour([p + SVector(1.8, 0.0) for p in circular_patch(1.0, 96, 1.0).nodes], 1.0),
            PVContour([p + SVector(1.8, 0.0) for p in circular_patch(0.98, 96, 1.0).nodes], 1.0),
        ]
        nested_state = DeviceContourState(deepcopy(nested), CPU())
        raw_nested = ContourDynamics._device_close_pair_candidate_buffer(
            nested_state, 0.05, domain, CPU())
        admissible_nested = ContourDynamics._device_admissible_close_segment_buffer(
            nested_state, 0.05, domain, CPU())
        nested_index = ContourDynamics.build_spatial_index(nested, 0.05, domain)
        expected_nested = ContourDynamics.find_close_segments(
            nested, nested_index, 0.05, domain)
        @test !isempty(ContourDynamics._unpack_close_pair_candidates(raw_nested))
        @test isempty(expected_nested)
        @test isempty(ContourDynamics._unpack_close_pair_candidates(admissible_nested))
    end

    @testset "Device surgery preserves weak PV levels and periodic containment" begin
        for T in (Float32, Float64)
            δ = T(0.02)
            for q in (one(T), T(1e-10)), factor in (-one(T), zero(T), one(T), T(2))
                cs = [circular_patch(0.2, 32, q; cx=-0.205, T=T),
                      circular_patch(0.2, 32, factor * q; cx=0.205, T=T)]
                candidates = ContourDynamics._device_admissible_close_segment_buffer(
                    cs, δ, UnboundedDomain(), CPU())
                pairs = ContourDynamics._unpack_close_pair_candidates(candidates)
                @test any(p -> p[1] != p[3], pairs) == (factor == one(T))
            end
            nested = [circular_patch(1, 64, 1e-10; T=T),
                      circular_patch(0.99, 64, 1e-10; T=T)]
            candidates = ContourDynamics._device_admissible_close_segment_buffer(
                nested, δ, UnboundedDomain(), CPU())
            @test isempty(ContourDynamics._unpack_close_pair_candidates(candidates))

            cs = [circular_patch(0.2, 64, 2; T=T),
                  circular_patch(0.05, 64, 1; cx=0.745, T=T),
                  circular_patch(0.05, 64, 1; cx=0.855, T=T)]
            state = DeviceContourState(cs, CPU())
            pairs_by_domain = map((UnboundedDomain(), PeriodicDomain(one(T)))) do d
                candidates = ContourDynamics._device_admissible_close_segment_buffer(state, δ, d, CPU())
                pairs = ContourDynamics._unpack_close_pair_candidates(candidates)
                Set(p for p in pairs if p[1] == 2 && p[3] == 3)
            end
            @test !isempty(pairs_by_domain[1])
            @test pairs_by_domain[2] == pairs_by_domain[1]
        end
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
        device_selected = _device_select_reconnection_pairs(contours, close_pairs, CPU())
        candidate_buffer = ContourDynamics._device_close_pair_candidate_buffer(contours, δ, CPU())
        buffer_pairs = ContourDynamics._unpack_close_pair_candidates(candidate_buffer)
        buffer_selected = _device_select_reconnection_pairs(contours, candidate_buffer, CPU())
        selected_buffer = ContourDynamics._device_select_reconnection_pair_buffer(
            contours, candidate_buffer, CPU())
        selected_buffer_pairs = ContourDynamics._unpack_close_pair_candidates(selected_buffer)
        plan = ContourDynamics._device_reconnection_plan(contours, close_pairs, CPU())
        buffer_plan = ContourDynamics._device_reconnection_plan(contours, candidate_buffer, CPU())
        rewrite = ContourDynamics._device_topology_rewrite_plan(contours, device_selected, CPU())
        buffer_rewrite = ContourDynamics._device_topology_rewrite_plan(contours, selected_buffer, CPU())
        _assert_device_layout_matches_host(contours, rewrite)
        materialized = ContourDynamics._unpack_rewrite_outputs(
            _device_materialize_rewrite_outputs(contours, device_selected, CPU()))
        expected_merge_lengths = [
            nnodes(contours[pair[1]]) + nnodes(contours[pair[3]]) + 1
            for pair in device_selected
        ]

        admissible_pairs = ContourDynamics._unpack_close_pair_candidates(
            ContourDynamics._device_admissible_close_segment_buffer(
                contours, δ, UnboundedDomain(), CPU()))
        @test !isempty(close_pairs)
        @test Set(admissible_pairs) == Set(close_pairs)
        @test issubset(Set(close_pairs), Set(buffer_pairs))
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
            _device_materialize_rewrite_outputs(contours, selected, CPU()))

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

        domain = PeriodicDomain(4.0, 4.0)
        periodic_state = DeviceContourState(deepcopy(contours), CPU())
        periodic_candidates = ContourDynamics._device_admissible_close_segment_buffer(
            periodic_state, δ, domain, CPU())
        periodic_selected = ContourDynamics._device_select_reconnection_pair_buffer(
            periodic_state, periodic_candidates, domain, CPU())
        ContourDynamics._device_rewrite_state!(
            periodic_state, periodic_selected, domain, CPU())
        periodic_actual = materialize_contours(periodic_state)
        periodic_expected = deepcopy(contours)
        periodic_index = ContourDynamics.build_spatial_index(
            periodic_expected, δ, domain)
        periodic_pairs = ContourDynamics.find_close_segments(
            periodic_expected, periodic_index, δ, domain)
        ContourDynamics.reconnect!(periodic_expected, periodic_pairs, domain)
        @test nnodes.(periodic_actual) == nnodes.(periodic_expected)
        @test all(zip(periodic_actual, periodic_expected)) do (a, b)
            a.corners == b.corners &&
                all(isapprox.(a.nodes, b.nodes; rtol=1e-12, atol=1e-12))
        end
    end

    @testset "Device full topology rewrite preserves unchanged contours" begin
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
end

@testset "Multi-layer device surgery" begin
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

    @testset "Periodic multi-layer device surgery matches CPU" begin
        F = 0.5
        kernel = MultiLayerQGKernel(
            SVector(1.0), SMatrix{2,2,Float64}(-F, F, F, -F))
        domain = PeriodicDomain(2.0, 2.0)
        layers = (
            [circular_patch(0.3, 16, 1.0)],
            [circular_patch(0.2, 12, -0.5; cx=0.6)],
        )
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)
        cpu_prob = MultiLayerContourProblem(kernel, domain, deepcopy(layers))
        states = ntuple(i -> DeviceContourState(deepcopy(layers[i]), CPU()), 2)

        surgery!(cpu_prob, params)
        ContourDynamics._device_multilayer_surgery!(states, params, domain, CPU())
        for layer in 1:2
            actual = materialize_contours(states[layer])
            @test length(actual) == length(cpu_prob.layers[layer])
            @test all(zip(actual, cpu_prob.layers[layer])) do (a, b)
                a.pv == b.pv && a.wrap == b.wrap && a.corners == b.corners &&
                    all(isapprox.(a.nodes, b.nodes; rtol=1e-12, atol=1e-12))
            end
        end
    end

    @testset "Periodic device surgery handles cross-seam reconnect and cleanup" begin
        domain = PeriodicDomain(2.0, 2.0)
        contours_in = [
            rectangle_patch(1.2, 1.99, -0.5, 0.5, 8, 1.0),
            rectangle_patch(-1.99, -1.2, -0.5, 0.5, 8, 1.0),
            PVContour([SVector(0.0, 1.5), SVector(1e-6, 1.5),
                       SVector(0.0, 1.5 + 1e-6)], 1.0),
        ]
        params = SurgeryParams(0.03, 0.12, 0.25, 1e-8, 10)
        cpu_prob = ContourProblem(EulerKernel(), domain, deepcopy(contours_in))
        state = DeviceContourState(deepcopy(contours_in), CPU())

        surgery!(cpu_prob, params)
        ContourDynamics._device_surgery_pipeline!(state, params, domain, CPU())
        actual = materialize_contours(state)

        @test length(actual) == length(cpu_prob.contours) == 1
        @test nnodes.(actual) == nnodes.(cpu_prob.contours)
        @test all(zip(actual, cpu_prob.contours)) do (a, b)
            a.pv == b.pv && a.wrap == b.wrap && a.corners == b.corners &&
                all(isapprox.(a.nodes, b.nodes; rtol=1e-8, atol=1e-10))
        end
    end
end

@testset "Device surgery and remesh regressions" begin
    circle_nodes(cx, cy, R, n) = [SVector(cx + R * cos(2π * k / n), cy + R * sin(2π * k / n))
                                  for k in 0:(n - 1)]
    function device_surgery(cs, params, domain)
        state = DeviceContourState(deepcopy(cs), CPU())
        ContourDynamics._device_surgery_pipeline!(state, params, domain, CPU())
        return materialize_contours(state)
    end
    net_circulation(cs) = sum(c.pv * vortex_area(c) for c in cs; init=0.0)

    @testset "merges compare physical far-side PV levels" begin
        params = SurgeryParams(0.005, 0.02, 0.1, 1e-6, 1)
        opposite = [PVContour(circle_nodes(-0.5015, 0.0, 0.5, 128), 1.0),
                    PVContour(reverse(circle_nodes(0.5015, 0.0, 0.5, 128)), 1.0)]
        out = device_surgery(opposite, params, UnboundedDomain())
        @test length(out) == 2
        @test abs(net_circulation(out)) < 1e-12

        ring = [PVContour(circle_nodes(0.0, 0.0, 1.0, 256), 1.0),
                PVContour(reverse(circle_nodes(0.197, 0.0, 0.8, 256)), 1.0)]
        out = device_surgery(ring, params, UnboundedDomain())
        @test length(out) == 1
        @test net_circulation(out) ≈ net_circulation(ring) rtol=2e-3
    end

    @testset "splits keep the trapped hole clockwise" begin
        # C-shape with a narrow gap between its tips.
        function cshape(r1, r2, θ0, h)
            nodes = SVector{2,Float64}[]
            nout = ceil(Int, r2 * (2π - 2θ0) / h)
            for k in 0:(nout - 1)
                θ = θ0 + (2π - 2θ0) * k / nout
                push!(nodes, SVector(r2 * cos(θ), r2 * sin(θ)))
            end
            ntip = max(2, ceil(Int, (r2 - r1) / h))
            for k in 0:(ntip - 1)
                ρ = r2 - (r2 - r1) * k / ntip
                push!(nodes, SVector(ρ * cos(2π - θ0), ρ * sin(2π - θ0)))
            end
            nin = ceil(Int, r1 * (2π - 2θ0) / h)
            for k in 0:(nin - 1)
                θ = (2π - θ0) - (2π - 2θ0) * k / nin
                push!(nodes, SVector(r1 * cos(θ), r1 * sin(θ)))
            end
            for k in 0:(ntip - 1)
                ρ = r1 + (r2 - r1) * k / ntip
                push!(nodes, SVector(ρ * cos(θ0), ρ * sin(θ0)))
            end
            return PVContour(nodes, 1.0)
        end
        c = cshape(0.5, 0.8, 0.01, 0.03)
        params = SurgeryParams(0.02, 0.08, 0.1, 1e-6, 10)
        out = device_surgery([c], params, UnboundedDomain())
        host = ContourProblem(EulerKernel(), UnboundedDomain(), [deepcopy(c)])
        surgery!(host, params)
        @test length(out) == 2
        @test count(d -> vortex_area(d) < 0, out) == 1
        @test net_circulation(out) ≈ net_circulation([c]) rtol=5e-3
        @test sort(vortex_area.(out)) ≈ sort(vortex_area.(contours(host))) rtol=1e-10
    end

    @testset "periodic self-image contacts are left alone" begin
        nodes = SVector{2,Float64}[]
        for k in 0:49; push!(nodes, SVector(-0.995 + 1.99 * k / 50, -0.1)); end
        for k in 0:4; push!(nodes, SVector(0.995, -0.1 + 0.2 * k / 5)); end
        for k in 0:49; push!(nodes, SVector(0.995 - 1.99 * k / 50, 0.1)); end
        for k in 0:4; push!(nodes, SVector(-0.995, 0.1 - 0.2 * k / 5)); end
        band = PVContour(nodes, 1.0)
        out = device_surgery([band], SurgeryParams(0.02, 0.08, 0.1, 1e-6, 10),
                             PeriodicDomain(1.0, 1.0))
        @test length(out) == 1
        @test net_circulation(out) ≈ net_circulation([band]) rtol=1e-6
    end

    @testset "remesh leaves contours too short for a curve unchanged" begin
        domain_period = SVector(2.0, 0.0)
        short = PVContour([SVector(-0.5, 0.1), SVector(0.5, 0.15)], 1.0, domain_period)
        empty_spanning = PVContour(SVector{2,Float64}[], 1.0, domain_period)
        patch = circular_patch(0.3, 48, 1.0)
        params = SurgeryParams(0.005, 0.02, 0.1, 1e-6, 1)
        state = DeviceContourState([short, empty_spanning, patch], CPU())
        ContourDynamics._device_remesh_state!(state, params, CPU())
        out = materialize_contours(state)
        host = deepcopy([short, empty_spanning, patch])
        ContourDynamics._remesh_all!(host, params, SVector{2,Float64}[], Float64[],
                                     SVector{2,Float64}[])
        @test nnodes(out[1]) == 2
        @test out[1].nodes == short.nodes
        @test nnodes(out[2]) == 0
        @test nnodes.(out) == nnodes.(host)
    end

    @testset "multi-layer periodic energy reads each mode's cache" begin
        # setup_ewald_cache! for a single-layer QG problem also configures the
        # domain's Euler cache; the baroclinic mode below keeps its own cache.
        clear_ewald_cache!()
        domain = PeriodicDomain(Float64(π))
        setup_ewald_cache!(domain, QGKernel(0.7); n_fourier=16, n_images=4)
        F = 0.5
        kernel = MultiLayerQGKernel(SVector(1 / sqrt(2F)), SMatrix{2,2}(-F, F, F, -F))
        layers = ([circular_patch(0.4, 48, 1.0; cx=0.3)],
                  [circular_patch(0.3, 40, -0.8; cx=-0.4, cy=0.2)])
        prob = MultiLayerContourProblem(kernel, domain, deepcopy(layers))
        states = ntuple(i -> DeviceContourState(deepcopy(layers[i]), CPU()), 2)
        @test ContourDynamics._ka_multilayer_energy_from_states(
            states, kernel, domain, CPU()) ≈ energy(prob) rtol=1e-12
        clear_ewald_cache!()
    end
end

@testset "Device surgery parity edge cases" begin
    # Two pinched "dumbbell" contours whose necks differ in gap. Both split in
    # the same reconnect round; the CPU appends split daughters in proximity
    # order, and the device must lay its pair buffer out the same way so the
    # contour vectors agree element for element, not only as sets.
    function dumbbell(x0, gap; T=Float64)
        N_half = 30
        nodes = SVector{2,T}[]
        for k in 0:N_half
            θ = -π/2 + π * k / N_half
            push!(nodes, SVector{2,T}(x0 + 1 + cos(θ), sin(θ)))
        end
        for k in 1:5
            push!(nodes, SVector{2,T}(x0 + 1 - k * 2 / 6, gap))
        end
        for k in 0:N_half
            θ = π/2 + π * k / N_half
            push!(nodes, SVector{2,T}(x0 - 1 + cos(θ), sin(θ)))
        end
        for k in 1:5
            push!(nodes, SVector{2,T}(x0 - 1 + k * 2 / 6, -gap))
        end
        return PVContour(nodes, T(1))
    end
    same_contours(a, b; rtol=1e-12, atol=1e-12) =
        length(a) == length(b) && all(zip(a, b)) do (p, q)
            p.pv == q.pv && p.wrap == q.wrap && p.corners == q.corners &&
                nnodes(p) == nnodes(q) &&
                all(isapprox.(p.nodes, q.nodes; rtol=rtol, atol=atol))
        end

    @testset "Simultaneous splits keep CPU daughter order" begin
        δ = 0.01
        # The second contour has the closer neck, so proximity order differs
        # from candidate-buffer (contour index) order.
        contours = [dumbbell(0.0, 0.004), dumbbell(6.0, 0.001)]
        domain = UnboundedDomain()

        idx = ContourDynamics.build_spatial_index(contours, δ)
        close_pairs = ContourDynamics.find_close_segments(contours, idx, δ)
        cpu_selected = ContourDynamics._select_reconnection_pairs(contours, close_pairs)
        @test length(cpu_selected) == 2
        @test cpu_selected[1][1] == 2  # closer neck first

        state = DeviceContourState(deepcopy(contours), CPU())
        candidates = ContourDynamics._device_admissible_close_segment_buffer(
            state, δ, domain, CPU())
        selected = ContourDynamics._device_select_reconnection_pair_buffer(
            state, candidates, domain, CPU())
        device_selected = collect(zip(to_cpu(selected.ci), to_cpu(selected.i),
                                      to_cpu(selected.cj), to_cpu(selected.j)))
        @test device_selected == cpu_selected

        cpu_contours = deepcopy(contours)
        ContourDynamics.reconnect!(cpu_contours, cpu_selected, domain)
        ContourDynamics._device_rewrite_state!(state, selected, domain, CPU())
        @test length(cpu_contours) == 4
        @test same_contours(materialize_contours(state), cpu_contours)

        # The full pipelines must agree in order too.
        params = SurgeryParams(δ, 0.05, 0.3, 1e-8, 10)
        cpu_prob = ContourProblem(EulerKernel(), domain, deepcopy(contours))
        surgery!(cpu_prob, params)
        pipeline_state = DeviceContourState(deepcopy(contours), CPU())
        ContourDynamics._device_surgery_pipeline!(pipeline_state, params, domain, CPU())
        @test length(cpu_prob.contours) == 4
        @test same_contours(materialize_contours(pipeline_state), cpu_prob.contours;
                            rtol=1e-10, atol=1e-10)
    end

    @testset "Float32 end-to-end surgery parity" begin
        δ = 0.01f0
        contours = [dumbbell(0f0, 0.004f0; T=Float32), dumbbell(6f0, 0.001f0; T=Float32)]
        params = SurgeryParams(δ, 0.05f0, 0.3f0, 1f-8, 10)
        cpu_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours))
        surgery!(cpu_prob, params)
        state = DeviceContourState(deepcopy(contours), CPU())
        ContourDynamics._device_surgery_pipeline!(state, params, UnboundedDomain(), CPU())
        @test length(cpu_prob.contours) == 4
        @test same_contours(materialize_contours(state), cpu_prob.contours;
                            rtol=1f-4, atol=1f-5)
    end

    @testset "Obtuse-corner demotion keeps corners at degenerate segments" begin
        # Node 3 is a right-angle corner and stays. Node 1 is a corner whose
        # previous segment has length 1e-20: below
        # eps, so its angle is undefined and the CPU keeps the corner even
        # though the dot product is (barely) negative.
        nodes = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0),
                 SVector(0.0, 1.0), SVector(-1e-20, 0.0)]
        corners = [true, false, true, false, false]
        c = PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)

        cpu_contours = [deepcopy(c)]
        ContourDynamics._demote_obtuse_corners!(cpu_contours)
        state = DeviceContourState([deepcopy(c)], CPU())
        ContourDynamics._demote_obtuse_corners!(state, CPU())
        actual = only(materialize_contours(state))

        @test cpu_contours[1].corners == [true, false, true, false, false]
        @test actual.corners == cpu_contours[1].corners
    end

    @testset "Fixed-corner remesh of a zero-length span matches CPU" begin
        base = circular_patch(1.0, 40, 1.0)
        nodes = copy(base.nodes)
        # Two coincident corners: the span between them has zero length.
        insert!(nodes, 11, nodes[10])
        corners = falses(length(nodes))
        corners[10] = true
        corners[11] = true
        corners[31] = true
        c = PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)
        params = SurgeryParams(0.005, 0.04, 0.16, 1e-6, 10)

        cpu_c = remesh(c, params; _density_sources=[c])
        dev_c = only(ContourDynamics._device_remesh_contours([c], params, CPU()))
        @test nnodes(dev_c) == nnodes(cpu_c)
        @test dev_c.corners == cpu_c.corners
        @test all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-10, rtol=1e-10))
    end

    @testset "Closed remesh of a zero-perimeter contour matches CPU" begin
        p = SVector(0.3, -0.2)
        c = PVContour([p, p, p, p], 1.0)
        params = SurgeryParams(0.005, 0.04, 0.16, 1e-6, 10)
        cpu_c = remesh(c, params)
        dev_c = only(ContourDynamics._device_remesh_contours([c], params, CPU()))
        @test nnodes(cpu_c) == 4
        @test nnodes(dev_c) == nnodes(cpu_c)
        @test dev_c.corners == cpu_c.corners
        @test dev_c.nodes == cpu_c.nodes
    end

    @testset "Segmented scan helpers match serial per-contour reductions" begin
        # Ragged layout with an empty contour in the middle and at the end.
        lengths = [3, 0, 5, 1, 0]
        n = sum(lengths)
        offsets = zeros(Int, 5)
        total = zeros(Int, 1)
        ContourDynamics.@_ka_launch CPU() 5 ContourDynamics._prefix_lengths_kernel!(
            offsets, total, lengths, 5)
        @test offsets == [1, 4, 4, 9, 10]
        @test total == [9]
        contour_of_node = zeros(Int, n)
        ContourDynamics.@_ka_launch CPU() 5 ContourDynamics._out_node_contour_kernel!(
            contour_of_node, offsets, lengths, 5)
        @test contour_of_node == [1, 1, 1, 3, 3, 3, 3, 3, 4]

        vals = (collect(1.0:n), [0.5, 2.0, 1.5, 3.0, 0.25, 7.0, 1.0, 2.5, 4.0])
        sums, maxes = ContourDynamics._device_segmented_scan(
            vals, contour_of_node, n, (+, max), CPU())
        expect_sum = [1, 3, 6, 4, 9, 15, 22, 30, 9]
        expect_max = [0.5, 2.0, 2.0, 3.0, 3.0, 7.0, 7.0, 7.0, 4.0]
        @test sums == expect_sum
        @test maxes == expect_max
        @test vals[1] == collect(1.0:n)   # inputs are left intact

        # Single-node input returns the inputs themselves.
        one_val = ([2.0],)
        @test ContourDynamics._device_segmented_scan(one_val, [1], 1, (+,), CPU())[1] === one_val[1]
    end

    @testset "Filament removal with nothing flagged leaves the state untouched" begin
        contours = [circular_patch(1.0, 48, 1.0), circular_patch(0.7, 40, -1.0)]
        state = DeviceContourState(contours, CPU())
        params = SurgeryParams(0.001, 0.005, 0.1, 1e-4, 10)
        x_before = state.x
        ContourDynamics._device_remove_filaments!(state, params, CPU())
        @test state.x === x_before
        @test length(state.lengths) == 2
    end

    @testset "Multi-layer device stall warnings name the layer" begin
        # The device loop must receive the same per-layer label as the CPU
        # path; check the keyword plumbing without triggering a stall.
        layers = ([circular_patch(0.5, 32, 1.0)], [circular_patch(0.3, 24, -0.7)])
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)
        states = ntuple(i -> DeviceContourState(deepcopy(layers[i]), CPU()), 2)
        @test_logs min_level=Logging.Warn ContourDynamics._device_multilayer_surgery!(
            states, params, UnboundedDomain(), CPU())
        @test_logs min_level=Logging.Warn ContourDynamics._device_surgery_pipeline!(
            states[1], params, UnboundedDomain(), CPU(); layer_label=" layer 1")
    end
end
