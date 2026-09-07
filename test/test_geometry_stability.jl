using Test, ContourDynamics, StaticArrays

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
        @test ContourDynamics._flat_split_part_area2(
            flat.x, flat.y, flat.offsets, 1, 0, 0.0, 0.0, 1, 4) ≈
              2 * expected_area rtol=0 atol=10eps(Float64)
        @test ContourDynamics._flat_wrapped_split_part_area2(
            flat.x, flat.y, flat.offsets, 1, 0, 0.0, 0.0, 1, 1, 4, 4) ≈
              2 * expected_area rtol=0 atol=10eps(Float64)
        @test ContourDynamics._flat_shoelace_noise_scale(
            flat.x, flat.y, flat.offsets, flat.lengths, 1) ≈ 4.0

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
