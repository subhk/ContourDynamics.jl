using Test, ContourDynamics, StaticArrays, Logging

@testset "Surgery" begin
    @testset "Filament Removal" begin
        # Create a tiny contour (area < area_min) and a normal one
        tiny = PVContour([
            SVector(0.0, 0.0), SVector(0.001, 0.0), SVector(0.0005, 0.001)
        ], 1.0)
        normal = circular_patch(1.0, 64, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [normal, tiny])
        params = SurgeryParams(0.001, 0.005, 0.1, 1e-4, 10)

        surgery!(prob, params)
        @test length(prob.contours) == 1  # tiny contour removed
        @test nnodes(prob.contours[1]) > 10  # normal contour survives
    end

    @testset "Filament Removal Drops Too-Few Corner Contours" begin
        nodes = SVector{2,Float64}[
            SVector(0.0, 0.0),
            SVector(1.0, 0.0),
            SVector(1.0, 1.0),
            SVector(0.0, 1.0),
        ]
        corners = falses(length(nodes))
        corners[1] = true
        contours = [PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)]

        ContourDynamics.remove_filaments!(contours, 1e-6, 0.1)

        @test isempty(contours)
    end

    @testset "Node Redistribution via Surgery" begin
        # Contour with irregular spacing should be regularized
        nodes = [SVector(cos(θ), sin(θ)) for θ in range(0, 2π - 0.01, length=20)]
        # Add a cluster of very close nodes
        push!(nodes, SVector(cos(0.01), sin(0.01)))
        push!(nodes, SVector(cos(0.02), sin(0.02)))
        c = PVContour(nodes, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        params = SurgeryParams(0.01, 0.05, 0.5, 1e-6, 10)

        surgery!(prob, params)

        # Check spacings are within bounds after surgery
        c_out = prob.contours[1]
        for i in 1:nnodes(c_out)
            j = mod1(i + 1, nnodes(c_out))
            d = c_out.nodes[j] - c_out.nodes[i]
            spacing = sqrt(d[1]^2 + d[2]^2)
            @test spacing >= params.μ * 0.9
        end
    end

    @testset "Remesh Preserves Closed Area" begin
        nodes = SVector{2,Float64}[
            SVector(1.0, 0.0),
            SVector(0.2, 0.9),
            SVector(-0.9, 0.4),
            SVector(-0.7, -0.8),
            SVector(0.6, -0.7),
        ]
        c = PVContour(nodes, 1.0)
        params = SurgeryParams(0.001, 0.08, 0.4, 1e-8, 10)

        c_new = remesh(c, params)

        @test vortex_area(c_new) ≈ vortex_area(c) rtol=1e-12 atol=1e-12
    end

    @testset "Remesh reuses arc lengths and prepared source geometry" begin
        target = circular_patch(1.0, 48, 1.0)
        source = circular_patch(0.2, 24, 2.0; cx=1.3)
        sources = [target, source]
        params = SurgeryParams(0.001, 0.02, 0.3, 1e-8, 10)
        arc_buf = Float64[]

        prepared = ContourDynamics._prepare_density_sources(sources)
        expected = ContourDynamics._dritschel_segment_densities(target, params, sources)
        actual = ContourDynamics._dritschel_segment_densities(
            target, params, sources; _arc_buf=arc_buf, _source_data=prepared)

        @test actual ≈ expected
        @test length(arc_buf) == nnodes(target)

        remesh_buf = SVector{2,Float64}[]
        vnodes_buf = SVector{2,Float64}[]
        remesh(target, params; _buf=remesh_buf, _arc_buf=arc_buf,
               _vnodes_buf=vnodes_buf, _density_sources=sources,
               _density_source_data=prepared)
        @test length(arc_buf) == nnodes(target)
    end

    @testset "Remesh Preserves Closed Area With Fixed Corners" begin
        # Post-reconnection contours carry fixed surgery corners and route
        # through `_remesh_with_fixed_corners`. That path must also conserve
        # polygon area; otherwise every merge/split accrues unmeasured drift.
        nodes = SVector{2,Float64}[
            SVector(1.0, 0.0),
            SVector(0.2, 0.9),
            SVector(-0.9, 0.4),
            SVector(-0.7, -0.8),
            SVector(0.6, -0.7),
        ]
        corners = falses(length(nodes))
        corners[1] = true
        corners[3] = true
        c = PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)
        params = SurgeryParams(0.001, 0.08, 0.4, 1e-8, 10)

        # Confirm this fixture actually exercises the fixed-corner remesh path.
        @test any(c.corners) && !ContourDynamics.is_spanning(c)

        c_new = remesh(c, params)

        @test any(c_new.corners)  # corners survive
        @test vortex_area(c_new) ≈ vortex_area(c) rtol=1e-10 atol=1e-12
        # Area is restored by moving only free nodes; corners stay bit-exact.
        new_corner_pts = c_new.nodes[corner_indices(c_new)]
        @test SVector(1.0, 0.0) in new_corner_pts
        @test SVector(-0.9, 0.4) in new_corner_pts
    end

    @testset "Surgery Pass Conserves Area Through Cleanup Remesh" begin
        # A high-aspect ellipse develops promoted corners at its tips, routing the
        # surgery cleanup remesh through the fixed-corner path. Total enclosed
        # area must survive a full surgery! pass (guards corner-remesh drift).
        c = elliptical_patch(2.0, 0.3, 80, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        A0 = sum(abs(vortex_area(cc)) for cc in prob.contours)
        surgery!(prob, SurgeryParams(0.002, 0.02, 0.2, 1e-8, 1))
        A1 = sum(abs(vortex_area(cc)) for cc in prob.contours)
        @test A1 ≈ A0 rtol=1e-4
    end

    @testset "Spanning Proximity Warning" begin
        L = 2.0
        domain = PeriodicDomain(L)
        n = 16
        spanning = PVContour([SVector(-L + 2L * (k - 1) / n, 0.0) for k in 1:n],
                             1.0, SVector(2L, 0.0), falses(n))
        @test is_spanning(spanning)

        # Closed contour with a node ~0.005 below the spanning line (within δ).
        near = circular_patch(0.1, 12, 1.0; cx=0.0, cy=0.105)
        # Well-separated closed contour.
        far = circular_patch(0.1, 12, 1.0; cx=0.0, cy=1.0)

        @test_logs (:warn,) match_mode=:any ContourDynamics._check_spanning_proximity(
            [spanning, near], 0.05, domain)
        @test_logs min_level=Logging.Warn ContourDynamics._check_spanning_proximity(
            [spanning, far], 0.05, domain)
    end

    @testset "Dritschel Redistribution Resolves High Curvature" begin
        c = elliptical_patch(2.0, 0.5, 64, 1.0)
        params = SurgeryParams(0.001, 0.02, 0.4, 1e-8, 10)

        c_new = remesh(c, params)
        lengths = arc_lengths(c_new)
        tip_sum = 0.0
        tip_count = 0
        side_sum = 0.0
        side_count = 0
        for i in 1:nnodes(c_new)
            mid = (c_new.nodes[i] + next_node(c_new, i)) / 2
            if abs(mid[1]) > 1.5
                tip_sum += lengths[i]
                tip_count += 1
            elseif abs(mid[1]) < 0.5
                side_sum += lengths[i]
                side_count += 1
            end
        end

        @test tip_count > 0
        @test side_count > 0
        @test tip_sum / tip_count < side_sum / side_count
    end

    @testset "Dritschel Nonlocal Density Uses Nearby Curvature" begin
        target = circular_patch(1.0, 96, 1.0)
        source = circular_patch(0.08, 48, 5.0; cx=1.18, cy=0.0)
        params = SurgeryParams(0.001, 0.02, 0.4, 1e-8, 10)

        local_density = ContourDynamics._dritschel_segment_densities(target, params)
        nonlocal_density = ContourDynamics._dritschel_segment_densities(target, params, [target, source])

        mids = [(target.nodes[i] + next_node(target, i)) / 2 for i in 1:nnodes(target)]
        distances = [sum(abs2, m - SVector(1.18, 0.0)) for m in mids]
        near_idx = argmin(distances)
        far_idx = argmax(distances)

        @test nonlocal_density[near_idx] > local_density[near_idx]
        @test nonlocal_density[near_idx] > nonlocal_density[far_idx]
    end

    @testset "Cubic Interpolation Bends Toward Curved Geometry" begin
        nodes = SVector{2,Float64}[
            SVector(1.0, 0.0),
            SVector(0.0, 1.0),
            SVector(-1.0, 0.0),
            SVector(0.0, -1.0),
        ]
        c = PVContour(nodes, 1.0)
        curvatures = ContourDynamics._signed_node_curvatures(c)
        cubic_mid = ContourDynamics._cubic_segment_point(c, 1, 0.5, curvatures)
        linear_mid = (nodes[1] + nodes[2]) / 2
        circular_mid = SVector(sqrt(0.5), sqrt(0.5))

        @test cubic_mid[1] ≈ cubic_mid[2] atol=1e-14
        cubic_error = sqrt(sum(abs2, cubic_mid - circular_mid))
        linear_error = sqrt(sum(abs2, linear_mid - circular_mid))
        @test cubic_error < linear_error
    end

    @testset "Curved Euler Velocity Uses Cubic Segment Geometry" begin
        kernel = EulerKernel()
        domain = UnboundedDomain()
        a = SVector(0.0, 0.0)
        b = SVector(1.0, 0.0)
        x = SVector(0.5, 0.25)

        straight = ContourDynamics.segment_velocity(kernel, domain, x, a, b)
        zero_curvature = ContourDynamics.curved_segment_velocity(
            kernel, domain, x, a, b, 0.0, 0.0)
        curved = ContourDynamics.curved_segment_velocity(
            kernel, domain, x, a, b, 1.0, 1.0)

        @test zero_curvature ≈ straight atol=1e-14 rtol=1e-14
        @test sqrt(sum(abs2, curved - straight)) > 1e-4
        @test all(isfinite, curved)
    end

    @testset "Curved Velocity Covers All Single-Layer Kernels" begin
        a = SVector(0.0, 0.0)
        b = SVector(1.0, 0.0)
        x = SVector(0.5, 0.25)
        cases = (
            (QGKernel(1.0), UnboundedDomain(), nothing),
            (SQGKernel(0.02), UnboundedDomain(), nothing),
            (EulerKernel(), PeriodicDomain(2.0, 2.0), ContourDynamics._get_ewald_cache(PeriodicDomain(2.0, 2.0), EulerKernel())),
            (QGKernel(1.0), PeriodicDomain(2.0, 2.0), ContourDynamics._get_ewald_cache(PeriodicDomain(2.0, 2.0), QGKernel(1.0))),
            (SQGKernel(0.02), PeriodicDomain(2.0, 2.0), ContourDynamics._get_ewald_cache(PeriodicDomain(2.0, 2.0), SQGKernel(0.02))),
        )

        for (kernel, domain, cache) in cases
            straight = segment_velocity(kernel, domain, x, a, b)
            zero_curvature = ContourDynamics.curved_segment_velocity(
                kernel, domain, x, a, b, 0.0, 0.0, cache)
            curved = ContourDynamics.curved_segment_velocity(
                kernel, domain, x, a, b, 1.0, 1.0, cache)

            @test zero_curvature ≈ straight atol=1e-10 rtol=1e-10
            @test sqrt(sum(abs2, curved - straight)) > 1e-5
            @test all(isfinite, curved)
        end
    end

    @testset "Remesh Preserves Fixed Corners" begin
        nodes = SVector{2,Float64}[
            SVector(1.0, 0.0),
            SVector(0.0, 1.0),
            SVector(-1.0, 0.0),
            SVector(0.0, -1.0),
        ]
        corners = falses(length(nodes))
        corners[1] = true
        c = PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)
        params = SurgeryParams(0.001, 0.2, 0.4, 1e-8, 10)

        c_new = remesh(c, params)
        idxs = corner_indices(c_new)

        @test length(idxs) == 1
        @test c_new.nodes[idxs[1]] == nodes[1]
    end

    @testset "High Curvature Acute Nodes Become Corners" begin
        nodes = SVector{2,Float64}[
            SVector(-1.0, 0.0),
            SVector(0.0, 0.0),
            SVector(0.02, 0.1),
            SVector(0.04, 0.0),
            SVector(1.0, 0.0),
            SVector(1.0, 1.0),
            SVector(-1.0, 1.0),
        ]
        c = PVContour(nodes, 1.0)

        c_corner = ContourDynamics._promote_high_curvature_corners(c, 0.2)

        @test !isempty(corner_indices(c_corner))
        @test 3 in corner_indices(c_corner)
    end

    @testset "Obtuse Corners Return To Simple Nodes" begin
        nodes = SVector{2,Float64}[
            SVector(0.0, 0.0),
            SVector(1.0, 0.0),
            SVector(1.0, 1.0),
            SVector(-1.0, 1.0),
            SVector(-1.0, 0.0),
        ]
        corners = falses(length(nodes))
        corners[1] = true
        c = PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)

        demoted = ContourDynamics._demote_obtuse_corners(c)

        @test isempty(corner_indices(demoted))
    end

    @testset "Multi-layer Surgery Demotes Obtuse Corners" begin
        nodes = SVector{2,Float64}[
            SVector(0.0, 0.0),
            SVector(1.0, 0.0),
            SVector(1.0, 1.0),
            SVector(-1.0, 1.0),
            SVector(-1.0, 0.0),
        ]
        corners = falses(length(nodes))
        corners[1] = true
        layer1 = [PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)]
        layer2 = PVContour{Float64}[]
        coupling = SMatrix{2,2,Float64,4}(-1.0, 1.0, 1.0, -1.0)
        kernel = MultiLayerQGKernel(SVector(1 / sqrt(2.0)), coupling)
        prob = MultiLayerContourProblem(kernel, UnboundedDomain(), (layer1, layer2))
        params = SurgeryParams(0.001, 0.05, 0.5, 1e-8, 10)

        surgery!(prob, params)

        @test isempty(corner_indices(prob.layers[1][1]))
    end

    @testset "Reconnection Uses Node To Segment Contact" begin
        a1 = SVector(-1.0, 0.0)
        b1 = SVector(1.0, 0.0)
        a2 = SVector(0.0, -1.0)
        b2 = SVector(0.0, 1.0)

        # Dritschel's surgery test is not the full segment-intersection
        # distance; it checks whether a node of either segment is within δ of
        # the other straight segment.
        contact_d2 = ContourDynamics._surgery_contact_distance2(a1, b1, a2, b2)
        full_segment_d2 = ContourDynamics._segment_min_dist2(a1, b1, a2, b2)

        @test contact_d2 ≈ 1.0
        @test full_segment_d2 ≈ 0.0
    end

    @testset "Spatial Index" begin
        # Build spatial index and verify nodes are in correct bins
        nodes = [SVector(0.5, 0.5), SVector(1.5, 0.5), SVector(0.5, 1.5)]
        c = PVContour(nodes, 1.0)
        δ = 1.0
        idx = ContourDynamics.build_spatial_index([c], δ)
        @test length(idx.bins) > 0  # at least one occupied bin
        # Verify that nodes land in expected bins
        # Node (0.5, 0.5) → bin (0, 0)
        @test haskey(idx.bins, (0, 0))
        # Node (1.5, 0.5) → bin (1, 0)
        @test haskey(idx.bins, (1, 0))
        # Node (0.5, 1.5) → bin (0, 1)
        @test haskey(idx.bins, (0, 1))
        # Bins reference the correct contour (contour 1)
        @test all(ci == 1 for (ci, _) in idx.bins[(0, 0)])

        # Finer δ produces more bins
        idx2 = ContourDynamics.build_spatial_index([c], 0.5)
        @test length(idx2.bins) >= length(idx.bins)
    end

    @testset "Reconnection: Merge Two Contours" begin
        # Two contours with matching PV and matching local interior vorticity
        # whose boundaries are within δ → should merge.
        δ = 0.02
        μ = 0.1
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

        # Two resolved rectangles very close to each other (gap < δ).
        c1 = rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0)
        c2 = rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
        params = SurgeryParams(δ, μ, 0.4, 1e-6, 10)

        surgery!(prob, params)
        # Should have merged into one valid contour.
        # Corner labels may be demoted during cleanup if they become obtuse.
        @test length(prob.contours) == 1
        @test nnodes(prob.contours[1]) >= 3
    end

    @testset "Reconnection: Merge Across Periodic Seam Stays In One Frame" begin
        # Two patches touching through the x-seam are stored in different
        # periodic images. The merge must shift one contour into the other's
        # frame; otherwise the merged polygon contains a ~period-long segment
        # and its area is garbage.
        domain = PeriodicDomain(2.0, 2.0)
        δ = 0.05
        cA = circular_patch(0.3, 32, 1.0; cx=1.95, cy=0.0)
        cB = circular_patch(0.3, 32, 1.0; cx=-1.95, cy=0.0)
        contours = [cA, cB]
        A0 = abs(vortex_area(cA)) + abs(vortex_area(cB))

        idx = ContourDynamics.build_spatial_index(contours, δ, domain)
        close_pairs = ContourDynamics.find_close_segments(contours, idx, δ, domain)
        @test any(p -> p[1] != p[3], close_pairs)

        ContourDynamics.reconnect!(contours, close_pairs, domain)
        @test length(contours) == 1
        merged = contours[1]
        maxseg = maximum(hypot((next_node(merged, k) - merged.nodes[k])...)
                         for k in 1:nnodes(merged))
        @test maxseg < 1.0
        @test abs(vortex_area(merged)) ≈ A0 rtol=0.2
    end

    @testset "No Reconnection: Different PV" begin
        # Two contours with different PV within δ → should NOT merge
        δ = 0.1
        μ = 0.5
        c1 = PVContour([
            SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)
        ], 1.0)
        c2 = PVContour([
            SVector(1.05, 0.0), SVector(2.0, 0.0), SVector(2.0, 1.0), SVector(1.05, 1.0)
        ], 2.0)  # different PV
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
        params = SurgeryParams(δ, μ, 1.5, 1e-6, 10)

        surgery!(prob, params)
        @test length(prob.contours) == 2  # should remain separate
    end

    @testset "No Reconnection: Different Interior Vorticity Levels" begin
        # Dritschel's contour merger condition requires identical interior
        # vorticity, not just equal PV jumps.  These concentric contours carry
        # the same jump and are within δ, but they bound different levels.
        δ = 0.05
        outer = circular_patch(1.0, 96, 1.0)
        inner = circular_patch(0.98, 96, 1.0)
        contours = [outer, inner]

        idx = ContourDynamics.build_spatial_index(contours, δ)
        close_pairs = ContourDynamics.find_close_segments(contours, idx, δ)

        @test isempty(close_pairs)
    end

    @testset "Reconnection: Split Self-Approaching Contour" begin
        # Test the split reconnection path directly (bypassing remesh) by
        # constructing a dumbbell contour where segments on opposite sides
        # of a narrow neck are within δ of each other.
        #
        # The contour is two circles connected by a narrow bridge at y ≈ ±gap.
        # Segments near the bridge on the top path (large node index) are
        # close in space to segments on the bottom path (small node index)
        # but far apart along the contour (dist_along >> 2).
        gap = 0.002        # half-gap at neck
        δ = 0.01           # proximity threshold (2*gap = 0.004 < δ)
        N_half = 30        # nodes per semicircle

        nodes = SVector{2,Float64}[]
        # Right lobe: semicircle from (1, -1) up to (1, 1), center (1, 0)
        for k in 0:N_half
            θ = -π/2 + π * k / N_half
            push!(nodes, SVector(1.0 + cos(θ), sin(θ)))
        end
        # Top bridge: straight across from right to left at y = +gap
        for k in 1:5
            x = 1.0 - k * 2.0 / 6
            push!(nodes, SVector(x, gap))
        end
        # Left lobe: semicircle from (-1, 1) down to (-1, -1), center (-1, 0)
        for k in 0:N_half
            θ = π/2 + π * k / N_half
            push!(nodes, SVector(-1.0 + cos(θ), sin(θ)))
        end
        # Bottom bridge: straight across from left to right at y = -gap
        for k in 1:5
            x = -1.0 + k * 2.0 / 6
            push!(nodes, SVector(x, -gap))
        end

        existing_corner_node = nodes[10]
        corners = falses(length(nodes))
        corners[10] = true
        c = PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)
        contours = [c]
        area_before = abs(vortex_area(c))
        @test nnodes(c) > 20  # sanity: enough nodes

        # Call internal surgery functions directly to test the split path
        idx = ContourDynamics.build_spatial_index(contours, δ)
        close_pairs = ContourDynamics.find_close_segments(contours, idx, δ)
        @test !isempty(close_pairs)

        # All close pairs should be self-pairs (ci == cj) since there's one contour
        @test all(p -> p[1] == p[3], close_pairs)

        ContourDynamics.reconnect!(contours, close_pairs)

        # The dumbbell should split into two contours at the neck
        @test length(contours) >= 2

        # Each daughter should have valid (non-degenerate) node count
        for ci in contours
            @test nnodes(ci) >= 3
            @test !isempty(corner_indices(ci))
        end
        @test any(contours) do ci
            any(corner_indices(ci)) do k
                ci.nodes[k] == existing_corner_node
            end
        end

        # Total area of daughters should approximate the original
        area_after = sum(abs(vortex_area(ci)) for ci in contours)
        @test area_after ≈ area_before rtol=0.1
    end

    @testset "Orientation Preserved After Surgery" begin
        c = circular_patch(1.0, 64, 1.0)
        pv_before = c.pv
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        params = SurgeryParams(0.005, 0.02, 0.3, 1e-6, 10)
        surgery!(prob, params)
        # PV jump should be preserved
        @test prob.contours[1].pv == pv_before
        # Area should remain positive (CCW orientation)
        @test vortex_area(prob.contours[1]) > 0
    end

    @testset "Merger Surgery Does Not Warn On Stitch Cleanup" begin
        N = 64
        R = 0.5
        sep = 3.0 * R
        pv = 2π

        c1 = circular_patch(R, N, pv; cx=-sep / 2)
        c2 = circular_patch(R, N, pv; cx=+sep / 2)
        prob = Problem(; contours=[c1, c2], dt=0.01,
                       surgery=SurgeryParams(0.002, 0.01, 0.15, 1e-8, 10))
        circulation0 = circulation(prob)

        @test_logs min_level=Logging.Warn evolve!(prob; nsteps=20)

        @test 1 <= length(contours(prob)) <= 2
        @test total_nodes(prob) >= 120
        @test abs(circulation(prob) - circulation0) / abs(circulation0) < 5e-3
    end
end
