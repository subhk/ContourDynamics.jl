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
            @test spacing >= params.mu * 0.9
        end
    end

    @testset "Spatial Index" begin
        # Build spatial index and verify nodes are in correct bins
        nodes = [SVector(0.5, 0.5), SVector(1.5, 0.5), SVector(0.5, 1.5)]
        c = PVContour(nodes, 1.0)
        delta = 1.0
        idx = ContourDynamics.build_spatial_index([c], delta)
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

        # Finer delta produces more bins
        idx2 = ContourDynamics.build_spatial_index([c], 0.5)
        @test length(idx2.bins) >= length(idx.bins)
    end

    @testset "Reconnection: Merge Two Contours" begin
        # Two contours with same PV whose boundaries are within delta → should merge
        delta = 0.1
        mu = 0.5
        # Two squares very close to each other (gap < delta)
        c1 = PVContour([
            SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)
        ], 1.0)
        c2 = PVContour([
            SVector(1.05, 0.0), SVector(2.0, 0.0), SVector(2.0, 1.0), SVector(1.05, 1.0)
        ], 1.0)  # gap of 0.05 < delta
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
        params = SurgeryParams(delta, mu, 1.5, 1e-6, 10)

        surgery!(prob, params)
        # Should have merged into one contour (same PV, within delta)
        @test length(prob.contours) == 1
    end

    @testset "No Reconnection: Different PV" begin
        # Two contours with different PV within delta → should NOT merge
        delta = 0.1
        mu = 0.5
        c1 = PVContour([
            SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)
        ], 1.0)
        c2 = PVContour([
            SVector(1.05, 0.0), SVector(2.0, 0.0), SVector(2.0, 1.0), SVector(1.05, 1.0)
        ], 2.0)  # different PV
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
        params = SurgeryParams(delta, mu, 1.5, 1e-6, 10)

        surgery!(prob, params)
        @test length(prob.contours) == 2  # should remain separate
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
end
