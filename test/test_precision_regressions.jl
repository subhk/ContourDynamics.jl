using Test, ContourDynamics, StaticArrays, LinearAlgebra

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
