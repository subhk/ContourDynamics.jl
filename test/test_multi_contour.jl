using Test, ContourDynamics, StaticArrays

@testset "Multi-Contour Interactions" begin
    @testset "Three co-rotating vortices" begin
        # Three identical vortices at 120° — should co-rotate as a system
        R = 0.3
        sep = 1.5
        pv = 1.0
        N = 32

        # Place 3 vortex patches in equilateral triangle
        centers = [SVector(sep * cos(2π*k/3), sep * sin(2π*k/3)) for k in 0:2]
        contours = [PVContour(
            [SVector(c[1] + R*cos(2π*i/N), c[2] + R*sin(2π*i/N)) for i in 0:N-1], pv
        ) for c in centers]

        prob = ContourProblem(EulerKernel(), UnboundedDomain(), contours)
        stepper = RK4Stepper(0.01, total_nodes(prob))
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)

        circ0 = circulation(prob)
        E0 = energy(prob)

        evolve!(prob, stepper, params; nsteps=20)

        # Conservation: circulation and energy should be preserved
        @test circulation(prob) ≈ circ0 rtol=1e-4
        @test energy(prob) ≈ E0 rtol=1e-3

        # All 3 contours should survive (well-separated, no merger)
        @test length(prob.contours) == 3
    end

    @testset "Opposite-sign vortex pair" begin
        # Dipole: +PV and -PV patches should translate as a pair
        R = 0.4
        N = 32
        c_pos = PVContour(
            [SVector(R*cos(2π*i/N), 0.5 + R*sin(2π*i/N)) for i in 0:N-1], 1.0)
        c_neg = PVContour(
            [SVector(R*cos(2π*i/N), -0.5 + R*sin(2π*i/N)) for i in 0:N-1], -1.0)

        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c_pos, c_neg])
        stepper = RK4Stepper(0.01, total_nodes(prob))
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)

        # Net circulation should be ~0
        @test abs(circulation(prob)) < 1e-2

        ctr0_pos = centroid(prob.contours[1])
        ctr0_neg = centroid(prob.contours[2])

        evolve!(prob, stepper, params; nsteps=20)

        # Both contours should survive
        @test length(prob.contours) == 2

        # Dipole should translate — centroids should have moved
        ctr1_pos = centroid(prob.contours[1])
        ctr1_neg = centroid(prob.contours[2])
        displacement_pos = sqrt((ctr1_pos[1] - ctr0_pos[1])^2 + (ctr1_pos[2] - ctr0_pos[2])^2)
        displacement_neg = sqrt((ctr1_neg[1] - ctr0_neg[1])^2 + (ctr1_neg[2] - ctr0_neg[2])^2)
        @test displacement_pos > 1e-4  # should have moved
        @test displacement_pos ≈ displacement_neg rtol=0.1  # move together
    end

    @testset "Mixed PV multi-contour diagnostics" begin
        # 4 contours with different PV values
        contours = PVContour{Float64}[]
        for (k, pv_val) in enumerate([1.0, -0.5, 2.0, -1.5])
            cx = 3.0 * cos(2π * k / 4)
            cy = 3.0 * sin(2π * k / 4)
            push!(contours, PVContour(
                [SVector(cx + 0.3*cos(2π*i/16), cy + 0.3*sin(2π*i/16)) for i in 0:15],
                pv_val))
        end
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), contours)

        # Diagnostics should work with mixed PV
        circ = circulation(prob)
        E = energy(prob)
        Z = enstrophy(prob)

        @test isfinite(circ)
        @test isfinite(E)
        @test isfinite(Z)
        @test Z > 0  # enstrophy always positive

        # Velocity should be finite for all nodes
        vel = zeros(SVector{2, Float64}, total_nodes(prob))
        velocity!(vel, prob)
        @test all(v -> all(isfinite, v), vel)
    end

    @testset "Surgery with multiple contours" begin
        # Multiple contours with surgery — test that surgery handles the multi-contour case
        contours = [circular_patch(0.5, 32, 1.0) for _ in 1:4]
        # Offset each contour
        for (k, c) in enumerate(contours)
            offset = SVector(2.0 * cos(2π * k / 4), 2.0 * sin(2π * k / 4))
            contours[k] = PVContour([n + offset for n in c.nodes], c.pv)
        end

        prob = ContourProblem(EulerKernel(), UnboundedDomain(), contours)
        params = SurgeryParams(0.005, 0.02, 0.3, 1e-6, 5)
        stepper = RK4Stepper(0.01, total_nodes(prob))

        circ0 = circulation(prob)
        evolve!(prob, stepper, params; nsteps=20)

        @test length(prob.contours) >= 1
        @test circulation(prob) ≈ circ0 rtol=0.05
    end
end
