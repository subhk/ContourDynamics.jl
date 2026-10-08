using Test, ContourDynamics, StaticArrays, LinearAlgebra

extended = get(ENV, "CONTOURDYNAMICS_EXTENDED_TESTS", "false") == "true"

# Record circulation, energy, and the first contour's area and centroid,
# evolve the problem, then record the same diagnostics again. Each caller
# states its own tolerances at the @test lines.
function conserved_after_evolve(prob, stepper, params; nsteps)
    snapshot(p) = (circulation = circulation(p), energy = energy(p),
                   area = vortex_area(p.contours[1]),
                   centroid = centroid(p.contours[1]))
    before = snapshot(prob)
    evolve!(prob, stepper, params; nsteps=nsteps)
    after = snapshot(prob)
    return (before = before, after = after)
end

@testset "2D Euler verification" begin
    @testset "straight-panel formula remains accurate in the far field" begin
        a = SVector(0.0, 0.0)
        b = SVector(1.0, 0.0)
        x = SVector(1.0e14, 0.37)

        reference = setprecision(256) do
            u = BigFloat(x[1])
            h = BigFloat(x[2])
            F(z) = z * log(z * z + h * h) - 2z + 2h * atan(z / h)
            -Float64(F(u) - F(u - 1)) / (4π)
        end

        v = segment_velocity(EulerKernel(), UnboundedDomain(), x, a, b)
        @test v[1] ≈ reference rtol=8eps(Float64)
        @test v[2] == 0.0

        inv4pi = 1 / (4π)
        vx, vy = ContourDynamics._straight_euler_contribution_scalar(
            x[1], x[2], a[1], a[2], b[1], b[2], 1.0, inv4pi)
        @test vx ≈ reference rtol=8eps(Float64)
        @test vy == 0.0
    end

    @testset "Rankine vortex sign, normalization, and curved-panel convergence" begin
        c = circular_patch(1.0, 64, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

        @test velocity(prob, SVector(0.3, 0.0)) ≈ SVector(0.0, 0.15) atol=2e-12
        @test velocity(prob, SVector(2.0, 0.0)) ≈ SVector(0.0, 0.25) atol=4e-7

        boundary_errors = Float64[]
        for n in (32, 64)
            cn = circular_patch(1.0, n, 1.0)
            pn = ContourProblem(EulerKernel(), UnboundedDomain(), [cn])
            push!(boundary_errors,
                  norm(velocity(pn, cn.nodes[1]) - SVector(0.0, 0.5)))
        end
        @test boundary_errors[1] / boundary_errors[2] > 7
        @test boundary_errors[2] < 2e-6

        # For a unit-radius, unit-vorticity disk, the renormalized Euler
        # Hamiltonian -(4π)⁻¹∫∫log|x-y| dxdy is π/16.
        @test energy(prob) ≈ π / 16 rtol=5e-6
        @test ContourDynamics._ka_energy(prob, ContourDynamics.CPU()) ≈
              energy(prob) rtol=2e-13
    end

    function polygon_fourier_coefficient(c, kx, ky)
        k2 = kx * kx + ky * ky
        coeff = 0.0im
        for i in 1:nnodes(c)
            a = c.nodes[i]
            b = ContourDynamics.next_node(c, i)
            ds = b - a
            kd = kx * ds[1] + ky * ds[2]
            segment_average = abs(kd) < 1e-13 ? 1.0 + 0.0im :
                              (1 - exp(-im * kd)) / (im * kd)
            phase = exp(-im * (kx * a[1] + ky * a[2]))
            coeff += im * (kx * ds[2] - ky * ds[1]) *
                     phase * segment_average / k2
        end
        return c.pv * coeff
    end

    function periodic_fourier_energy(domain, contours, modes)
        area = 4 * domain.Lx * domain.Ly
        result = 0.0
        for m in -modes:modes, n in -modes:modes
            (m == 0 && n == 0) && continue
            kx = π * m / domain.Lx
            ky = π * n / domain.Ly
            qhat = sum(polygon_fourier_coefficient(c, kx, ky) for c in contours)
            result += abs2(qhat) / (2 * area * (kx * kx + ky * ky))
        end
        return result
    end

    @testset "periodic Hamiltonian matches an independent Fourier sum" begin
        clear_ewald_cache!()
        domain = PeriodicDomain(3.0, 2.0)
        contour = circular_patch(0.5, 64, 1.0; cx=0.31, cy=-0.27)
        prob = ContourProblem(EulerKernel(), domain, [contour])
        # The polygon spectrum decays algebraically; extrapolate the tail
        # (∝ 1/K²) of the independent Fourier sum to its converged value.
        coarse = periodic_fourier_energy(domain, [contour], 100)
        fine = periodic_fourier_energy(domain, [contour], 200)
        reference = (4 * fine - coarse) / 3

        default_energy = energy(prob)
        @test default_energy ≈ reference rtol=2e-6
        @test ContourDynamics._ka_energy(prob, ContourDynamics.CPU()) ≈
              default_energy rtol=1e-13

        # The Ewald split converges at the default truncation: refining it
        # leaves the energy unchanged instead of adding high-k content.
        setup_ewald_cache!(domain, EulerKernel(); n_fourier=16, n_images=3)
        @test energy(prob) ≈ default_energy rtol=1e-12
        clear_ewald_cache!()
    end

    @testset "periodic Hamiltonian of a small patch is converged" begin
        # A patch much smaller than the Fourier cutoff scale used to lose most
        # of its energy to truncation of an undamped k⁻⁴ series.
        clear_ewald_cache!()
        domain = PeriodicDomain(Float64(π))
        contour = circular_patch(0.1, 64, 1.0)
        prob = ContourProblem(EulerKernel(), domain, [contour])
        coarse = periodic_fourier_energy(domain, [contour], 150)
        fine = periodic_fourier_energy(domain, [contour], 300)
        @test energy(prob) ≈ (4 * fine - coarse) / 3 rtol=2e-5
        clear_ewald_cache!()
    end
end

@testset "Euler patch dynamics" begin
    @testset "Kirchhoff Ellipse" begin
        a = 2.0
        b = 1.0
        pv = 1.0
        N_nodes = extended ? 64 : 32

        c = elliptical_patch(a, b, N_nodes, pv)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

        Omega = a * b * pv / (a + b)^2
        T_period = 2π / Omega

        nsteps = extended ? 500 : 100
        dt = T_period / nsteps
        stepper = RK4Stepper(dt, total_nodes(prob))
        params = SurgeryParams(0.001, 0.01, 0.2, 1e-8, nsteps + 1)

        initial_nodes = copy(prob.contours[1].nodes)
        d = conserved_after_evolve(prob, stepper, params; nsteps=nsteps)

        @test d.after.area ≈ d.before.area rtol=1e-3
        @test d.after.circulation ≈ d.before.circulation rtol=1e-4

        # After one full period, nodes should return near initial positions
        node_tol = extended ? 0.08 : 0.10
        for i in 1:N_nodes
            @test prob.contours[1].nodes[i] ≈ initial_nodes[i] atol=node_tol
        end

        ratio_final, _ = ellipse_moments(prob.contours[1])
        @test ratio_final ≈ a / b rtol=0.05
    end

    @testset "Vortex Merger" begin
        R = 1.0
        sep = 3.0
        pv = 1.0
        # Full merger test under the extended flag: 64 nodes, 100 steps,
        # check conservation. Otherwise a smoke test of the merger pipeline.
        N_nodes = extended ? 64 : 32
        c1_nodes = [SVector(R * cos(2π * i / N_nodes) - sep/2, R * sin(2π * i / N_nodes)) for i in 0:(N_nodes-1)]
        c2_nodes = [SVector(R * cos(2π * i / N_nodes) + sep/2, R * sin(2π * i / N_nodes)) for i in 0:(N_nodes-1)]
        c1 = PVContour(c1_nodes, pv)
        c2 = PVContour(c2_nodes, pv)

        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
        stepper = RK4Stepper(0.05, total_nodes(prob))
        params = SurgeryParams(0.005, 0.02, 0.1, 1e-4, 5)

        if extended
            d = conserved_after_evolve(prob, stepper, params; nsteps=100)
            @test d.after.circulation ≈ d.before.circulation rtol=0.05
            @test length(prob.contours) <= 2
        else
            d = conserved_after_evolve(prob, stepper, params; nsteps=10)
            @test total_nodes(prob) > 0
            @test length(prob.contours) >= 1
            @test d.after.circulation ≈ d.before.circulation rtol=0.15
        end
    end

    @testset "Conservation" begin
        @testset "Circular Patch Steady State (Euler)" begin
            R = 1.0
            pv_val = 1.0
            N_nodes = extended ? 64 : 32
            c = circular_patch(R, N_nodes, pv_val)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

            dt = 0.01
            nsteps = extended ? 500 : 50
            stepper = RK4Stepper(dt, total_nodes(prob))
            params = SurgeryParams(0.001, 0.01, 0.2, 1e-8, nsteps + 1)

            d = conserved_after_evolve(prob, stepper, params; nsteps=nsteps)
            E0, E1 = d.before.energy, d.after.energy
            c0, c1 = d.before.centroid, d.after.centroid

            energy_tol = extended ? 1e-5 : 1e-4
            @test abs(E1 - E0) / abs(E0) < energy_tol
            @test d.after.area ≈ d.before.area rtol=1e-6
            @test d.after.circulation ≈ d.before.circulation rtol=1e-6
            @test sqrt((c1[1] - c0[1])^2 + (c1[2] - c0[2])^2) < 1e-6
        end

        # The QG steady state lives here (not in test_qg.jl) because it
        # shares the conserved_after_evolve helper with the Euler case.
        @testset "Circular Patch Steady State (QG)" begin
            R = 1.0
            pv_val = 1.0
            N_nodes = extended ? 64 : 32
            c = circular_patch(R, N_nodes, pv_val)
            prob = ContourProblem(QGKernel(2.0), UnboundedDomain(), [c])

            dt = 0.01
            nsteps = extended ? 100 : 20
            stepper = RK4Stepper(dt, total_nodes(prob))
            params = SurgeryParams(0.001, 0.01, 0.2, 1e-8, nsteps + 1)

            d = conserved_after_evolve(prob, stepper, params; nsteps=nsteps)
            E0, E1 = d.before.energy, d.after.energy

            @test d.after.area ≈ d.before.area rtol=1e-6
            @test d.after.circulation ≈ d.before.circulation rtol=1e-6
            qg_energy_tol = extended ? 1e-5 : 1e-4
            @test abs(E1 - E0) / abs(E0) < qg_energy_tol
        end
    end

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

            d = conserved_after_evolve(prob, stepper, params; nsteps=20)

            # Conservation: circulation and energy should be preserved
            @test d.after.circulation ≈ d.before.circulation rtol=1e-4
            @test d.after.energy ≈ d.before.energy rtol=1e-3

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

            d = conserved_after_evolve(prob, stepper, params; nsteps=20)

            @test length(prob.contours) >= 1
            @test d.after.circulation ≈ d.before.circulation rtol=0.05
        end
    end
end
