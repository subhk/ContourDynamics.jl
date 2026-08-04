using Test, ContourDynamics, StaticArrays, LinearAlgebra

extended = get(ENV, "CONTOURDYNAMICS_EXTENDED_TESTS", "false") == "true"

@testset "Periodic QG/SQG" begin
    clear_ewald_cache!()

    @testset "Ewald truncation parameters are non-negative" begin
        domain = PeriodicDomain(2.0, 3.0)
        kernels = (EulerKernel(), QGKernel(1.2), SQGKernel(0.03))
        for kernel in kernels
            @test_throws ArgumentError build_ewald_cache(
                domain, kernel; n_fourier=-1)
            @test_throws ArgumentError build_ewald_cache(
                domain, kernel; n_images=-1)
            @test_throws ArgumentError setup_ewald_cache!(
                domain, kernel; n_fourier=-1)
            @test_throws ArgumentError setup_ewald_cache!(
                domain, kernel; n_images=-1)

            cache = build_ewald_cache(domain, kernel; n_fourier=0, n_images=0)
            @test length(cache.kx) == 1
            @test length(cache.ky) == 1
            @test cache.n_images == 0
            @test setup_ewald_cache!(
                domain, kernel; n_fourier=0, n_images=0) === nothing
        end

        generic_domain = PeriodicDomain(big"2.0", big"3.0")
        @test_throws ArgumentError setup_ewald_cache!(
            generic_domain, EulerKernel(); n_fourier=-1)
        @test_throws ArgumentError setup_ewald_cache!(
            generic_domain, EulerKernel(); n_images=-1)
    end

    straight_contour(nodes, pv) =
        PVContour(nodes, pv, zero(SVector{2,Float64}), trues(length(nodes)))

    function periodic_fourier_velocity(kernel, domain, contours, x; modes=160)
        v = zero(SVector{2,Float64})
        area = 4 * domain.Lx * domain.Ly
        kappa2 = kernel isa QGKernel ? 1 / kernel.Ld^2 : 0.0
        for c in contours
            nc = length(c.nodes)
            for i in 1:nc
                a = c.nodes[i]
                b = ContourDynamics.next_node(c, i)
                ds = b - a
                for m in -modes:modes, n in -modes:modes
                    (m == 0 && n == 0) && continue
                    kx = π * m / domain.Lx
                    ky = π * n / domain.Ly
                    k2 = kx^2 + ky^2
                    coeff = kernel isa EulerKernel ? 1 / (area * k2) :
                        1 / (area * (k2 + kappa2))
                    k_dot_ds = kx * ds[1] + ky * ds[2]
                    phase = kx * (x[1] - a[1]) + ky * (x[2] - a[2])
                    segment_average = abs(k_dot_ds) < 1e-13 ? cos(phase) :
                        (sin(phase) - sin(phase - k_dot_ds)) / k_dot_ds
                    v += c.pv * ds * coeff * segment_average
                end
            end
        end
        return v
    end

    function horizontal_spanning_fourier_velocity(kernel, domain, contours, x; modes=20_000)
        v = zero(SVector{2,Float64})
        area = 4 * domain.Lx * domain.Ly
        kappa2 = kernel isa QGKernel ? 1 / kernel.Ld^2 : 0.0
        for c in contours
            y = c.nodes[1][2]
            for n in -modes:modes
                n == 0 && continue
                ky = π * n / domain.Ly
                k2 = ky^2
                coeff = kernel isa EulerKernel ? 1 / (area * k2) :
                    1 / (area * (k2 + kappa2))
                v += c.pv * c.wrap * coeff * cos(ky * (x[2] - y))
            end
        end
        return v
    end

    function regularized_sqg_energy_reference(c, delta)
        g_nodes, g_weights = ContourDynamics._gl5_nodes_weights(Float64)
        total = 0.0
        nc = length(c.nodes)
        for i in 1:nc, j in 1:nc
            ai = c.nodes[i]
            bi = ContourDynamics.next_node(c, i)
            aj = c.nodes[j]
            bj = ContourDynamics.next_node(c, j)
            dsi = bi - ai
            dsj = bj - aj
            midi = (ai + bi) / 2
            midj = (aj + bj) / 2
            half_dsi = dsi / 2
            half_dsj = dsj / 2
            dot_ds = dsi[1] * dsj[1] + dsi[2] * dsj[2]
            quad = 0.0
            for qi in 1:5, qj in 1:5
                pi_pt = midi + g_nodes[qi] * half_dsi
                pj_pt = midj + g_nodes[qj] * half_dsj
                dx = pi_pt[1] - pj_pt[1]
                dy = pi_pt[2] - pj_pt[2]
                r_delta = sqrt(dx^2 + dy^2 + delta^2)
                phi = r_delta - delta * log(delta + r_delta)
                quad += g_weights[qi] * g_weights[qj] * phi
            end
            total += quad / 4 * dot_ds
        end
        return -(c.pv^2 / (4π)) * total / 2
    end

    @testset "QG velocity: periodic ≈ unbounded" begin
        # Small vortex in large domain — periodic image contributions negligible
        N = 32
        Ld = 2.0
        c = circular_patch(0.1, N, 1.0)
        prob_u = ContourProblem(QGKernel(Ld), UnboundedDomain(), [c])
        prob_p = ContourProblem(QGKernel(Ld), PeriodicDomain(10.0, 10.0), [c])

        vel_u = zeros(SVector{2, Float64}, N)
        vel_p = zeros(SVector{2, Float64}, N)
        velocity!(vel_u, prob_u)
        velocity!(vel_p, prob_p)

        for i in 1:N
            @test vel_p[i] ≈ vel_u[i] rtol=0.15
        end
    end

    @testset "QG periodic velocity < Euler periodic velocity" begin
        # QG screening reduces velocity at all scales relative to Euler
        N = 32
        c = circular_patch(0.1, N, 1.0)
        domain = PeriodicDomain(10.0, 10.0)

        prob_euler = ContourProblem(EulerKernel(), domain, [c])
        prob_qg = ContourProblem(QGKernel(0.5), domain, [c])

        vel_euler = zeros(SVector{2, Float64}, N)
        vel_qg = zeros(SVector{2, Float64}, N)
        velocity!(vel_euler, prob_euler)
        velocity!(vel_qg, prob_qg)

        euler_speed = sqrt(vel_euler[1][1]^2 + vel_euler[1][2]^2)
        qg_speed = sqrt(vel_qg[1][1]^2 + vel_qg[1][2]^2)
        @test qg_speed < euler_speed
    end

    @testset "Euler and QG Ewald velocities match Fourier reference" begin
        domain = PeriodicDomain(1.7, 1.2)
        contours = [
            straight_contour([
                SVector(-0.31, -0.18), SVector(-0.08, -0.22),
                SVector(0.04, 0.02), SVector(-0.18, 0.21),
                SVector(-0.38, 0.06),
            ], 1.0),
            straight_contour([
                SVector(0.52, 0.30), SVector(0.73, 0.34),
                SVector(0.68, 0.51), SVector(0.49, 0.47),
            ], -0.35),
        ]
        x = SVector(0.42, -0.47)

        for kernel in (EulerKernel(), QGKernel(0.9))
            clear_ewald_cache!()
            setup_ewald_cache!(domain, kernel; n_fourier=64, n_images=5)
            prob = ContourProblem(kernel, domain, deepcopy(contours))

            v_ewald = velocity(prob, x)
            v_fourier = periodic_fourier_velocity(kernel, domain, contours, x)

            @test norm(v_ewald - v_fourier) / norm(v_fourier) < 1e-5
        end
    end

    @testset "Spanning Euler and QG velocities match Fourier reference" begin
        domain = PeriodicDomain(1.7, 1.2)
        spanning_line(y, pv, n) = PVContour(
            [SVector(-domain.Lx + 2domain.Lx * (i - 1) / n, y) for i in 1:n],
            pv, SVector(2domain.Lx, 0.0))
        contours = [
            spanning_line(-0.36, 0.42, 8),
            spanning_line(0.43, -0.27, 8),
        ]
        x = SVector(0.37, 0.08)

        for kernel in (EulerKernel(), QGKernel(0.9))
            clear_ewald_cache!()
            setup_ewald_cache!(domain, kernel; n_fourier=64, n_images=5)
            prob = ContourProblem(kernel, domain, deepcopy(contours))

            v_ewald = velocity(prob, x)
            v_fourier = horizontal_spanning_fourier_velocity(kernel, domain, contours, x)

            @test norm(v_ewald - v_fourier) / norm(v_fourier) < 1e-5
        end
    end

    @testset "QG periodic segment kernel stays allocation-light after warm-up" begin
        domain = PeriodicDomain(5.0, 5.0)
        kernel = QGKernel(1.5)
        x = SVector(0.3, -0.2)
        a = SVector(-0.5, 0.1)
        b = SVector(0.7, 0.4)

        ContourDynamics.segment_velocity(kernel, domain, x, a, b)
        alloc = @allocated ContourDynamics.segment_velocity(kernel, domain, x, a, b)
        @test alloc <= 256
    end

    @testset "SQG velocity: periodic ≈ unbounded" begin
        N = 32
        delta = 0.01
        c = circular_patch(0.1, N, 1.0)
        prob_u = ContourProblem(SQGKernel(delta), UnboundedDomain(), [c])
        prob_p = ContourProblem(SQGKernel(delta), PeriodicDomain(10.0, 10.0), [c])

        vel_u = zeros(SVector{2, Float64}, N)
        vel_p = zeros(SVector{2, Float64}, N)
        velocity!(vel_u, prob_u)
        velocity!(vel_p, prob_p)

        for i in 1:N
            @test vel_p[i] ≈ vel_u[i] rtol=0.15
        end
    end

    @testset "SQG positive patch rotates counterclockwise" begin
        c = circular_patch(0.5, 64, 1.0)
        x = c.nodes[1]

        for domain in (UnboundedDomain(), PeriodicDomain(4.0, 4.0))
            clear_ewald_cache!()
            prob = ContourProblem(SQGKernel(0.02), domain, [c])
            v = velocity(prob, x)
            @test abs(v[1]) < 1e-10
            @test v[2] > 0
            @test energy(prob) > 0
        end
    end

    @testset "SQG Ewald velocity matches direct periodic image sum" begin
        function direct_sqg_image_velocity(kernel, domain, contours, x; n_images=32)
            v = zero(SVector{2,Float64})
            for px in -n_images:n_images, py in -n_images:n_images
                shift = SVector(2 * domain.Lx * px, 2 * domain.Ly * py)
                for c in contours
                    nc = length(c.nodes)
                    for i in 1:nc
                        a = c.nodes[i] + shift
                        b = c.nodes[mod1(i + 1, nc)] + shift
                        v += c.pv * ContourDynamics.segment_velocity(
                            kernel, UnboundedDomain(), x, a, b)
                    end
                end
            end
            return v
        end

        domain = PeriodicDomain(1.7, 1.2)
        # A material δ is essential here: δ≈0 cannot detect an Ewald split
        # that regularizes only the central real-space term instead of every
        # periodic image.
        kernel = SQGKernel(0.2)
        contours = [
            straight_contour([
                SVector(-0.31, -0.18), SVector(-0.08, -0.22),
                SVector(0.04, 0.02), SVector(-0.18, 0.21),
                SVector(-0.38, 0.06),
            ], 1.0),
            straight_contour([
                SVector(0.52, 0.30), SVector(0.73, 0.34),
                SVector(0.68, 0.51), SVector(0.49, 0.47),
            ], -0.35),
        ]
        x = SVector(0.42, -0.47)

        clear_ewald_cache!()
        setup_ewald_cache!(domain, kernel; n_fourier=64, n_images=5)
        prob = ContourProblem(kernel, domain, deepcopy(contours))

        v_ewald = velocity(prob, x)
        v_images = direct_sqg_image_velocity(kernel, domain, contours, x)

        @test v_ewald ≈ v_images rtol=2e-3
    end

    @testset "SQG periodic energy potential matches velocity kernel" begin
        domain = PeriodicDomain(1.7, 1.2)
        kernel = SQGKernel(0.2)
        cache = build_ewald_cache(domain, kernel; n_fourier=32, n_images=5)
        r = SVector(0.37, -0.29)
        origin = zero(r)

        phi(rv) = ContourDynamics._eval_sqg_periodic_energy_potential(
            rv, cache, domain, kernel.delta)
        h = 2e-4
        ex = SVector(h, 0.0)
        ey = SVector(0.0, h)
        laplacian_phi = (phi(r + ex) + phi(r - ex) + phi(r + ey) + phi(r - ey) -
                         4 * phi(r)) / h^2

        G = inv(2π * sqrt(sum(abs2, r) + kernel.delta^2)) +
            ContourDynamics._periodic_sqg_green_correction(
                kernel, domain, cache, r, origin)
        @test laplacian_phi ≈ 2π * G rtol=2e-6
    end

    @testset "SQG unbounded energy uses regularized potential" begin
        delta = 0.35
        c = straight_contour([
            SVector(-0.7, -0.4),
            SVector(0.8, -0.3),
            SVector(0.6, 0.5),
            SVector(-0.6, 0.7),
        ], 1.0)
        prob = ContourProblem(SQGKernel(delta), UnboundedDomain(), [c])

        @test energy(prob) ≈ regularized_sqg_energy_reference(c, delta) rtol=5e-4
    end

    @testset "SQG periodic energy" begin
        N = 32
        delta = 0.01
        c = circular_patch(0.1, N, 1.0)
        prob_u = ContourProblem(SQGKernel(delta), UnboundedDomain(), [c])
        prob_p = ContourProblem(SQGKernel(delta), PeriodicDomain(10.0, 10.0), [c])

        E_u = energy(prob_u)
        E_p = energy(prob_p)

        @test isfinite(E_p)
        @test E_p ≈ E_u rtol=0.15
    end

    @testset "QG periodic energy conservation" begin
        # Circular patch is an exact steady state — energy drift signals formula errors
        R = 0.5
        N_nodes = extended ? 64 : 32
        Ld = 2.0
        c = circular_patch(R, N_nodes, 1.0)
        domain = PeriodicDomain(5.0, 5.0)
        prob = ContourProblem(QGKernel(Ld), domain, [c])

        dt = 0.01
        nsteps = extended ? 100 : 20
        stepper = RK4Stepper(dt, total_nodes(prob))
        params = SurgeryParams(0.001, 0.01, 0.2, 1e-8, nsteps + 1)

        E0 = energy(prob)
        G0 = circulation(prob)

        evolve!(prob, stepper, params; nsteps=nsteps)

        E1 = energy(prob)
        G1 = circulation(prob)

        energy_tol = extended ? 1e-5 : 1e-4
        @test abs(E1 - E0) / abs(E0) < energy_tol
        @test G1 ≈ G0 rtol=1e-6
    end

    @testset "Multi-layer periodic energy" begin
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)
        kernel = MultiLayerQGKernel(Ld, coupling)

        c1 = circular_patch(0.3, 32, 1.0)
        c2 = circular_patch(0.3, 32, -1.0)
        domain = PeriodicDomain(5.0, 5.0)
        prob = MultiLayerContourProblem(kernel, domain, ([c1], [c2]))

        E = energy(prob)
        @test isfinite(E)

        # Evolve and check conservation
        dt = 0.01
        nsteps = 10
        stepper = RK4Stepper(dt, total_nodes(prob))
        params = SurgeryParams(0.001, 0.01, 0.2, 1e-8, nsteps + 1)

        evolve!(prob, stepper, params; nsteps=nsteps)

        E1 = energy(prob)
        @test isfinite(E1)
        E_scale = max(abs(E), abs(E1), eps(Float64))
        @test abs(E1 - E) / E_scale < 1e-3
    end
end
