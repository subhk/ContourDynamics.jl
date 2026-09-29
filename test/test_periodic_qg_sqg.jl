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

        setup_ewald_cache!(generic_domain, EulerKernel();
                           n_fourier=1, n_images=0)
        generic_cache = ContourDynamics._get_ewald_cache(
            generic_domain, EulerKernel())
        @test length(generic_cache.kx) == 3
        @test length(generic_cache.ky) == 3
        @test generic_cache.n_images == 0
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
                    coeff = if kernel isa EulerKernel
                        1 / (area * k2)
                    elseif kernel isa QGKernel
                        1 / (area * (k2 + kappa2))
                    else
                        # Fourier transform of 1/(2π√(r²+δ²)) in two
                        # dimensions is exp(-δ|k|)/|k|.
                        exp(-kernel.δ * sqrt(k2)) / (area * sqrt(k2))
                    end
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

    function regularized_sqg_energy_reference(c, δ)
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
                r_δ = sqrt(dx^2 + dy^2 + δ^2)
                phi = r_δ - δ * log(δ + r_δ)
                quad += g_weights[qi] * g_weights[qj] * phi
            end
            total += quad / 4 * dot_ds
        end
        return -(c.pv^2 / (4π)) * total
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
        δ = 0.01
        c = circular_patch(0.1, N, 1.0)
        prob_u = ContourProblem(SQGKernel(δ), UnboundedDomain(), [c])
        prob_p = ContourProblem(SQGKernel(δ), PeriodicDomain(10.0, 10.0), [c])

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
        v_fourier = periodic_fourier_velocity(kernel, domain, contours, x; modes=80)

        @test v_ewald ≈ v_images rtol=2e-3
        @test v_ewald ≈ v_fourier rtol=5e-6
    end

    @testset "SQG periodic energy potential matches velocity kernel" begin
        domain = PeriodicDomain(1.7, 1.2)
        kernel = SQGKernel(0.2)
        cache = build_ewald_cache(domain, kernel; n_fourier=32, n_images=5)
        r = SVector(0.37, -0.29)
        origin = zero(r)

        phi(rv) = ContourDynamics._sqg_periodic_energy_potential_scalar(
            rv[1], rv[2], cache.α, domain.Lx, domain.Ly, kernel.δ,
            cache.n_images, cache.dkx, cache.dky, cache.energy_cos)
        h = 1e-3
        ex = SVector(h, 0.0)
        ey = SVector(0.0, h)
        laplacian_phi = (phi(r + ex) + phi(r - ex) + phi(r + ey) + phi(r - ey) -
                         4 * phi(r)) / h^2

        G = inv(2π * sqrt(sum(abs2, r) + kernel.δ^2)) +
            ContourDynamics._periodic_green_correction(
                kernel, domain, cache, r, origin)
        @test laplacian_phi ≈ 4π * G rtol=2e-6
    end

    @testset "SQG unbounded energy uses regularized potential" begin
        δ = 0.35
        c = straight_contour([
            SVector(-0.7, -0.4),
            SVector(0.8, -0.3),
            SVector(0.6, 0.5),
            SVector(-0.6, 0.7),
        ], 1.0)
        prob = ContourProblem(SQGKernel(δ), UnboundedDomain(), [c])

        @test energy(prob) ≈ regularized_sqg_energy_reference(c, δ) rtol=5e-4
    end

    @testset "SQG periodic energy" begin
        N = 32
        δ = 0.01
        c = circular_patch(0.1, N, 1.0)
        prob_u = ContourProblem(SQGKernel(δ), UnboundedDomain(), [c])
        prob_p = ContourProblem(SQGKernel(δ), PeriodicDomain(10.0, 10.0), [c])

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

    @testset "periodic QG velocity matches explicit periodic copies" begin
        # Short deformation radii sum the kernel over images directly; longer
        # ones use the Ewald-split correction. Both must reproduce the
        # unbounded solver applied to enough explicit periodic copies (the
        # reference's K₀ approximation limits the Ewald comparison to ~1e-8).
        domain = PeriodicDomain(Float64(π))
        L = domain.Lx
        for (Ld, rings, tol) in ((0.1, 1, 1e-12), (1.0, 8, 1e-7))
            clear_ewald_cache!()
            kernel = QGKernel(Ld)
            c = circular_patch(0.5, 24, 1.0; cx=0.4, cy=-0.3)
            prob = ContourProblem(kernel, domain, [c])
            vel = zeros(SVector{2,Float64}, nnodes(c))
            velocity!(vel, prob)
            copies = [PVContour([p + SVector(2L * px, 2L * py) for p in c.nodes], c.pv)
                      for px in -rings:rings for py in -rings:rings]
            free = ContourProblem(kernel, UnboundedDomain(), copies)
            for k in 1:6:nnodes(c)
                reference = velocity(free, c.nodes[k])
                @test norm(vel[k] - reference) <= tol * norm(reference)
            end
        end
        clear_ewald_cache!()
    end

    @testset "periodic SQG spanning velocity does not depend on the Ewald split" begin
        # Spanning contours with Σ pv·wrap ≠ 0 feel the k = 0 content of the
        # real-space sum, which the zero-mean inversion must remove.
        domain = PeriodicDomain(Float64(π), 2.0)
        kernel = SQGKernel(0.05)
        area = 4 * domain.Lx * domain.Ly
        contours = [PVContour([SVector(p[1], p[2] + 0.2 * sin(p[1])) for p in c.nodes],
                              c.pv, c.wrap)
                    for c in beta_staircase(1.0, domain, 2; nodes_per_contour=32)]
        function cache_with_alpha(scale)
            base = build_ewald_cache(domain, kernel; n_fourier=24, n_images=6)
            α = base.α * scale
            coeffs = [iszero(kx^2 + ky^2) ? 0.0 :
                      ContourDynamics._ewald_fourier_coefficient(kernel, kx^2 + ky^2, α, area)
                      for kx in base.kx, ky in base.ky]
            return EwaldCache(α, base.kx, base.ky, coeffs, 6, zeros(0, 0))
        end
        velocities = map((1.0, 0.5)) do scale
            clear_ewald_cache!()
            ContourDynamics._store_ewald!(domain, kernel, cache_with_alpha(scale))
            prob = ContourProblem(kernel, domain, deepcopy(contours))
            vel = zeros(SVector{2,Float64}, total_nodes(prob))
            velocity!(vel, prob)
            vel
        end
        @test maximum(norm.(velocities[1] .- velocities[2])) < 1e-12
        clear_ewald_cache!()
    end

    @testset "periodic SQG energy is exact at half-period separations" begin
        function polygon_transform(nodes, kx, ky)
            k2 = kx^2 + ky^2
            s = 0.0 + 0.0im
            n = length(nodes)
            for j in 1:n
                a = nodes[j]
                b = nodes[mod1(j + 1, n)]
                β = (kx * (b[1] - a[1]) + ky * (b[2] - a[2])) / 2
                sincβ = abs(β) < 1e-12 ? 1.0 : sin(β) / β
                s += (kx * (b[2] - a[2]) - ky * (b[1] - a[1])) *
                     cis(-(kx * (a[1] + b[1]) + ky * (a[2] + b[2])) / 2) * sincβ
            end
            return im * s / k2
        end
        function spectral_energy(cs, L, δ, K)
            area = 4L^2
            E = 0.0
            for m in -K:K, n in -K:K
                (m == 0 && n == 0) && continue
                kx, ky = π * m / L, π * n / L
                k = hypot(kx, ky)
                q = sum(c.pv * polygon_transform(c.nodes, kx, ky) for c in cs)
                E += area / 2 * abs2(q / area) * exp(-δ * k) / k
            end
            return E
        end
        clear_ewald_cache!()
        L = Float64(π)
        δ = 0.05
        for d in (0.95L, L)
            cs = [circular_patch(0.15, 32, 1.0; cx=-d / 2), circular_patch(0.15, 32, 1.0; cx=d / 2)]
            prob = ContourProblem(SQGKernel(δ), PeriodicDomain(L), cs)
            reference = spectral_energy(cs, L, δ, 200)
            @test energy(prob) ≈ reference rtol=1e-7
            @test ContourDynamics._ka_energy(prob, ContourDynamics.CPU()) ≈ energy(prob) rtol=1e-12
        end
        clear_ewald_cache!()
    end

    @testset "multi-layer Ewald caches can be configured and stay configured" begin
        clear_ewald_cache!()
        domain = PeriodicDomain(2.0)
        F = 0.5
        kernel = MultiLayerQGKernel(SVector(1 / sqrt(2F)), SMatrix{2,2}(-F, F, F, -F))
        setup_ewald_cache!(domain, kernel; n_fourier=12, n_images=3)
        baroclinic = QGKernel(1 / sqrt(2F))
        @test length(ContourDynamics._get_ewald_cache(domain, EulerKernel()).kx) == 25
        @test length(ContourDynamics._get_ewald_cache(domain, baroclinic).kx) == 25
        # Automatically built caches are bounded; configured ones are not evicted.
        for i in 1:(ContourDynamics._EWALD_CACHE_MAX + 5)
            ContourDynamics._get_ewald_cache(domain, QGKernel(0.5 + i / 100))
        end
        @test ContourDynamics._get_ewald_cache(domain, baroclinic).n_images == 3
        @test ContourDynamics._get_ewald_cache(domain, EulerKernel()).n_images == 3
        clear_ewald_cache!()
    end

    @testset "Ewald cosine tables reproduce the full Fourier sums" begin
        # Every table is folded onto m, n ≥ 0 and summed by the Chebyshev
        # recurrence; compare with the direct sum over the full kx × ky grid,
        # including separations beyond the central cell.
        direct(c, kx, ky, r) = sum(c[i, j] * cos(kx[i] * r[1] + ky[j] * r[2])
                                   for i in eachindex(kx), j in eachindex(ky))
        folded(w, cache, r) = ContourDynamics._ewald_cosine_sum(
            w, cache.dkx, cache.dky, r[1], r[2])
        points = (SVector(0.3, -0.2), SVector(-1.6, 1.1), SVector(3.3, -2.3), SVector(0.0, 0.0))
        domain = PeriodicDomain(1.7, 1.2)
        for kernel in (EulerKernel(), QGKernel(0.8), SQGKernel(0.2))
            cache = build_ewald_cache(domain, kernel; n_fourier=12, n_images=1)
            for (full, w) in ((cache.fourier_coeffs, cache.fourier_cos),
                              (cache.corr_coeffs, cache.corr_cos),
                              (cache.energy_coeffs, cache.energy_cos))
                isempty(full) && continue
                for r in points
                    @test folded(w, cache, r) ≈ direct(full, cache.kx, cache.ky, r) atol=1e-14 * sum(abs, full)
                end
            end
        end

        # Grids with different kx and ky extents fold the same way.
        kx = [π * m / 1.7 for m in -3:3]
        ky = [π * n / 1.2 for n in -5:5]
        c = [iszero(a^2 + b^2) ? 0.0 : exp(-(a^2 + b^2) / 8) / (a^2 + b^2) for a in kx, b in ky]
        cache = EwaldCache(1.0, kx, ky, c, 1, zeros(0, 0))
        for r in points
            @test folded(cache.fourier_cos, cache, r) ≈ direct(c, kx, ky, r) atol=1e-14 * sum(abs, c)
        end
    end

    @testset "EwaldCache rejects tables it cannot fold" begin
        kx = [π * m for m in -2:2]
        c = [iszero(a^2 + b^2) ? 0.0 : 1 / (a^2 + b^2) for a in kx, b in kx]
        @test EwaldCache(1.0, kx, kx, c, 1, zeros(0, 0)) isa EwaldCache
        @test_throws ArgumentError EwaldCache(1.0, kx[1:4], kx, c[1:4, :], 1, zeros(0, 0))
        @test_throws ArgumentError EwaldCache(1.0, kx .^ 3 ./ π^2, kx, c, 1, zeros(0, 0))
        @test_throws DimensionMismatch EwaldCache(1.0, kx, kx, c[:, 1:4], 1, zeros(0, 0))
        # Even under k → -k (a valid cos(k·r) series) but not under kx → -kx
        # alone, so no cos(kx·x)cos(ky·y) table represents it.
        anisotropic = [c[i, j] * (1 + sign(kx[i] * kx[j]) / 10) for i in 1:5, j in 1:5]
        @test_throws ArgumentError EwaldCache(1.0, kx, kx, anisotropic, 1, zeros(0, 0))

        # A hand-built cache without an energy or QG correction table fails
        # loudly instead of dropping that Fourier part.
        clear_ewald_cache!()
        domain = PeriodicDomain(2.0)
        base = build_ewald_cache(domain, EulerKernel())
        partial_cache = EwaldCache(base.α, base.kx, base.ky, base.fourier_coeffs,
                                   base.n_images, zeros(0, 0))
        ContourDynamics._store_ewald!(domain, EulerKernel(), partial_cache)
        prob = ContourProblem(EulerKernel(), domain, [circular_patch(0.4, 16, 1.0)])
        @test all(v -> all(isfinite, v), velocity!(zeros(SVector{2,Float64}, 16), prob))
        @test_throws ArgumentError energy(prob)
        @test_throws ArgumentError ContourDynamics._ka_energy(prob, CPU())
        qg = QGKernel(1.0)
        ContourDynamics._store_ewald!(domain, qg, partial_cache)
        qprob = ContourProblem(qg, domain, [circular_patch(0.4, 16, 1.0)])
        @test_throws ArgumentError velocity(qprob, SVector(0.1, 0.2))
        @test_throws ArgumentError ContourDynamics._ka_velocity!(
            zeros(SVector{2,Float64}, 16), qprob, CPU())
        clear_ewald_cache!()
    end
end
