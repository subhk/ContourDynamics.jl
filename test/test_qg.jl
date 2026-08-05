using Test, ContourDynamics, StaticArrays, LinearAlgebra, SpecialFunctions

function _qg_polygon_fourier_coefficient(c, kx, ky)
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

function _qg_periodic_fourier_energy(domain, contours, Ld, modes)
    area = 4 * domain.Lx * domain.Ly
    kappa2 = 1 / Ld^2
    gamma = sum(c.pv * vortex_area(c) for c in contours)
    result = gamma^2 / kappa2
    for m in -modes:modes, n in -modes:modes
        (m == 0 && n == 0) && continue
        kx = π * m / domain.Lx
        ky = π * n / domain.Ly
        qhat = sum(_qg_polygon_fourier_coefficient(c, kx, ky) for c in contours)
        result += abs2(qhat) / (kx * kx + ky * ky + kappa2)
    end
    return result / (2 * area)
end

function _multilayer_periodic_fourier_energy(domain, layers, coupling, H, modes)
    area = 4 * domain.Lx * domain.Ly
    result = 0.0
    for m in -modes:modes, n in -modes:modes
        iszero_mode = m == 0 && n == 0
        # The fixtures using this reference have zero circulation separately in
        # every layer, so the singular barotropic zero mode contributes nothing.
        iszero_mode && continue
        kx = π * m / domain.Lx
        ky = π * n / domain.Ly
        k2 = kx * kx + ky * ky
        qhat = SVector{length(layers),ComplexF64}(ntuple(layer ->
            sum(_qg_polygon_fourier_coefficient(c, kx, ky) for c in layers[layer]),
            length(layers)))
        physical_inverse = k2 * Matrix{Float64}(I, length(layers), length(layers)) -
                           Matrix(coupling)
        result += real(dot(qhat, Diagonal(Vector(H)) * (physical_inverse \ qhat)))
    end
    return result / (2 * area)
end

function _multilayer_periodic_fourier_velocity(domain, layers, coupling, x; modes=100)
    N = length(layers)
    area = 4 * domain.Lx * domain.Ly
    velocity_layers = [zero(SVector{2,Float64}) for _ in 1:N]
    for m in -modes:modes, n in -modes:modes
        (m == 0 && n == 0) && continue
        kx = π * m / domain.Lx
        ky = π * n / domain.Ly
        k2 = kx * kx + ky * ky
        green = inv(k2 * Matrix{Float64}(I, N, N) - Matrix(coupling)) / area
        for source in 1:N, c in layers[source]
            for i in 1:nnodes(c)
                a = c.nodes[i]
                b = ContourDynamics.next_node(c, i)
                ds = b - a
                kd = kx * ds[1] + ky * ds[2]
                phase = kx * (x[1] - a[1]) + ky * (x[2] - a[2])
                segment_average = abs(kd) < 1e-13 ? cos(phase) :
                    (sin(phase) - sin(phase - kd)) / kd
                for target in 1:N
                    velocity_layers[target] += c.pv * green[target, source] *
                                               segment_average * ds
                end
            end
        end
    end
    return tuple(velocity_layers...)
end

@testset "Single-layer and multi-layer QG verification" begin
    @testset "single-layer Rankine vortex" begin
        R, Ld = 1.0, 1.0
        exact_speed = R * besseli(1, R / Ld) * besselk(1, R / Ld)
        exact_energy = π * R^2 * Ld^2 *
                       (0.5 - besseli(1, R / Ld) * besselk(1, R / Ld))

        energy_errors = Float64[]
        for n in (64, 128)
            c = circular_patch(R, n, 1.0)
            prob = ContourProblem(QGKernel(Ld), UnboundedDomain(), [c])
            push!(energy_errors, abs(energy(prob) - exact_energy))
            n == 128 && @test velocity(prob, c.nodes[1]) ≈
                              SVector(0.0, exact_speed) atol=3e-7
        end
        @test energy_errors[1] / energy_errors[2] > 3.9
        @test energy_errors[2] / exact_energy < 6.1e-4

        prob = ContourProblem(QGKernel(Ld), UnboundedDomain(),
                              [circular_patch(R, 128, 1.0)])
        @test energy(prob) > 0
        @test ContourDynamics._ka_energy(prob, ContourDynamics.CPU()) ≈
              energy(prob) rtol=2e-13
    end

    @testset "far-field screened velocity avoids logarithmic cancellation" begin
        a = SVector(0.0, 0.0)
        b = SVector(1.0, 0.0)
        x = SVector(30.0, 0.3)
        panels = 4_000
        h = 1 / panels
        integrand(t) = besselk(0, hypot(x[1] - t, x[2]))
        simpson = integrand(0.0) + integrand(1.0)
        for i in 1:(panels - 1)
            simpson += (isodd(i) ? 4 : 2) * integrand(i * h)
        end
        reference = simpson * h / (3 * 2π)
        actual = segment_velocity(QGKernel(1.0), UnboundedDomain(), x, a, b)
        @test actual[1] ≈ reference rtol=2e-7 atol=0
        @test actual[2] == 0.0

        vx, vy = ContourDynamics._curved_qg_contribution_scalar(
            x[1], x[2], a[1], a[2], b[1], b[2], 1.0, 0.0, 0.0,
            1.0, 1 / (2π), 1 / (4π))
        @test vx ≈ reference rtol=2e-7 atol=0
        @test vy == 0.0
    end

    @testset "periodic single-layer energy includes the Helmholtz zero mode" begin
        clear_ewald_cache!()
        domain = PeriodicDomain(3.0, 2.0)
        kernel = QGKernel(1.0)
        modes = 12
        contour = circular_patch(0.5, 64, 1.0; cx=0.31, cy=-0.27)
        setup_ewald_cache!(domain, kernel; n_fourier=modes, n_images=2)
        prob = ContourProblem(kernel, domain, [contour])
        reference = _qg_periodic_fourier_energy(domain, [contour], kernel.Ld, modes)

        @test energy(prob) ≈ reference rtol=3e-9
        @test ContourDynamics._ka_energy(prob, ContourDynamics.CPU()) ≈
              reference rtol=3e-9
        clear_ewald_cache!()
    end

    @testset "unequal-depth modal inversion matches the physical-layer matrix" begin
        H = SVector(1.0, 3.0)
        F1, F2 = 0.6, 0.2
        coupling = SMatrix{2,2,Float64}(-F1, F2, F1, -F2)
        kernel = MultiLayerQGKernel(SVector(1 / sqrt(F1 + F2)), coupling, H)
        keyword_kernel = MultiLayerQGKernel(
            SVector(1 / sqrt(F1 + F2)), coupling; layer_thicknesses=H)
        inferred_kernel = MultiLayerQGKernel(SVector(1 / sqrt(F1 + F2)), coupling)

        @test kernel.layer_thicknesses == H
        @test keyword_kernel.layer_thicknesses == H
        @test inferred_kernel.layer_thicknesses ≈ H / (sum(H) / length(H))
        @test kernel.symmetric_coupling ≈ kernel.symmetric_coupling'
        @test kernel.modal_to_physical * Diagonal(kernel.eigenvalues) *
              kernel.physical_to_modal ≈ coupling
        @test kernel.physical_to_modal * kernel.modal_to_physical ≈ I

        @test_throws ArgumentError MultiLayerQGKernel(
            kernel.Ld, coupling, SVector(1.0, 1.0))
        one_way = SMatrix{2,2,Float64}(-F1, 0.0, F1, 0.0)
        @test_throws ArgumentError MultiLayerQGKernel(kernel.Ld, one_way)

        domain = PeriodicDomain(2.3, 1.7)
        straight(nodes, pv) = PVContour(
            nodes, pv, zero(SVector{2,Float64}), trues(length(nodes)))
        layers = (
            [straight([SVector(-0.42, -0.18), SVector(-0.08, -0.25),
                       SVector(0.09, 0.05), SVector(-0.19, 0.24)], 0.9)],
            [straight([SVector(0.38, 0.16), SVector(0.69, 0.12),
                       SVector(0.62, 0.43), SVector(0.35, 0.39)], -0.55)],
        )
        modes = 48
        clear_ewald_cache!()
        setup_ewald_cache!(domain, EulerKernel(); n_fourier=modes, n_images=5)
        for lam in kernel.eigenvalues
            ContourDynamics._is_barotropic_mode(kernel, lam) && continue
            setup_ewald_cache!(domain, QGKernel(1 / sqrt(abs(lam)));
                               n_fourier=modes, n_images=5)
        end
        prob = MultiLayerContourProblem(kernel, domain, deepcopy(layers))
        x = SVector(0.27, -0.41)
        expected_velocity = _multilayer_periodic_fourier_velocity(
            domain, layers, coupling, x; modes=120)
        actual_velocity = velocity(prob, x)
        for layer in 1:2
            @test norm(actual_velocity[layer] - expected_velocity[layer]) /
                  norm(expected_velocity[layer]) < 2e-5
        end

        velocity_states = ntuple(i -> DeviceContourState(deepcopy(layers[i]),
                                                         ContourDynamics.CPU()), 2)
        device_velocity = ContourDynamics._ka_multilayer_velocity_at_states(
            velocity_states, kernel, domain, x, ContourDynamics.CPU())
        for layer in 1:2
            @test device_velocity[layer] ≈ actual_velocity[layer] rtol=2e-13 atol=2e-13
        end

        inferred_prob = MultiLayerContourProblem(inferred_kernel, domain, deepcopy(layers))
        for layer in 1:2
            @test velocity(inferred_prob, x)[layer] ≈ actual_velocity[layer] rtol=2e-13
        end

        # Use circulation-neutral pairs of finer straight polygons for the
        # energy comparison. This removes the singular physical barotropic
        # zero mode and lets the reference invert the raw nonsymmetric matrix
        # directly, without using any implementation modal transforms. The
        # independent exact segment Fourier integral is not dominated by the
        # diagnostic's fixed three-point panel quadrature at the highest mode.
        energy_layers = (
            [straight(circular_patch(0.18, 24, 0.9; cx=-0.42, cy=0.12).nodes, 0.9),
             straight(circular_patch(0.18, 24, -0.9; cx=0.08, cy=-0.31).nodes, -0.9)],
            [straight(circular_patch(0.15, 20, 0.55; cx=0.47, cy=0.28).nodes, 0.55),
             straight(circular_patch(0.15, 20, -0.55; cx=0.14, cy=0.04).nodes, -0.55)],
        )
        @test all(abs(sum(c.pv * vortex_area(c) for c in layer)) < 2e-16
                  for layer in energy_layers)
        energy_prob = MultiLayerContourProblem(kernel, domain, deepcopy(energy_layers))
        reference_energy = _multilayer_periodic_fourier_energy(
            domain, energy_layers, coupling, H, modes)
        @test energy(energy_prob) ≈ reference_energy rtol=5e-7

        states = ntuple(i -> DeviceContourState(deepcopy(energy_layers[i]),
                                                ContourDynamics.CPU()), 2)
        @test ContourDynamics._ka_multilayer_energy_from_states(
            states, kernel, domain, ContourDynamics.CPU()) ≈
            reference_energy rtol=5e-7

        inferred_energy_prob = MultiLayerContourProblem(
            inferred_kernel, domain, deepcopy(energy_layers))
        @test energy(inferred_energy_prob) ≈
              reference_energy / (sum(H) / length(H)) rtol=5e-7

        # A pure weighted eigenmode must have exactly the corresponding
        # single-layer QG Hamiltonian in an unbounded domain.
        mode = findfirst(lam -> !ContourDynamics._is_barotropic_mode(kernel, lam),
                         kernel.eigenvalues)
        base = circular_patch(0.45, 48, 1.0)
        mode_layers = ntuple(layer -> [PVContour(copy(base.nodes),
            kernel.modal_to_physical[layer, mode])], 2)
        mode_prob = MultiLayerContourProblem(kernel, UnboundedDomain(), mode_layers)
        single_mode = ContourProblem(
            QGKernel(1 / sqrt(abs(kernel.eigenvalues[mode]))),
            UnboundedDomain(), [base])
        @test energy(mode_prob) ≈ energy(single_mode) rtol=2e-12
        mode_states = ntuple(i -> DeviceContourState(deepcopy(mode_layers[i]),
                                                     ContourDynamics.CPU()), 2)
        @test ContourDynamics._ka_multilayer_energy_from_states(
            mode_states, kernel, UnboundedDomain(), ContourDynamics.CPU()) ≈
            energy(single_mode) rtol=2e-12
        clear_ewald_cache!()
    end

    @testset "modal validation and eigenmode normalization" begin
        F = 0.5
        coupling = SMatrix{2,2,Float64}(-F, F, F, -F)
        kernel = MultiLayerQGKernel(SVector(1 / sqrt(2F)), coupling)
        mode = findfirst(lam -> !ContourDynamics._is_barotropic_mode(kernel, lam),
                         kernel.eigenvalues)
        base = circular_patch(0.6, 64, 1.0)
        layers = ntuple(layer -> [PVContour(copy(base.nodes),
            kernel.eigenvectors[layer, mode])], 2)
        multi = MultiLayerContourProblem(kernel, UnboundedDomain(), layers)
        single = ContourProblem(QGKernel(1 / sqrt(abs(kernel.eigenvalues[mode]))),
                                UnboundedDomain(), [base])
        @test energy(multi) ≈ energy(single) rtol=2e-12

        positive = SMatrix{2,2,Float64}(F, -F, -F, F)
        @test_throws ArgumentError MultiLayerQGKernel(SVector(1 / sqrt(2F)), positive)

        # Negative semidefiniteness and one zero eigenvalue are insufficient:
        # the physical stretching operator must leave a depth-uniform
        # streamfunction unchanged. This matrix has eigenvalues (-5, 0), but
        # its null vector is (-2, 1), not the barotropic vector (1, 1).
        rotated_null = SMatrix{2,2,Float64}(-1.0, -2.0, -2.0, -4.0)
        @test_throws ArgumentError MultiLayerQGKernel(
            SVector(1 / sqrt(5.0)), rotated_null)

        tiny_F = 1e-20
        tiny = MultiLayerQGKernel(
            SVector(1 / sqrt(2tiny_F)),
            SMatrix{2,2,Float64}(-tiny_F, tiny_F, tiny_F, -tiny_F))
        @test count(lam -> ContourDynamics._is_barotropic_mode(tiny, lam),
                    tiny.eigenvalues) == 1
    end
end
