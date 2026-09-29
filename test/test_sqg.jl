using Test
using ContourDynamics
using LinearAlgebra
using SpecialFunctions
using StaticArrays

function _sqg_gl5_integral(f, a, b, panels)
    nodes, weights = ContourDynamics._gl5_nodes_weights(Float64)
    total = 0.0
    for panel in 1:panels
        left = a + (b - a) * (panel - 1) / panels
        right = a + (b - a) * panel / panels
        midpoint = (left + right) / 2
        half_width = (right - left) / 2
        for q in eachindex(nodes)
            total += half_width * weights[q] * f(midpoint + half_width * nodes[q])
        end
    end
    return total
end

function _sqg_rankine_energy(R, jump, δ)
    # Independent Fourier-Bessel evaluation of
    # H = 1/2 ∫ θ Λ_δ^(-1) θ dA, where the softened Green function has
    # Fourier multiplier exp(-δ*|k|)/|k|.
    integrand(k) = exp(-δ * k) * (besselj1(R * k) / k)^2
    cutoff = 30 / δ
    return π * jump^2 * R^2 * _sqg_gl5_integral(integrand, 0.0, cutoff, 400)
end

function _sqg_rankine_speed(R, jump, δ)
    integrand(phi) = cos(phi) / sqrt(2R^2 * (1 - cos(phi)) + δ^2)
    return jump * R / (2π) * _sqg_gl5_integral(integrand, 0.0, 2π, 400)
end

function _sqg_polygon_fourier_integral(c, kx, ky)
    k2 = kx^2 + ky^2
    value = 0.0 + 0.0im
    for i in eachindex(c.nodes)
        a = c.nodes[i]
        ds = ContourDynamics.next_node(c, i) - a
        phase = kx * a[1] + ky * a[2]
        k_dot_ds = kx * ds[1] + ky * ds[2]
        segment_average = cis(-(phase + k_dot_ds / 2)) * sinc(k_dot_ds / (2π))
        value += im * (kx * ds[2] - ky * ds[1]) / k2 * segment_average
    end
    return value
end

function _periodic_sqg_fourier_energy(domain, contours, δ, modes)
    area = 4 * domain.Lx * domain.Ly
    result = 0.0
    for m in -modes:modes, n in -modes:modes
        (m == 0 && n == 0) && continue
        kx = π * m / domain.Lx
        ky = π * n / domain.Ly
        k = hypot(kx, ky)
        theta_hat = sum(c.pv * _sqg_polygon_fourier_integral(c, kx, ky)
                        for c in contours)
        result += exp(-δ * k) / k * abs2(theta_hat) / (2 * area)
    end
    return result
end

@testset "Surface quasi-geostrophic verification" begin
    @testset "far-field straight panels retain relative accuracy" begin
        x = SVector(1.0e14, 0.0)
        a = SVector(0.0, 0.0)
        b = SVector(1.0, 0.0)
        δ = 0.1
        reference = setprecision(256) do
            xb = BigFloat("1e14")
            δb = BigFloat("0.1")
            Float64((asinh(xb / δb) - asinh((xb - 1) / δb)) /
                    (2 * big(π)))
        end

        direct = segment_velocity(SQGKernel(δ), UnboundedDomain(), x, a, b)
        @test direct[1] ≈ reference rtol=1e-12
        @test iszero(direct[2])

        device_x, device_y = ContourDynamics._curved_sqg_contribution_scalar(
            x[1], x[2], a[1], a[2], b[1], b[2], 1.0, 0.0, 0.0,
            δ, 1 / (2π))
        @test device_x ≈ reference rtol=1e-12
        @test iszero(device_y)

        # Exercise the distinct straight-panel branch in the periodic KA
        # kernel. A vanishing splitting parameter, no images or modes, and a
        # vast cell make every periodic correction term (the central
        # erf(αr)/r ~ α and the zero-mode constant ~ 1/(αA)) negligible, which
        # isolates the free-space contribution.
        segment = ContourDynamics.SegmentData(
            [a[1]], [a[2]], [b[1]], [b[2]], [1.0], [0.0], [0.0])
        cache = EwaldCache(1.0e-30, Float64[], Float64[], zeros(0, 0), 0,
                           zeros(0, 0))
        periodic_x = zeros(1)
        periodic_y = zeros(1)
        ContourDynamics._ka_periodic_sqg_velocity!(
            periodic_x, periodic_y, [x[1]], [x[2]], segment,
            PeriodicDomain(1.0e30, 1.0e30), cache, δ, CPU())
        @test periodic_x[1] ≈ reference rtol=1e-12
        @test iszero(periodic_y[1])
    end

    @testset "regularized panel endpoints remain finite and accurate" begin
        x = SVector(1.0, 0.0)
        a = SVector(0.0, 0.0)
        b = SVector(1.0, 0.0)

        for δ in (1.0e-12, 1.0e-16)
            reference = asinh(inv(δ)) / (2π)

            direct = segment_velocity(SQGKernel(δ), UnboundedDomain(), x, a, b)
            @test isfinite(direct[1])
            @test direct[1] ≈ reference rtol=10eps(Float64)
            @test iszero(direct[2])

            device_x, device_y = ContourDynamics._curved_sqg_contribution_scalar(
                x[1], x[2], a[1], a[2], b[1], b[2], 1.0, 0.0, 0.0,
                δ, 1 / (2π))
            @test isfinite(device_x)
            @test device_x ≈ reference rtol=10eps(Float64)
            @test iszero(device_y)
        end
    end

    @testset "regularized Rankine patch" begin
        R, jump, δ = 0.8, 1.2, 0.3
        exact_energy = _sqg_rankine_energy(R, jump, δ)
        exact_speed = _sqg_rankine_speed(R, jump, δ)
        energy_errors = Float64[]

        for n in (64, 128)
            contour = circular_patch(R, n, jump)
            prob = ContourProblem(SQGKernel(δ), UnboundedDomain(), [contour])
            push!(energy_errors, abs(energy(prob) - exact_energy))
            n == 128 && @test velocity(prob, contour.nodes[1]) ≈
                              SVector(0.0, exact_speed) atol=2e-8
        end

        @test energy_errors[1] / energy_errors[2] > 3.9
        @test energy_errors[2] / exact_energy < 7e-4

        prob = ContourProblem(SQGKernel(δ), UnboundedDomain(),
                              [circular_patch(R, 128, jump)])
        @test energy(prob) > 0
        @test ContourDynamics._ka_energy(prob, ContourDynamics.CPU()) ≈
              energy(prob) rtol=2e-13
    end

    @testset "periodic energy matches independent Fourier inversion" begin
        domain = PeriodicDomain(1.7, 1.2)
        δ = 0.2
        contours = [
            circular_patch(0.24, 24, 0.9; cx=-0.18, cy=0.07),
            circular_patch(0.18, 20, -0.55; cx=0.42, cy=-0.16),
        ]
        clear_ewald_cache!()
        setup_ewald_cache!(domain, SQGKernel(δ); n_fourier=12, n_images=12)
        prob = ContourProblem(SQGKernel(δ), domain, deepcopy(contours))
        reference = _periodic_sqg_fourier_energy(domain, contours, δ, 64)

        # The softened real-image correction decays algebraically; this
        # tolerance includes the configured finite 12-image truncation.
        @test energy(prob) ≈ reference rtol=1e-4
        @test ContourDynamics._ka_energy(prob, ContourDynamics.CPU()) ≈
              reference rtol=1e-4
        clear_ewald_cache!()
    end

    @testset "curved panels stay accurate for a target at their endpoint" begin
        # Adjacent panels place the target at a panel endpoint. A nearly
        # straight cubic must reproduce the analytic straight panel there even
        # when ds ≫ δ, and must not jump at the straightness threshold.
        inv2pi = 1 / (2π)
        for (L, δ) in ((0.2, 0.01), (0.02, 2e-4))
            straight = ContourDynamics._curved_sqg_contribution_scalar(
                0.0, 0.0, 0.0, 0.0, L, 0.0, 1.0, 0.0, 0.0, δ, inv2pi)
            @test straight[1] ≈ inv2pi * asinh(L / δ) rtol=1e-12
            for κL in (0.99 * sqrt(eps()), 1.01 * sqrt(eps()), 1e-6, 1e-4)
                curved = ContourDynamics._curved_sqg_contribution_scalar(
                    0.0, 0.0, 0.0, 0.0, L, 0.0, 1.0, κL / L, κL / L, δ, inv2pi)
                @test curved[1] ≈ straight[1] rtol=10κL + 1e-12
            end
        end
    end
end
