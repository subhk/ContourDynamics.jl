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
end
