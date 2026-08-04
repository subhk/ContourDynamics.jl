using Test, ContourDynamics, StaticArrays, LinearAlgebra

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
        modes = 8
        setup_ewald_cache!(domain, EulerKernel();
                           n_fourier=modes, n_images=2)
        prob = ContourProblem(EulerKernel(), domain, [contour])
        reference = periodic_fourier_energy(domain, [contour], modes)

        @test energy(prob) ≈ reference rtol=2e-9
        @test ContourDynamics._ka_energy(prob, ContourDynamics.CPU()) ≈
              reference rtol=2e-9
        clear_ewald_cache!()
    end
end
