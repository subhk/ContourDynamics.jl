# Reproduce the circular-patch verification table from the repository root:
#   julia --project=. -e 'using Pkg; Pkg.instantiate()'
#   julia --project=. --threads=2 paper/validate.jl
# Uses only package dependencies; Float64, CPU, unbounded domains, no surgery.
using ContourDynamics, LinearAlgebra, Printf, SpecialFunctions, StaticArrays

# Independent composite midpoint quadrature, with a refinement check below.
midpoint_integral(f, a, b, n) =
    (b - a) / n * sum(f(a + (i - 0.5) * (b - a) / n) for i in 1:n)

function sqg_reference(R, q, δ, n)
    # Exact circular-boundary integral at x=(R,0).
    speed = q * R / (2π) * midpoint_integral(
        φ -> cos(φ) / sqrt(2R^2 * (1 - cos(φ)) + δ^2), 0.0, 2π, n)
    # Reduce Hδ = q²/(4π) ∬ 1/√(|x-y|²+δ²) dA_x dA_y
    # using the overlap area of two radius-R disks separated by r.
    overlap(r) = 2R^2 * acos(r / (2R)) - r / 2 * sqrt(4R^2 - r^2)
    hamiltonian = q^2 / 2 * midpoint_integral(
        r -> r * overlap(r) / sqrt(r^2 + δ^2), 0.0, 2R, n)
    return speed, hamiltonian
end

function main()
    R, q, δ = 0.8, 1.2, 0.3
    sqg_coarse = sqg_reference(R, q, δ, 16_384)
    sqg_fine = sqg_reference(R, q, δ, 32_768)
    @assert all(isapprox.(sqg_coarse, sqg_fine; rtol=1e-8, atol=1e-12))
    qg_speed = besseli(1, 1.0) * besselk(1, 1.0)
    cases = (
        # Euler H is the logarithmic Hamiltonian with reference length 1,
        # not the divergent total kinetic energy of an isolated vortex.
        ("Euler", EulerKernel(), 1.0, 1.0, 0.5, π / 16),
        ("QG", QGKernel(1.0), 1.0, 1.0, qg_speed, π * (0.5 - qg_speed)),
        ("SQG", SQGKernel(δ), R, q, sqg_fine...),
    )
    println("Julia ", VERSION, "; ContourDynamics ", pkgversion(ContourDynamics),
            "; Float64; CPU; threads=", Threads.nthreads())
    println("| Model | Nodes | Boundary velocity error (absolute) | Hamiltonian error (relative) |")
    println("|-------|------:|-----------------------------------:|-----------------------------:|")
    for (label, kernel, radius, jump, speed, hamiltonian) in cases
        errors = Tuple{Float64,Float64}[]
        for n in (64, 128)
            contour = circular_patch(radius, n, jump)
            problem = ContourProblem(kernel, UnboundedDomain(), [contour])
            velocity_error = norm(velocity(problem, SVector(radius, 0.0)) -
                                  SVector(0.0, speed))
            energy_error = abs(energy(problem) / hamiltonian - 1)
            push!(errors, (velocity_error, energy_error))
            @printf("| %s | %d | %.2e | %.2e |\n", label, n, velocity_error, energy_error)
        end
        # These are convergence checks against independent continuum solutions,
        # rather than comparisons between two implementations of the same kernel.
        @assert errors[2][1] < errors[1][1] "$label velocity failed to converge"
        @assert errors[2][2] < errors[1][2] "$label Hamiltonian failed to converge"
    end
end

main()
