# Precompile the kernel/domain combinations exercised by tutorials,
# examples, and common simulations.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    N = 16
    @compile_workload begin
        cE = circular_patch(0.5, N, 2π)
        probE = Problem(; contours=[cE], dt=0.01)
        evolve!(probE; nsteps=2)
        energy(probE); circulation(probE); enstrophy(probE)
        centroid(cE); ellipse_moments(cE)

        cQ = circular_patch(0.5, N, 2π)
        probQ = Problem(; contours=[cQ], dt=0.01, kernel=:qg, Ld=1.0)
        evolve!(probQ; nsteps=2)
        energy(probQ); circulation(probQ)

        cQP = circular_patch(0.3, N, 2π)
        probQP = Problem(; contours=[cQP], dt=0.005, kernel=:qg, Ld=1.0,
                         domain=:periodic, Lx=3.0, Ly=3.0)
        evolve!(probQP; nsteps=2)
        circulation(probQP)

        cQP2 = circular_patch(0.3, N, 1.0)
        probQP2 = Problem(; contours=[cQP2], dt=0.05, kernel=:qg, Ld=2.0,
                          domain=:periodic, Lx=Float64(π), Ly=Float64(π))
        evolve!(probQP2; nsteps=2)

        staircase = beta_staircase(
            1.0, PeriodicDomain(3.0), 2; nodes_per_contour=N)
        vortex = circular_patch(0.3, N, 2π)
        probBP = Problem(; contours=vcat(staircase, [vortex]), dt=0.005,
                         kernel=:beta_plane_qg, beta=1.0, Ld=1.0,
                         domain=:periodic, Lx=3.0, Ly=3.0)
        evolve!(probBP; nsteps=2)

        cS = circular_patch(0.5, N, 1.0)
        probS = Problem(; contours=[cS], dt=0.005, kernel=:sqg, δ_sqg=0.02)
        evolve!(probS; nsteps=2)

        cM = circular_patch(0.5, N, 2π)
        F = 0.5
        probM = Problem(; layers=([cM], PVContour{Float64}[]),
                        dt=0.01, kernel=:multilayer_qg,
                        Ld=[1.0], coupling=[-F F; F -F])
        evolve!(probM; nsteps=2)
        energy(probM); circulation(probM)
    end
end
