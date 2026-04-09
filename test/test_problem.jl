using ContourDynamics
using StaticArrays
using Test

@testset "Problem API" begin
    @testset "evolve! without surgery (nothing params)" begin
        c = circular_patch(1.0, 64, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        stepper = RK4Stepper(0.01, total_nodes(prob))

        area_before = vortex_area(prob.contours[1])
        # This should work with nothing surgery params — just timestepping, no surgery
        evolve!(prob, stepper, nothing; nsteps=10)
        area_after = vortex_area(prob.contours[1])
        @test area_after ≈ area_before rtol=1e-6
    end

    @testset "Problem struct and accessors" begin
        c = circular_patch(1.0, 64, 1.0)
        cp = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        st = RK4Stepper(0.01, total_nodes(cp))
        sp = SurgeryParams(0.002, 0.01, 0.2, 1e-6, 5)

        prob = Problem(cp, st, sp)
        @test prob isa Problem
        @test prob.contour_problem === cp
        @test prob.stepper === st
        @test prob.surgery_params === sp

        # Forwarded accessors
        @test contours(prob) === cp.contours
        @test kernel(prob) === cp.kernel
        @test domain(prob) === cp.domain
        @test total_nodes(prob) == 64
        @test energy(prob) == energy(cp)
        @test circulation(prob) == circulation(cp)
        @test enstrophy(prob) == enstrophy(cp)
        @test angular_momentum(prob) == angular_momentum(cp)

        # Nothing surgery
        prob_nosurg = Problem(cp, st, nothing)
        @test prob_nosurg.surgery_params === nothing
    end

    @testset "Problem evolve!" begin
        c = circular_patch(1.0, 64, 1.0)
        cp = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        st = RK4Stepper(0.01, total_nodes(cp))
        sp = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)

        prob = Problem(cp, st, sp)
        circ_before = circulation(prob)
        evolve!(prob; nsteps=10)
        circ_after = circulation(prob)
        @test isapprox(circ_before, circ_after; rtol=1e-6)
    end

    @testset "Problem evolve! without surgery" begin
        c = circular_patch(1.0, 64, 1.0)
        cp = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        st = RK4Stepper(0.01, total_nodes(cp))

        prob = Problem(cp, st, nothing)
        area_before = vortex_area(contours(prob)[1])
        evolve!(prob; nsteps=10)
        area_after = vortex_area(contours(prob)[1])
        @test area_after ≈ area_before rtol=1e-6
    end

    @testset "Problem evolve! with callbacks" begin
        c = circular_patch(1.0, 64, 1.0)
        cp = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        st = RK4Stepper(0.01, total_nodes(cp))
        sp = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)

        prob = Problem(cp, st, sp)
        steps_seen = Int[]
        cb = (p, step) -> push!(steps_seen, step)
        evolve!(prob; nsteps=5, callbacks=[cb])
        @test steps_seen == [0, 1, 2, 3, 4, 5]
    end

    @testset "Problem factory — defaults" begin
        prob = Problem(; contours=[circular_patch(0.5, 64, 2π)], dt=0.01)
        @test prob isa Problem
        @test kernel(prob) isa EulerKernel
        @test domain(prob) isa UnboundedDomain
        @test prob.stepper isa RK4Stepper
        @test prob.stepper.dt == 0.01
        @test prob.surgery_params isa SurgeryParams
        @test total_nodes(prob) == 64
    end

    @testset "Problem factory — kernel selection" begin
        c = circular_patch(0.5, 32, 1.0)

        # QG kernel
        prob_qg = Problem(; kernel=:qg, Ld=1.0, contours=[c], dt=0.01)
        @test kernel(prob_qg) isa QGKernel
        @test kernel(prob_qg).Ld == 1.0

        # SQG kernel
        prob_sqg = Problem(; kernel=:sqg, delta_sqg=0.01, contours=[c], dt=0.01)
        @test kernel(prob_sqg) isa SQGKernel

        # Missing Ld for :qg
        @test_throws ArgumentError Problem(; kernel=:qg, contours=[c], dt=0.01)

        # Missing delta_sqg for :sqg
        @test_throws ArgumentError Problem(; kernel=:sqg, contours=[c], dt=0.01)

        # Unknown kernel
        @test_throws ArgumentError Problem(; kernel=:unknown, contours=[c], dt=0.01)
    end

    @testset "Problem factory — domain selection" begin
        c = circular_patch(0.5, 32, 1.0)

        # Periodic domain
        prob = Problem(; domain=:periodic, Lx=3.0, Ly=3.0, contours=[c], dt=0.01)
        @test domain(prob) isa PeriodicDomain
        @test domain(prob).Lx == 3.0
        @test domain(prob).Ly == 3.0

        # Missing Lx/Ly for :periodic
        @test_throws ArgumentError Problem(; domain=:periodic, contours=[c], dt=0.01)

        # Unknown domain
        @test_throws ArgumentError Problem(; domain=:unknown, contours=[c], dt=0.01)
    end

    @testset "Problem factory — stepper selection" begin
        c = circular_patch(0.5, 32, 1.0)

        prob_lf = Problem(; stepper=:leapfrog, contours=[c], dt=0.01, ra_coeff=0.1)
        @test prob_lf.stepper isa LeapfrogStepper
        @test prob_lf.stepper.ra_coeff == 0.1

        # Unknown stepper
        @test_throws ArgumentError Problem(; stepper=:unknown, contours=[c], dt=0.01)
    end

    @testset "Problem factory — surgery presets" begin
        c = circular_patch(0.5, 32, 1.0)

        prob_std = Problem(; contours=[c], dt=0.01, surgery=:standard)
        @test prob_std.surgery_params isa SurgeryParams
        @test prob_std.surgery_params.n_surgery == 5

        prob_con = Problem(; contours=[c], dt=0.01, surgery=:conservative)
        @test prob_con.surgery_params.n_surgery == 10

        prob_agg = Problem(; contours=[c], dt=0.01, surgery=:aggressive)
        @test prob_agg.surgery_params.n_surgery == 3

        prob_none = Problem(; contours=[c], dt=0.01, surgery=:none)
        @test prob_none.surgery_params === nothing

        # Direct SurgeryParams passthrough
        sp = SurgeryParams(0.001, 0.005, 0.1, 1e-6, 7)
        prob_manual = Problem(; contours=[c], dt=0.01, surgery=sp)
        @test prob_manual.surgery_params === sp

        # Unknown surgery preset
        @test_throws ArgumentError Problem(; contours=[c], dt=0.01, surgery=:unknown)
    end

    @testset "Problem factory — device" begin
        c = circular_patch(0.5, 32, 1.0)

        prob = Problem(; contours=[c], dt=0.01, dev=:cpu)
        @test prob.contour_problem.dev === CPU()

        # GPU with EulerKernel+UnboundedDomain constructs successfully
        # (error only at velocity! time when CUDA is not loaded)
        prob_gpu = Problem(; contours=[c], dt=0.01, dev=:gpu)
        @test prob_gpu.contour_problem.dev === GPU()

        # GPU with unsupported kernel/domain errors at construction
        @test_throws ArgumentError Problem(; kernel=:qg, Ld=1.0, contours=[c], dt=0.01, dev=:gpu)

        # Unknown device
        @test_throws ArgumentError Problem(; contours=[c], dt=0.01, dev=:unknown)
    end

    @testset "Problem factory — multi-layer" begin
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)
        c1 = circular_patch(0.5, 64, 1.0)

        prob = Problem(; kernel=:multilayer_qg, Ld=Ld, coupling=coupling,
                         layers=([c1], PVContour{Float64}[]), dt=0.01)
        @test prob.contour_problem isa MultiLayerContourProblem
        @test nlayers(prob.contour_problem) == 2

        # contours and layers mutually exclusive
        @test_throws ArgumentError Problem(; kernel=:multilayer_qg, Ld=Ld, coupling=coupling,
                                             contours=[c1], layers=([c1],), dt=0.01)

        # layers required for :multilayer_qg
        @test_throws ArgumentError Problem(; kernel=:multilayer_qg, Ld=Ld, coupling=coupling,
                                             contours=[c1], dt=0.01)
    end

    @testset "Problem factory — contours required" begin
        @test_throws ArgumentError Problem(; dt=0.01)
    end

    @testset "Problem factory — Float type" begin
        c = circular_patch(0.5, 32, 1.0; T=Float32)
        prob = Problem(; contours=[c], dt=Float32(0.01), T=Float32)
        @test prob.stepper.dt isa Float32
    end

    @testset "Problem factory — full evolve!" begin
        prob = Problem(; contours=[circular_patch(0.5, 64, 2π)], dt=0.01)
        circ_before = circulation(prob)
        evolve!(prob; nsteps=20)
        circ_after = circulation(prob)
        @test isapprox(circ_before, circ_after; rtol=1e-5)
    end
end
