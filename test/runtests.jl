using Test

include("test_utils.jl")

@testset "ContourDynamics.jl" begin
    @testset "Core Types" begin
        # PVContour construction
        c = circular_patch(1.0, 64, 1.0)
        @test nnodes(c) == 64
        @test c.pv == 1.0

        # EulerKernel
        k = EulerKernel()
        @test k isa AbstractKernel

        # QGKernel validation
        @test_throws ArgumentError QGKernel(-1.0)
        qg = QGKernel(2.5)
        @test qg.Ld == 2.5

        # Domains
        d = UnboundedDomain()
        @test d isa AbstractDomain
        pd = PeriodicDomain(Float64(π), Float64(π))
        @test pd.Lx == Float64(π)
        @test_throws MethodError PeriodicDomain(1, 1)  # Int not allowed
        @test_throws ArgumentError PeriodicDomain(-1.0, 1.0)

        # ContourProblem
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        @test total_nodes(prob) == 64

        # SurgeryParams validation
        sp = SurgeryParams(0.001, 0.005, 0.1, 1e-6, 10)
        @test sp.n_surgery == 10
        @test_throws ArgumentError SurgeryParams(0.01, 0.005, 0.003, 1e-6, 10)  # Delta_max < mu

        # RK4Stepper construction
        rk = RK4Stepper(0.01, 64)
        @test rk.dt == 0.01
        @test length(rk.k1) == 64
    end

    @testset "Euler Kernel" begin
        c = circular_patch(1.0, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        vel = zeros(SVector{2, Float64}, total_nodes(prob))
        velocity!(vel, prob)

        # Velocity at each node: tangential, magnitude = pv*R/2 = 0.5
        expected_speed = 0.5
        for i in 1:nnodes(c)
            speed = sqrt(vel[i][1]^2 + vel[i][2]^2)
            @test speed ≈ expected_speed rtol=0.05
        end

        # Check tangential direction: velocity perpendicular to position
        for i in 1:nnodes(c)
            pos = c.nodes[i]
            @test abs(vel[i][1]*pos[1] + vel[i][2]*pos[2]) < 0.05
        end
    end

    @testset "QG Kernel" begin
        N = 32
        c = circular_patch(1.0, N, 1.0)
        prob_euler = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        prob_qg = ContourProblem(QGKernel(100.0), UnboundedDomain(), [c])

        vel_euler = zeros(SVector{2, Float64}, N)
        vel_qg = zeros(SVector{2, Float64}, N)
        velocity!(vel_euler, prob_euler)
        velocity!(vel_qg, prob_qg)

        # Large Ld → QG approaches Euler
        for i in 1:N
            @test vel_qg[i] ≈ vel_euler[i] rtol=0.1
        end

        # Small Ld → QG velocity weaker than Euler
        prob_qg_small = ContourProblem(QGKernel(0.5), UnboundedDomain(), [c])
        vel_qg_small = zeros(SVector{2, Float64}, N)
        velocity!(vel_qg_small, prob_qg_small)

        euler_speed = sqrt(vel_euler[1][1]^2 + vel_euler[1][2]^2)
        qg_speed = sqrt(vel_qg_small[1][1]^2 + vel_qg_small[1][2]^2)
        @test qg_speed < euler_speed
    end

    @testset "Per-Contour Diagnostics" begin
        c = circular_patch(1.0, 64, 1.0)
        @test vortex_area(c) ≈ π rtol=5e-3

        cx = centroid(c)
        @test abs(cx[1]) < 1e-10
        @test abs(cx[2]) < 1e-10

        e = elliptical_patch(2.0, 1.0, 64, 1.0)
        @test vortex_area(e) ≈ 2π rtol=5e-3

        ratio, angle = ellipse_moments(e)
        @test ratio ≈ 2.0 rtol=0.05
        @test abs(angle) < 0.1

        # Rotated ellipse: Jxy ≠ 0, catches incorrect product-of-inertia formula
        θ = π / 4  # 45 degrees
        e_rot = rotated_elliptical_patch(2.0, 1.0, 128, 1.0, θ)
        ratio_rot, angle_rot = ellipse_moments(e_rot)
        @test ratio_rot ≈ 2.0 rtol=0.05
        @test angle_rot ≈ θ atol=0.05

        # 30-degree rotation
        θ30 = π / 6
        e_rot30 = rotated_elliptical_patch(2.0, 1.0, 128, 1.0, θ30)
        ratio30, angle30 = ellipse_moments(e_rot30)
        @test ratio30 ≈ 2.0 rtol=0.05
        @test angle30 ≈ θ30 atol=0.05
    end

    @testset "Problem-Level Diagnostics" begin
        c = circular_patch(1.0, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

        @test circulation(prob) ≈ π rtol=0.01
        @test enstrophy(prob) ≈ π / 2 rtol=0.01

        E = energy(prob)
        @test E > 0
        @test isfinite(E)

        L = angular_momentum(prob)
        @test L ≈ π / 2 rtol=0.02
    end

    @testset "Node Management" begin
        nodes = SVector{2, Float64}[
            SVector(0.0, 0.0), SVector(0.001, 0.0), SVector(0.002, 0.0),
            SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0),
        ]
        c = PVContour(nodes, 1.0)
        params = SurgeryParams(0.002, 0.01, 0.05, 1e-6, 10)

        c_new = remesh(c, params)

        for i in 1:nnodes(c_new)
            j = mod1(i + 1, nnodes(c_new))
            d = c_new.nodes[j] - c_new.nodes[i]
            spacing = sqrt(d[1]^2 + d[2]^2)
            @test spacing >= params.mu * 0.9
            @test spacing <= params.Delta_max * 1.1
        end

        @test vortex_area(c_new) ≈ vortex_area(c) rtol=0.1
    end

    include("test_surgery.jl")

    @testset "Time Steppers" begin
        @testset "RK4 single step" begin
            c = circular_patch(1.0, 64, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            stepper = RK4Stepper(0.01, total_nodes(prob))

            area_before = vortex_area(prob.contours[1])
            timestep!(prob, stepper)
            area_after = vortex_area(prob.contours[1])

            @test area_after ≈ area_before rtol=1e-8
        end

        @testset "Leapfrog construction" begin
            lf = LeapfrogStepper(0.01, 64)
            @test lf.dt == 0.01
            @test length(lf.nodes_prev) == 64
            @test lf.initialized == false
            @test lf.ra_coeff == 0.05  # default

            lf0 = LeapfrogStepper(0.01, 32; ra_coeff=0.0)
            @test lf0.ra_coeff == 0.0
        end

        @testset "Leapfrog single step (bootstrap)" begin
            c = circular_patch(1.0, 64, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            stepper = LeapfrogStepper(0.01, total_nodes(prob))

            @test stepper.initialized == false
            area_before = vortex_area(prob.contours[1])
            timestep!(prob, stepper)
            area_after = vortex_area(prob.contours[1])

            # Bootstrap uses RK2 midpoint — area conserved to high precision
            @test area_after ≈ area_before rtol=1e-6
            @test stepper.initialized == true
        end

        @testset "Leapfrog multiple steps" begin
            c = circular_patch(1.0, 64, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            stepper = LeapfrogStepper(0.01, total_nodes(prob))

            area_before = vortex_area(prob.contours[1])
            # Step 1: bootstrap (RK2), Steps 2+: leapfrog
            for _ in 1:10
                timestep!(prob, stepper)
            end
            area_after = vortex_area(prob.contours[1])
            @test area_after ≈ area_before rtol=1e-4
        end

        @testset "Leapfrog resize_buffers! resets initialization" begin
            c = circular_patch(1.0, 64, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            stepper = LeapfrogStepper(0.01, total_nodes(prob))

            # Take one step to initialize
            timestep!(prob, stepper)
            @test stepper.initialized == true

            # resize_buffers! should reset
            resize_buffers!(stepper, prob)
            @test stepper.initialized == false

            # Next step should re-bootstrap without error
            timestep!(prob, stepper)
            @test stepper.initialized == true
        end

        @testset "Leapfrog ra_coeff=0 vs default" begin
            # Two identical problems, one with RA filter, one without
            c1 = circular_patch(1.0, 32, 1.0)
            c2 = circular_patch(1.0, 32, 1.0)
            prob1 = ContourProblem(EulerKernel(), UnboundedDomain(), [c1])
            prob2 = ContourProblem(EulerKernel(), UnboundedDomain(), [c2])
            lf_filtered = LeapfrogStepper(0.01, 32; ra_coeff=0.05)
            lf_pure = LeapfrogStepper(0.01, 32; ra_coeff=0.0)

            for _ in 1:5
                timestep!(prob1, lf_filtered)
                timestep!(prob2, lf_pure)
            end

            # After multiple leapfrog steps, the filtered and unfiltered
            # trajectories should differ (the RA filter modifies the history)
            any_differ = false
            for i in 1:32
                if !(prob1.contours[1].nodes[i] ≈ prob2.contours[1].nodes[i])
                    any_differ = true
                    break
                end
            end
            @test any_differ
        end

        @testset "Leapfrog evolve! with surgery" begin
            c = circular_patch(1.0, 64, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            stepper = LeapfrogStepper(0.01, total_nodes(prob))
            params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 5)

            A0 = vortex_area(prob.contours[1])
            evolve!(prob, stepper, params; nsteps=10)
            A1 = vortex_area(prob.contours[1])

            # Area conserved even with surgery triggering re-bootstrap
            @test A1 ≈ A0 rtol=1e-3
        end

        @testset "evolve! with callbacks" begin
            c = circular_patch(1.0, 64, 1.0)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
            stepper = RK4Stepper(0.01, total_nodes(prob))
            params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)

            initial_area = vortex_area(prob.contours[1])
            areas = Float64[]
            cb = (p, step) -> push!(areas, vortex_area(p.contours[1]))

            evolve!(prob, stepper, params; nsteps=10, callbacks=[cb])
            @test length(areas) == 11  # step 0 (initial) + steps 1-10
            @test all(a -> abs(a - initial_area) / abs(initial_area) < 1e-4, areas)
        end
    end

    include("test_kirchhoff.jl")

    include("test_conservation.jl")

    @testset "Periodic Domain (Ewald)" begin
        clear_ewald_cache!()
        c = circular_patch(0.1, 16, 1.0)
        prob_unbounded = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        prob_periodic = ContourProblem(EulerKernel(), PeriodicDomain(10.0, 10.0), [c])

        vel_u = zeros(SVector{2, Float64}, 16)
        vel_p = zeros(SVector{2, Float64}, 16)
        velocity!(vel_u, prob_unbounded)
        velocity!(vel_p, prob_periodic)

        for i in 1:16
            @test vel_p[i] ≈ vel_u[i] rtol=0.15
        end
    end

    @testset "Periodic contour wrapping preserves geometry" begin
        domain = PeriodicDomain(1.0, 1.0)
        R, N = 0.3, 64
        cx = 0.85
        nodes = [SVector(cx + R * cos(2π * k / N), R * sin(2π * k / N)) for k in 0:(N - 1)]
        c = PVContour(nodes, 1.0)
        prob = ContourProblem(EulerKernel(), domain, [c])

        A0 = vortex_area(c)
        ctr0 = centroid(c)
        rel0 = [node - ctr0 for node in c.nodes]

        wrap_nodes!(prob)

        wrapped = prob.contours[1]
        A1 = vortex_area(wrapped)
        ctr1 = centroid(wrapped)
        rel1 = [node - ctr1 for node in wrapped.nodes]

        @test A1 ≈ A0 rtol=1e-12 atol=1e-12
        @test ctr1 ≈ ContourDynamics.wrap_node(ctr0, domain) rtol=1e-12 atol=1e-12
        @test all(rel1[i] ≈ rel0[i] for i in eachindex(rel0))
    end

    include("test_merger.jl")

    include("test_periodic_qg_sqg.jl")

    include("test_show.jl")

    @testset "Multi-Layer QG" begin
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)

        kernel = MultiLayerQGKernel(Ld, coupling)
        @test nlayers(kernel) == 2

        @test_throws ArgumentError MultiLayerQGKernel(SVector(3.0), coupling)

        c1 = circular_patch(1.0, 64, 1.0)
        c2 = circular_patch(0.5, 64, -1.0)
        domain = UnboundedDomain()
        prob = MultiLayerContourProblem(kernel, domain, ([c1], [c2]))

        @test nlayers(prob) == 2
        @test total_nodes(prob) == 128

        vel = (zeros(SVector{2, Float64}, 64), zeros(SVector{2, Float64}, 64))
        velocity!(vel, prob)

        @test all(v -> all(isfinite, v), vel[1])
        @test all(v -> all(isfinite, v), vel[2])
        @test any(v -> sqrt(v[1]^2 + v[2]^2) > 1e-10, vel[1])

        # Changing the deformation radius through a consistent coupling matrix
        # should change the multilayer velocity field.
        Ld2 = SVector(2.0)
        F2 = 1.0 / (2 * Ld2[1]^2)
        kernel2 = MultiLayerQGKernel(Ld2, SMatrix{2,2}(-F2, F2, F2, -F2))
        prob2 = MultiLayerContourProblem(kernel2, domain, ([c1], [c2]))
        vel2 = (zeros(SVector{2, Float64}, 64), zeros(SVector{2, Float64}, 64))
        velocity!(vel2, prob2)
        @test maximum(sqrt((vel[1][i][1] - vel2[1][i][1])^2 + (vel[1][i][2] - vel2[1][i][2])^2) for i in 1:64) > 1e-6
    end

    include("test_fmm.jl")

    @testset "Spanning Contours & Beta Staircase" begin
        T = Float64
        domain = PeriodicDomain(T(3.0))

        # beta_staircase creates correct number of spanning contours
        staircase = beta_staircase(T(1.0), domain, 6; nodes_per_contour=16)
        @test length(staircase) == 5  # n_steps - 1

        # Each contour is spanning with correct wrap
        for c in staircase
            @test is_spanning(c)
            @test c.wrap == SVector{2,T}(6.0, 0.0)
            @test nnodes(c) == 16
        end

        # PV jump = beta * dy
        dy = 2 * 3.0 / 6
        @test staircase[1].pv ≈ 1.0 * dy

        # Spanning contours have zero area (skip in diagnostics)
        @test vortex_area(staircase[1]) == zero(T)

        # next_node wraps correctly for spanning contours
        c = staircase[1]
        last_node = c.nodes[end]
        wrapped = next_node(c, nnodes(c))
        @test wrapped ≈ c.nodes[1] + c.wrap

        # Velocity computation works with spanning + closed contours mixed
        vortex = PVContour([SVector{2,T}(0.3*cos(2π*k/16), 0.5 + 0.3*sin(2π*k/16)) for k in 0:15], T(2π))
        all_contours = vcat(staircase, [vortex])
        kernel = QGKernel(T(1.0))
        prob = ContourProblem(kernel, domain, all_contours)
        vel = zeros(SVector{2,T}, total_nodes(prob))
        velocity!(vel, prob)
        @test all(v -> all(isfinite, v), vel)

        # Surgery skips spanning contours in reconnection and filament removal
        params = SurgeryParams(T(0.01), T(0.05), T(0.5), T(1e-4), 10)
        surgery!(prob, params)
        # All spanning contours should survive surgery
        n_spanning = count(is_spanning, prob.contours)
        @test n_spanning == 5

        # Remesh preserves wrap
        remeshed = remesh(staircase[1], params)
        @test is_spanning(remeshed)
        @test remeshed.wrap == staircase[1].wrap
    end

    # Extension tests (only run when deps are available)
    if haskey(Base.loaded_modules, Base.PkgId(Base.UUID("033835bb-8acc-5ee8-8aae-3f567f8a3819"), "JLD2"))
        include("test_jld2.jl")
    else
        try
            using JLD2
            include("test_jld2.jl")
        catch e
            @warn "JLD2 not available — skipping JLD2 extension tests" exception=e
        end
    end
end
