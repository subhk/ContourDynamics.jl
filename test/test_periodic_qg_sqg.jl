using Test, ContourDynamics, StaticArrays

extended = get(ENV, "CONTOURDYNAMICS_EXTENDED_TESTS", "false") == "true"

@testset "Periodic QG/SQG" begin
    clear_ewald_cache!()

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
