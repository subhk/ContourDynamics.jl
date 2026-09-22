using Test, ContourDynamics, StaticArrays

@testset "Periodic velocity and energy are independent of coordinate units" begin
    clear_ewald_cache!()
    for T in (Float32, Float64), model in (:euler, :qg, :sqg)
        scale = T === Float32 ? T(1e5) : T(1e10)
        rtol = T === Float32 ? T(2e-4) : T(2e-11)
        results = map((one(T), scale)) do s
            kernel = model === :euler ? EulerKernel() :
                     model === :qg ? QGKernel(T(0.4) * s) : SQGKernel(T(0.03) * s)
            domain = PeriodicDomain(s)
            cs = [circular_patch(T(0.2) * s, 16, one(T); T=T)]
            prob = ContourProblem(kernel, domain, cs)
            state = DeviceContourState(cs, CPU())
            point = SVector(T(0.65) * s, T(0.17) * s)
            # With fixed PV, Euler/QG velocity scales as length and energy as
            # length^4. SQG velocity is unchanged and energy scales as length^3
            # when its regularization length is scaled with the geometry.
            vscale = model === :sqg ? one(T) : s
            escale = model === :sqg ? s^3 : s^4
            (velocity(prob, point) / vscale,
             ContourDynamics._ka_velocity_at_state(state, kernel, domain, point, CPU()) / vscale,
             energy(prob) / escale,
             ContourDynamics._ka_energy_from_state(state, kernel, domain, CPU()) / escale)
        end
        base, scaled = results
        @testset "$T $model" begin
            @test scaled[1] ≈ base[1] rtol=rtol
            @test scaled[2] ≈ base[1] rtol=rtol
            @test scaled[3] ≈ base[3] rtol=rtol
            @test scaled[4] ≈ base[3] rtol=rtol
        end
    end
end
