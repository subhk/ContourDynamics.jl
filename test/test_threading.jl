using ContourDynamics
using StaticArrays
using Test

@isdefined(circular_patch) || include("test_utils.jl")

@testset "Threaded Paths" begin
    if Threads.nthreads() == 1
        @test true
    else
        @testset "Threaded Direct Velocity Matches Pointwise Evaluation" begin
            c1 = circular_patch(1.0, 96, 1.0)
            c2 = PVContour([SVector(2.5 + cos(2π * i / 96), 0.5 * sin(2π * i / 96)) for i in 0:95], -0.75)
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])

            N = total_nodes(prob)
            @test ContourDynamics._should_thread_velocity(N)

            vel_direct = zeros(SVector{2,Float64}, N)
            vel_pointwise = similar(vel_direct)
            ContourDynamics._direct_velocity!(vel_direct, prob)

            idx = 1
            for c in prob.contours
                for node in c.nodes
                    vel_pointwise[idx] = velocity(prob, node)
                    idx += 1
                end
            end

            max_err = maximum(sqrt(sum((vel_direct[i] - vel_pointwise[i]).^2)) for i in 1:N)
            @test max_err < 1e-12
        end

        @testset "Threaded Multilayer Direct Velocity Matches Pointwise Evaluation" begin
            Ld = SVector(1.0)
            F = 1.0 / (2 * Ld[1]^2)
            coupling = SMatrix{2,2}(-F, F, F, -F)
            kernel = MultiLayerQGKernel(Ld, coupling)
            c1 = circular_patch(0.5, 128, 1.0)
            c2 = PVContour([SVector(2.0 + 0.5 * cos(2π * i / 128), 0.5 * sin(2π * i / 128)) for i in 0:127], 0.5)
            prob = MultiLayerContourProblem(kernel, UnboundedDomain(), ([c1], [c2]))

            vel_direct = ContourDynamics._make_vel_tuple(prob)
            vel_pointwise = ContourDynamics._make_vel_tuple(prob)
            ContourDynamics._direct_velocity!(vel_direct, prob)

            for layer in 1:2
                for (j, node) in enumerate(only(prob.layers[layer]).nodes)
                    vel_pointwise[layer][j] = velocity(prob, node)[layer]
                end
            end

            for layer in 1:2
                max_err = maximum(sqrt(sum((vel_pointwise[layer][j] - vel_direct[layer][j]).^2))
                                  for j in eachindex(vel_direct[layer]))
                @test max_err < 1e-10
            end
        end
    end
end
