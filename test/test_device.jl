using ContourDynamics
using StaticArrays
using Test

include("test_utils.jl")

@testset "Device abstraction" begin
    @testset "ContourProblem defaults to CPU" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        @test prob.dev === CPU()
    end

    @testset "ContourProblem accepts dev keyword" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=CPU())
        @test prob.dev === CPU()
    end

    @testset "GPU() without CUDA gives helpful error" begin
        @test_throws ErrorException device_array(GPU())
    end

    @testset "CPU device_array returns Array" begin
        @test device_array(CPU()) === Array
    end

    @testset "to_cpu is identity for Array" begin
        x = [1.0, 2.0, 3.0]
        @test to_cpu(x) === x
    end

    @testset "device_zeros CPU" begin
        z = device_zeros(CPU(), Float64, 5)
        @test z == zeros(5)
        @test z isa Vector{Float64}
    end
end
