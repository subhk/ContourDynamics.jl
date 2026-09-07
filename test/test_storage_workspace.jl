using Test, ContourDynamics, StaticArrays

@testset "Storage ownership" begin
    input = [circular_patch(1., 16, 1.)]
    prob = ContourProblem(EulerKernel(), UnboundedDomain(), input)
    @test contours(prob) === input
    @test prob.contours === input
    @test materialize_contours(prob) === input # compatibility accessor
    saved = snapshot_contours(prob)
    original = saved[1].nodes[1]
    input[1].nodes[1] = SVector(2., 0.)
    input[1].corners[1] = true
    @test saved[1].nodes[1] == original
    @test !saved[1].corners[1]
    @test contours(prob)[1].nodes[1] == SVector(2., 0.)

    state = DeviceContourState(saved, CPU())
    storage = ContourDynamics._DeviceContourStorage(state)
    @test_throws ErrorException ContourDynamics._borrow_contours(storage)
    before = ContourDynamics._snapshot_storage(storage)
    state.x[1] += 1
    after = ContourDynamics._snapshot_storage(storage)
    @test after[1].nodes[1][1] == before[1].nodes[1][1] + 1
    @test before[1].nodes[1] == original

    kernel = MultiLayerQGKernel(SVector(1.), SMatrix{2,2}(-.5, .5, .5, -.5))
    multi = MultiLayerContourProblem(kernel, UnboundedDomain(), (saved, deepcopy(saved)))
    copy_layers = snapshot_contours(multi)
    contours(multi)[2][1].nodes[1] += SVector(1., 0.)
    @test copy_layers[2][1].nodes[1] == original
    wrapped = Problem(prob, RK4Stepper(.01, total_nodes(prob)), nothing)
    @test snapshot_contours(wrapped)[1].nodes == contours(prob)[1].nodes
end

@testset "Explicit computational workspace" begin
    ws = ExecutionWorkspace()
    prob = Problem(contours=[circular_patch(1., 16, 1.)], dt=.01, workspace=ws)
    other = Problem(contours=[circular_patch(1., 12, 1.)], dt=.01)
    @test execution_workspace(prob) === ws
    @test execution_workspace(other) !== ws
    @test prob.contour_problem.velocity_scratch === ws.cpu

    state = DeviceContourState(snapshot_contours(prob), CPU())
    vel = zeros(SVector{2,Float64}, total_nodes(prob))
    ContourDynamics._ka_velocity_from_state!(vel, state, EulerKernel(), UnboundedDomain(), CPU(); workspace=ws)
    @test !isempty(ws.buffers)
    buffer = ContourDynamics._get_state_workspace(CPU(), Float64, length(vel); workspace=ws)
    other_buffer = ContourDynamics._get_state_workspace(CPU(), Float64, 12; workspace=execution_workspace(other))
    @test ContourDynamics._get_state_workspace(CPU(), Float64, length(vel); workspace=ws) === buffer
    @test buffer !== other_buffer
    ContourDynamics._rk4_state_step!(state, EulerKernel(), UnboundedDomain(), prob.stepper, CPU(); workspace=ws)
    @test ContourDynamics._get_state_workspace(CPU(), Float64, length(vel); workspace=ws) === buffer
    clear_state_workspace_cache!(prob)
    @test isempty(ws.buffers)
    @test !isempty(execution_workspace(other).buffers)
    @test prob.contour_problem.velocity_scratch === ws.cpu

    # Reusing a workspace sequentially across different models must invalidate
    # modal transforms even when both problems have the same number of layers.
    layers = ([circular_patch(.3, 12, 1.)], [circular_patch(.2, 12, -.5; cx=.6)])
    k1 = MultiLayerQGKernel(SVector(1.), SMatrix{2,2}(-.5, .5, .5, -.5))
    k2 = MultiLayerQGKernel(SVector(1.), SMatrix{2,2}(-.75, .25, .75, -.25))
    p1 = MultiLayerContourProblem(k1, UnboundedDomain(), deepcopy(layers); workspace=ws)
    p2 = MultiLayerContourProblem(k2, UnboundedDomain(), deepcopy(layers); workspace=ws)
    reference = MultiLayerContourProblem(k2, UnboundedDomain(), deepcopy(layers))
    velocity(p1, SVector(.1, .2))
    @test all(isapprox.(velocity(p2, SVector(.1, .2)), velocity(reference, SVector(.1, .2)); rtol=1e-12))
end
