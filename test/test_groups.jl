const TEST_GROUPS = (
    # core: types, storage/workspace ownership, geometry, stepping, surgery,
    # threading, display, and the example-script checks.
    core = ["test_core.jl", "test_problem.jl", "test_show.jl", "test_surgery.jl",
            "test_threading.jl", "test_examples.jl"],
    # numerical: independent analytical, Fourier, and image-sum oracles.
    numerical = ["test_beta_plane.jl", "test_euler.jl", "test_qg.jl", "test_sqg.jl",
                 "test_periodic_qg_sqg.jl", "test_periodic_velocity_regression.jl"],
    # device: KernelAbstractions state/velocity/energy and surgery parity on CPU.
    device = ["test_device_state.jl", "test_device_surgery.jl"],
    performance = ["test_allocations.jl"],
)
const EXTENSION_TESTS = (
    jld2 = ("JLD2", :ContourDynamicsJLD2Ext, "test_jld2.jl"),
    diffeq = ("OrdinaryDiffEq", :ContourDynamicsDiffEqExt, "test_diffeq.jl"),
    recorded = ("RecordedArrays", :ContourDynamicsRecordedArraysExt, "test_recorded_arrays.jl"),
)

function run_extension_tests(name; required=true)
    dependency, extension, filename = getproperty(EXTENSION_TESTS, name)
    if Base.find_package(dependency) === nothing
        required && error("Test group $name requires $dependency in the active environment")
        @info "Skipping unavailable optional extension" dependency
        return
    end
    # Loading and test execution deliberately sit outside any catch block.
    # An installed but broken extension must fail the suite, not look absent.
    Base.eval(Main, Expr(:using, Expr(:., Symbol(dependency))))
    Base.get_extension(ContourDynamics, extension) === nothing &&
        error("$dependency loaded, but $extension did not activate")
    include(filename)
end

function run_test_groups(requested)
    valid = ("all", "extensions", "hardware", string.(keys(TEST_GROUPS))...,
             string.(keys(EXTENSION_TESTS))...)
    all(g -> g in valid, requested) || error("Unknown test group; choose from $(join(valid, ", "))")
    selected = String[]
    for group in requested
        expanded = group == "all" ? [string.(keys(TEST_GROUPS))...,
                                      string.(keys(EXTENSION_TESTS))...] :
                   group == "extensions" ? string.(keys(EXTENSION_TESTS)) : [group]
        append!(selected, expanded)
    end
    for group in unique(selected)
        println("Running test group: ", group)
        flush(stdout)
        name = Symbol(group)
        if hasproperty(TEST_GROUPS, name)
            for filename in getproperty(TEST_GROUPS, name)
                include(filename)
                flush(stdout)
            end
        elseif hasproperty(EXTENSION_TESTS, name)
            run_extension_tests(name; required=!("all" in requested) || group in requested || "extensions" in requested)
        else
            Base.eval(Main, :(using CUDA))
            Base.invokelatest(CUDA.functional) || error("hardware tests require a functional CUDA device")
            include("test_cuda_surgery.jl")
        end
    end
end
