using Test, Logging
include("test_utils.jl")
include("test_groups.jl")

# No arguments preserves the standard CPU test suite and installed extensions.
# Explicit groups fail if a requested dependency or hardware backend is absent.
groups = isempty(ARGS) ? ["all"] : ARGS
run_test_groups(groups)
