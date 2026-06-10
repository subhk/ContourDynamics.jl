# Micro-benchmark for periodic velocity.
#
# A smooth (circular) contour exercises the CURVED segment path, which shares the
# `_periodic_{euler,qg,sqg}_green_correction` helpers with the straight path. This
# guards against the `@inline` helper change regressing the hot path. (The straight
# `segment_velocity` periodic path is rarely hot — only near-straight segments —
# and its correctness is pinned directly by the test suite.)
#
# Euler/QG periodic velocity is ~1 s per call at n=64 (E₁/Bessel evaluation over
# Ewald images), so keep `reps` small.
using ContourDynamics
using StaticArrays

function bench(kernel, dom; n=64, reps=5)
    prob = ContourProblem(kernel, dom, [circular_patch(0.5, n, 1.0)])
    vel = zeros(SVector{2,Float64}, total_nodes(prob))
    velocity!(vel, prob); velocity!(vel, prob)   # warm up
    alloc = @allocated velocity!(vel, prob)
    t = minimum(@elapsed(velocity!(vel, prob)) for _ in 1:reps)
    return t, alloc
end

dom = PeriodicDomain(10.0, 10.0)
for (name, k) in (("euler", EulerKernel()), ("qg", QGKernel(1.0)), ("sqg", SQGKernel(0.02)))
    t, a = bench(k, dom)
    println(rpad(name, 6), " ", round(t * 1e3; digits=3), " ms   ", a, " bytes")
end
