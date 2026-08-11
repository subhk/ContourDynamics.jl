using Printf

# Display methods favor compact one-line summaries for ordinary `show` and a
# tree-shaped expanded layout for `MIME"text/plain"`, which is what the REPL and
# notebooks use for top-level values.

# ── Device types ─────────────────────────────────────────

Base.show(io::IO, ::CPU) = print(io, "CPU()")
Base.show(io::IO, ::GPU) = print(io, "GPU()")

# ── Tree-drawing helpers ────────────────────────────────

_tree_prefix(is_last::Bool) = is_last ? "└── " : "├── "
_tree_indent(is_last::Bool) = is_last ? "    " : "│   "

# Maximum number of contours to list before truncating. Large simulations can
# contain hundreds of contours after surgery; summaries should stay readable.
const _MAX_CONTOURS_SHOWN = 5

# ── Kernels ─────────────────────────────────────────────

Base.show(io::IO, ::EulerKernel) = print(io, "EulerKernel")
Base.show(io::IO, ::MIME"text/plain", k::EulerKernel) = show(io, k)

Base.show(io::IO, k::QGKernel{T}) where {T} = print(io, "QGKernel{$T}: Ld = ", k.Ld)
Base.show(io::IO, ::MIME"text/plain", k::QGKernel) = show(io, k)

function Base.show(io::IO, k::BetaPlaneQGKernel{T}) where {T}
    print(io, "BetaPlaneQGKernel{$T}: beta = ", k.beta,
          ", Ld = ", k.Ld,
          ", reference contours = ", length(k.reference_contours))
end
Base.show(io::IO, ::MIME"text/plain", k::BetaPlaneQGKernel) = show(io, k)

Base.show(io::IO, k::SQGKernel{T}) where {T} = print(io, "SQGKernel{$T}: δ = ", k.δ)
Base.show(io::IO, ::MIME"text/plain", k::SQGKernel) = show(io, k)

function Base.show(io::IO, k::MultiLayerQGKernel{N, M, T}) where {N, M, T}
    print(io, "MultiLayerQGKernel{$N, $T}")
end

function Base.show(io::IO, ::MIME"text/plain", k::MultiLayerQGKernel{N, M, T}) where {N, M, T}
    println(io, "MultiLayerQGKernel{$N, $T}")
    _show_kernel_details(io, k, "")
end

# ── Domains ─────────────────────────────────────────────

Base.show(io::IO, ::UnboundedDomain) = print(io, "UnboundedDomain")
Base.show(io::IO, ::MIME"text/plain", d::UnboundedDomain) = show(io, d)

function Base.show(io::IO, d::PeriodicDomain{T}) where {T}
    print(io, "PeriodicDomain{$T}: x ∈ [-", d.Lx, ", ", d.Lx, ") × y ∈ [-", d.Ly, ", ", d.Ly, ")")
end
Base.show(io::IO, ::MIME"text/plain", d::PeriodicDomain) = show(io, d)

# ── PVContour ───────────────────────────────────────────

function _contour_summary(io::IO, c::PVContour{T}) where {T}
    # Keep this allocation-light because problem summaries call it repeatedly.
    # Centroids are only computed for closed contours with enough nodes.
    n = nnodes(c)
    print(io, n, " node", n == 1 ? "" : "s", ", Δq = ", c.pv, ", ")
    if is_spanning(c)
        print(io, "spanning")
    else
        print(io, "closed")
        if n >= 3
            ctr = centroid(c)
            print(io, @sprintf(", centered at (%.2f, %.2f)", ctr[1], ctr[2]))
        end
    end
end

function Base.show(io::IO, c::PVContour{T}) where {T}
    print(io, "PVContour{$T}: ")
    _contour_summary(io, c)
end

Base.show(io::IO, ::MIME"text/plain", c::PVContour) = show(io, c)

# ── ContourProblem ──────────────────────────────────────

function Base.show(io::IO, prob::ContourProblem{K, D, T}) where {K, D, T}
    print(io, "ContourProblem{", _type_name(K), ", ", _type_name(D), ", $T}")
end

function Base.show(io::IO, ::MIME"text/plain", prob::ContourProblem{K, D, T}) where {K, D, T}
    # The expanded view mirrors the fields users most often inspect while
    # keeping contour details behind a capped nested list.
    _show_contour_problem(io, prob, prob.contours)
end

function Base.show(io::IO, ::MIME"text/plain", prob::ContourProblem{K, D, T, GPU}) where {K, D, T}
    _show_contour_problem(io, prob, materialize_contours(prob))
end

function _show_contour_problem(io::IO, prob::ContourProblem{K, D, T}, contours) where {K, D, T}
    println(io, "ContourProblem{", _type_name(K), ", ", _type_name(D), ", $T}")
    println(io, "├── kernel: ", prob.kernel)
    println(io, "├── domain: ", prob.domain)
    println(io, "├── device: ", prob.dev)
    nc = length(contours)
    print(io, "└── contours: $nc PVContour{$T}")
    _show_contour_list(io, contours, "    ")
end

"""Print a short type name without parameters for readability."""
_type_name(::Type{T}) where {T} = string(nameof(T))

function _show_contour_list(io::IO, contours::Vector{PVContour{T}}, pad::String) where {T}
    # `pad` is the indentation inherited from the parent tree branch. The helper
    # is shared by single-layer and multi-layer problem displays.
    nc = length(contours)
    nc == 0 && return
    n_show = min(nc, _MAX_CONTOURS_SHOWN)
    truncated = nc > _MAX_CONTOURS_SHOWN
    for i in 1:n_show
        is_last = !truncated && i == nc
        println(io)
        print(io, pad, _tree_prefix(is_last))
        _contour_summary(io, contours[i])
    end
    if truncated
        println(io)
        print(io, pad, "└── … and $(nc - n_show) more")
    end
end

# ── MultiLayerContourProblem ────────────────────────────

function Base.show(io::IO, prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    print(io, "MultiLayerContourProblem{$N, ", _type_name(D), ", $T}")
end

function Base.show(io::IO, ::MIME"text/plain", prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    _show_multilayer_problem(io, prob, prob.layers)
end

function Base.show(io::IO, ::MIME"text/plain", prob::MultiLayerContourProblem{N, K, D, T, GPU}) where {N, K, D, T}
    _show_multilayer_problem(io, prob, materialize_contours(prob))
end

function _show_multilayer_problem(io::IO, prob::MultiLayerContourProblem{N, K, D, T}, layers) where {N, K, D, T}
    println(io, "MultiLayerContourProblem{$N, ", _type_name(D), ", $T}")
    # Kernel details get their own nested subtree because coupling/eigenvalues
    # are the core state that distinguishes multi-layer QG problems.
    println(io, "├── kernel: ", prob.kernel)
    _show_kernel_details(io, prob.kernel, "│   ")
    println(io)
    println(io, "├── domain: ", prob.domain)
    println(io, "├── device: ", prob.dev)
    print(io, "└── layers: $N layer", N == 1 ? "" : "s")
    for i in 1:N
        is_last_layer = i == N
        layer = layers[i]
        nlc = length(layer)
        println(io)
        layer_pad = "    "
        print(io, layer_pad, _tree_prefix(is_last_layer),
              "Layer $i: $nlc contour", nlc == 1 ? "" : "s")
        contour_pad = layer_pad * _tree_indent(is_last_layer)
        _show_contour_list(io, layer, contour_pad)
    end
end

function _show_kernel_details(io::IO, k::MultiLayerQGKernel{N, M, T}, pad::String) where {N, M, T}
    println(io, pad, "├── Ld: ", k.Ld)
    println(io, pad, "├── layer thicknesses: ", k.layer_thicknesses)
    println(io, pad, "├── coupling: $(N)×$(N) SMatrix{$T}")
    print(io, pad, "└── eigenvalues: ", k.eigenvalues)
end

# ── Time Steppers ───────────────────────────────────────

function Base.show(io::IO, s::RK4Stepper{T}) where {T}
    print(io, "RK4Stepper{$T}: dt = ", s.dt)
end
Base.show(io::IO, ::MIME"text/plain", s::RK4Stepper) = show(io, s)

# ── SurgeryParams ───────────────────────────────────────

function Base.show(io::IO, p::SurgeryParams{T}) where {T}
    print(io, "SurgeryParams{$T}")
end

function Base.show(io::IO, ::MIME"text/plain", p::SurgeryParams{T}) where {T}
    println(io, "SurgeryParams{$T}")
    println(io, "├── δ (proximity): ", p.δ)
    println(io, "├── μ (min segment): ", p.μ)
    println(io, "├── Δ_max (max segment): ", p.Δ_max)
    println(io, "├── area_min: ", p.area_min)
    print(io,   "└── n_surgery: ", p.n_surgery, " steps")
end

# ── Problem ────────────────────────────────────────────

function Base.show(io::IO, prob::Problem)
    print(io, "Problem(", prob.contour_problem, ", ", prob.stepper, ")")
end

function Base.show(io::IO, ::MIME"text/plain", prob::Problem)
    println(io, "Problem")
    println(io, "├── ", prob.contour_problem)
    println(io, "├── ", prob.stepper)
    if prob.surgery_params === nothing
        print(io, "└── surgery: disabled")
    else
        sp = prob.surgery_params
        print(io, "└── surgery: δ=", sp.δ, ", μ=", sp.μ,
              ", Δ_max=", sp.Δ_max, ", every ", sp.n_surgery, " steps")
    end
end
