using Printf

# ── Device types ─────────────────────────────────────────

Base.show(io::IO, ::CPU) = print(io, "CPU()")
Base.show(io::IO, ::GPU) = print(io, "GPU()")

# ── Tree-drawing helpers ────────────────────────────────

_tree_prefix(is_last::Bool) = is_last ? "└── " : "├── "
_tree_indent(is_last::Bool) = is_last ? "    " : "│   "

# Maximum number of contours to list before truncating
const _MAX_CONTOURS_SHOWN = 5

# ── Kernels ─────────────────────────────────────────────

Base.show(io::IO, ::EulerKernel) = print(io, "EulerKernel")
Base.show(io::IO, ::MIME"text/plain", k::EulerKernel) = show(io, k)

Base.show(io::IO, k::QGKernel{T}) where {T} = print(io, "QGKernel{$T}: Ld = ", k.Ld)
Base.show(io::IO, ::MIME"text/plain", k::QGKernel) = show(io, k)

Base.show(io::IO, k::SQGKernel{T}) where {T} = print(io, "SQGKernel{$T}: δ = ", k.delta)
Base.show(io::IO, ::MIME"text/plain", k::SQGKernel) = show(io, k)

function Base.show(io::IO, k::MultiLayerQGKernel{N, M, T}) where {N, M, T}
    print(io, "MultiLayerQGKernel{$N, $T}")
end

function Base.show(io::IO, ::MIME"text/plain", k::MultiLayerQGKernel{N, M, T}) where {N, M, T}
    println(io, "MultiLayerQGKernel{$N, $T}")
    println(io, "├── Ld: ", k.Ld)
    println(io, "├── coupling: $(N)×$(N) SMatrix{$T}")
    print(io,   "└── eigenvalues: ", k.eigenvalues)
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
    println(io, "ContourProblem{", _type_name(K), ", ", _type_name(D), ", $T}")
    println(io, "├── kernel: ", prob.kernel)
    println(io, "├── domain: ", prob.domain)
    println(io, "├── device: ", prob.dev)
    nc = length(prob.contours)
    print(io, "└── contours: $nc PVContour{$T}")
    _show_contour_list(io, prob.contours, "    ")
end

"""Print a short type name without parameters for readability."""
_type_name(::Type{T}) where {T} = string(nameof(T))

function _show_contour_list(io::IO, contours::Vector{PVContour{T}}, pad::String) where {T}
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
    println(io, "MultiLayerContourProblem{$N, ", _type_name(D), ", $T}")
    # Kernel with nested sub-tree
    println(io, "├── kernel: ", prob.kernel)
    _show_kernel_details(io, prob.kernel, "│   ")
    # Domain
    println(io, "├── domain: ", prob.domain)
    # Device
    println(io, "├── device: ", prob.dev)
    # Layers
    print(io, "└── layers: $N layer", N == 1 ? "" : "s")
    for i in 1:N
        is_last_layer = i == N
        layer = prob.layers[i]
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
    println(io, pad, "├── coupling: $(N)×$(N) SMatrix{$T}")
    println(io, pad, "└── eigenvalues: ", k.eigenvalues)
end

# ── Time Steppers ───────────────────────────────────────

function Base.show(io::IO, s::RK4Stepper{T}) where {T}
    print(io, "RK4Stepper{$T}: dt = ", s.dt)
end
Base.show(io::IO, ::MIME"text/plain", s::RK4Stepper) = show(io, s)

function Base.show(io::IO, s::LeapfrogStepper{T}) where {T}
    print(io, "LeapfrogStepper{$T}: dt = ", s.dt, ", Robert-Asselin coeff = ", s.ra_coeff)
end
Base.show(io::IO, ::MIME"text/plain", s::LeapfrogStepper) = show(io, s)

# ── SurgeryParams ───────────────────────────────────────

function Base.show(io::IO, p::SurgeryParams{T}) where {T}
    print(io, "SurgeryParams{$T}")
end

function Base.show(io::IO, ::MIME"text/plain", p::SurgeryParams{T}) where {T}
    println(io, "SurgeryParams{$T}")
    println(io, "├── δ (proximity): ", p.delta)
    println(io, "├── μ (min segment): ", p.mu)
    println(io, "├── Δ_max (max segment): ", p.Delta_max)
    println(io, "├── area_min: ", p.area_min)
    print(io,   "└── n_surgery: ", p.n_surgery, " steps")
end
