using Printf

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
