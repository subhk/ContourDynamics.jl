# ── Kernels ──────────────────────────────────────────────

"""
    AbstractKernel

Supertype for contour-dynamics Green's-function kernels. Kernel choice
determines the contour-integral law used by [`velocity!`](@ref),
[`segment_velocity`](@ref), and diagnostics such as [`energy`](@ref).
"""
abstract type AbstractKernel end

"""
    EulerKernel <: AbstractKernel

Kernel for 2-D Euler (barotropic) vortex dynamics, using the Green's function
`G(r) = -log(r) / (2π)`.
"""
struct EulerKernel <: AbstractKernel end

"""
    QGKernel{T}(Ld)

Kernel for single-layer quasi-geostrophic dynamics with finite, positive
deformation radius `Ld`. Uses the modified Bessel function Green's function
`K₀(r/Ld)/(2π)`. Use [`EulerKernel`](@ref) for the infinite-radius limit.
"""
struct QGKernel{T<:AbstractFloat} <: AbstractKernel
    Ld::T
    function QGKernel(Ld::T) where {T<:AbstractFloat}
        isfinite(Ld) && Ld > zero(T) || throw(ArgumentError(
            "Deformation radius Ld must be finite and positive, got $Ld"))
        new{T}(Ld)
    end
end

@inline function _qg_modal_eigenvalue_tolerance(eigenvalues::SVector{N,T}) where {N,T}
    scale = maximum(abs, eigenvalues)
    return scale * sqrt(eps(T)) * T(100)
end

@inline _is_barotropic_mode(kernel, λ) =
    abs(λ) <= _qg_modal_eigenvalue_tolerance(kernel.eigenvalues)

"""
    SQGKernel{T}(δ)

Kernel for surface quasi-geostrophic (SQG) dynamics using the contour kernel
`G_C(r) = 1/(2πr)`.

The SQG velocity at a contour boundary is singular (the tangential component
diverges logarithmically), so a **regularization length** `δ > 0` is
required.  The kernel `1/r` is replaced by `1/√(r² + δ²)`.  A typical choice
is `δ ≈ μ`, the minimum segment length used for surgery.

# Sign convention

Positive PV (surface buoyancy) induces counter-clockwise circulation, matching
the `EulerKernel` convention. With `u = (-ψ_y, ψ_x)`, this is the lower-boundary
convention `θ = -(-Δ)^(1/2)ψ`; the corresponding streamfunction Green's
function is `-1/(2πr)` and integration by parts gives the positive contour
kernel above.
"""
struct SQGKernel{T<:AbstractFloat} <: AbstractKernel
    δ::T
    function SQGKernel(δ::T) where {T<:AbstractFloat}
        isfinite(δ) && δ > zero(T) || throw(ArgumentError(
            "Regularization δ must be finite and positive, got $δ"))
        new{T}(δ)
    end
end

# ASCII spelling retained for backwards compatibility.
function Base.getproperty(k::SQGKernel, name::Symbol)
    name === :delta && return getfield(k, :δ)
    return getfield(k, name)
end

Base.propertynames(::SQGKernel; private::Bool=false) = (:δ, :delta)

"""
    MultiLayerQGKernel{N,M,T}(Ld, coupling[, H])
    MultiLayerQGKernel(Ld, coupling; layer_thicknesses=H)

Kernel for `N`-layer quasi-geostrophic dynamics with `M = N-1` deformation radii `Ld`
and physical-layer coupling matrix `coupling`. The optional positive layer
thicknesses `H` define the weighted inner product used by the modal decomposition.

For unequal layer depths, the physical coupling matrix is generally nonsymmetric
but must satisfy `H[i] * coupling[i,j] == H[j] * coupling[j,i]`. The constructor
diagonalizes the symmetric similarity transform
`sqrt(Diagonal(H)) * coupling / sqrt(Diagonal(H))`. If `H` is omitted, symmetric
matrices retain the historical unit weights; for a connected nonsymmetric matrix,
the relative thicknesses are inferred and normalized to have mean one. Pass `H`
explicitly when disconnected blocks leave their relative weights undetermined.
The physical stretching matrix must also annihilate a uniform streamfunction,
`coupling * ones(N) ≈ 0`, so its single null mode is the barotropic mode.

`physical_to_modal` maps layer PV to the weighted modal representation, while
`modal_to_physical` reconstructs layer streamfunction and velocity. The energy is
therefore weighted by the stored `layer_thicknesses`.

!!! note
    The evolution and energy diagnostics derive modal deformation radii from the
    coupling matrix eigenvalues (`Ld_mode = 1/√|λ|`). The constructor therefore
    validates that the provided `Ld` values match the nonzero modal radii implied
    by `coupling`, preventing silent simulation of a different physical model.
"""
struct MultiLayerQGKernel{N, M, T<:AbstractFloat} <: AbstractKernel
    Ld::SVector{M, T}
    coupling::SMatrix{N, N, T}
    layer_thicknesses::SVector{N, T}
    symmetric_coupling::SMatrix{N, N, T}
    eigenvalues::SVector{N, T}
    eigenvectors::SMatrix{N, N, T}
    eigenvectors_inv::SMatrix{N, N, T}
    physical_to_modal::SMatrix{N, N, T}
    modal_to_physical::SMatrix{N, N, T}
end

"""Return the deformation radius associated with a nonbarotropic eigenvalue."""
@inline _modal_deformation_radius(λ::T) where {T<:AbstractFloat} =
    one(T) / sqrt(abs(λ))

"""
    _dispatch_qg_mode(f, kernel, λ, args...)

Call `f(mode_kernel, args...)` with a concrete [`EulerKernel`](@ref) for the
barotropic mode or a concrete [`QGKernel`](@ref) for a baroclinic mode. Keeping
the branch here prevents each velocity and energy implementation from
repeating the eigenvalue classification while preserving specialization in
their inner loops.
"""
@inline function _dispatch_qg_mode(f::F, kernel::MultiLayerQGKernel,
                                   λ::T, args...) where {F,T<:AbstractFloat}
    if _is_barotropic_mode(kernel, λ)
        return f(EulerKernel(), args...)
    end
    return f(QGKernel(_modal_deformation_radius(λ)), args...)
end

@inline function _qg_coupling_symmetry_tolerance(coupling::AbstractMatrix{T}) where {T}
    scale = maximum(abs, coupling)
    return scale * sqrt(eps(T)) * T(100)
end

@inline function _qg_is_symmetric(coupling::AbstractMatrix{T}) where {T}
    return norm(coupling - coupling', Inf) <= _qg_coupling_symmetry_tolerance(coupling)
end

function _infer_qg_layer_thicknesses(coupling::SMatrix{N,N,T}) where {N,T<:AbstractFloat}
    cmat = Matrix(coupling)
    _qg_is_symmetric(cmat) && return ones(SVector{N,T})

    # Diagonal symmetrizability gives H[j]/H[i] = C[i,j]/C[j,i]. Work in
    # logarithms so very unequal depths do not overflow while traversing the
    # coupling graph.
    scale = maximum(abs, cmat)
    edge_tol = scale * eps(T) * T(100)
    logH = fill(T(NaN), N)
    logH[1] = zero(T)
    queue = Int[1]
    cursor = 1
    consistency_tol = sqrt(eps(T)) * T(100)

    while cursor <= length(queue)
        i = queue[cursor]
        cursor += 1
        for j in 1:N
            i == j && continue
            cij = cmat[i, j]
            cji = cmat[j, i]
            max(abs(cij), abs(cji)) <= edge_tol && continue
            min(abs(cij), abs(cji)) > edge_tol || throw(ArgumentError(
                "Coupling matrix is not symmetrizable by positive layer thicknesses: " *
                "C[$i,$j] and C[$j,$i] must either both vanish or both be nonzero."))
            cij * cji > zero(T) || throw(ArgumentError(
                "Coupling matrix is not symmetrizable by positive layer thicknesses: " *
                "C[$i,$j] and C[$j,$i] must have the same sign."))

            candidate = logH[i] + log(abs(cij)) - log(abs(cji))
            if isnan(logH[j])
                logH[j] = candidate
                push!(queue, j)
            elseif !isapprox(logH[j], candidate;
                             rtol=consistency_tol, atol=consistency_tol)
                throw(ArgumentError(
                    "Coupling matrix has inconsistent layer-thickness ratios and " *
                    "cannot be transformed to a symmetric modal representation."))
            end
        end
    end

    all(isfinite, logH) || throw(ArgumentError(
        "Layer-thickness ratios are not uniquely determined by this disconnected " *
        "nonsymmetric coupling matrix; pass `H` (or `layer_thicknesses`) explicitly."))
    shifted = exp.(logH .- maximum(logH))
    inferred = shifted .* (T(N) / sum(shifted))
    return SVector{N,T}(inferred)
end

function _build_multilayer_qg_kernel(
        Ld::SVector{M,T}, coupling::SMatrix{N,N,T},
        H::SVector{N,T}) where {N,M,T<:AbstractFloat}
    M == N - 1 || throw(ArgumentError(
        "Number of deformation radii M=$M must equal N-1=$(N-1)"))
    all(x -> isfinite(x) && x > zero(T), Ld) || throw(ArgumentError(
        "Deformation radii Ld must all be finite and positive"))
    all(isfinite, coupling) || throw(ArgumentError("Coupling matrix entries must be finite"))
    all(x -> isfinite(x) && x > zero(T), H) || throw(ArgumentError(
        "Layer thicknesses H must all be finite and positive"))

    cmat = Matrix(coupling)
    uniform_residual = norm(cmat * ones(T, N), Inf)
    uniform_tolerance = _qg_coupling_symmetry_tolerance(cmat) * T(N)
    uniform_residual <= uniform_tolerance || throw(ArgumentError(
        "Coupling matrix must annihilate a uniform streamfunction (C * 1 = 0); " *
        "got ‖C * 1‖∞ = $uniform_residual."))

    hmat = Diagonal(Vector(H))
    weighted = hmat * cmat
    weighted_asymmetry = norm(weighted - weighted', Inf)
    weighted_asymmetry <= _qg_coupling_symmetry_tolerance(weighted) || throw(ArgumentError(
        "Coupling matrix is not symmetric under the supplied layer-thickness weights; " *
        "got ‖H*C-(H*C)ᵀ‖ = $weighted_asymmetry."))

    sqrtH = sqrt.(Vector(H))
    invsqrtH = inv.(sqrtH)
    transformed = Diagonal(sqrtH) * cmat * Diagonal(invsqrtH)
    # Average the two triangles after tolerance validation so roundoff in
    # independently computed physical coefficients cannot select one side.
    transformed = (transformed + transformed') / T(2)
    symmetric_coupling = SMatrix{N,N,T}(transformed)
    eig = eigen(Symmetric(transformed))
    eigenvalues = SVector{N,T}(eig.values)
    eigenvectors = SMatrix{N,N,T}(eig.vectors)
    # P is orthogonal only in the thickness-weighted representation.
    eigenvectors_inv = SMatrix{N,N,T}(eig.vectors')
    physical_to_modal = SMatrix{N,N,T}(eig.vectors' * Diagonal(sqrtH))
    modal_to_physical = SMatrix{N,N,T}(Diagonal(invsqrtH) * eig.vectors)

    λtol = _qg_modal_eigenvalue_tolerance(eigenvalues)
    for (m, λ) in enumerate(eigenvalues)
        λ > λtol && throw(ArgumentError(
            "Coupling eigenvalue λ[$m] = $λ is positive; the QG stretching " *
            "operator must be negative semidefinite."))
    end

    modal_radii = T[one(T) / sqrt(abs(λ)) for λ in eigenvalues if abs(λ) > λtol]
    length(modal_radii) == M || throw(ArgumentError(
        "Coupling matrix implies $(length(modal_radii)) significant baroclinic mode(s), " *
        "but $(M) deformation radius value(s) were provided. " *
        "Expected one zero barotropic eigenvalue and $(M) nonzero modal eigenvalues."
    ))

    sorted_modal = sort(modal_radii)
    sorted_Ld = sort(Vector(Ld))
    for m in 1:M
        isapprox(sorted_modal[m], sorted_Ld[m];
                 rtol=sqrt(eps(T)) * T(100), atol=eps(T) * T(100)) || throw(ArgumentError(
            "Provided deformation radii $(Vector(Ld)) are inconsistent with the coupling matrix. " *
            "The coupling-implied modal radii are $(modal_radii)."
        ))
    end

    return MultiLayerQGKernel{N,M,T}(
        Ld, coupling, H, symmetric_coupling, eigenvalues, eigenvectors,
        eigenvectors_inv, physical_to_modal, modal_to_physical)
end

function MultiLayerQGKernel(
        Ld::SVector{M,T}, coupling::SMatrix{N,N,T};
        layer_thicknesses=nothing) where {N,M,T<:AbstractFloat}
    H = if layer_thicknesses === nothing
        _infer_qg_layer_thicknesses(coupling)
    else
        length(layer_thicknesses) == N || throw(ArgumentError(
            "Expected $N layer thicknesses, got $(length(layer_thicknesses))"))
        SVector{N,T}(T.(collect(layer_thicknesses)))
    end
    return _build_multilayer_qg_kernel(Ld, coupling, H)
end

function MultiLayerQGKernel(
        Ld::SVector{M,T}, coupling::SMatrix{N,N,T},
        H::SVector{N,T}) where {N,M,T<:AbstractFloat}
    return _build_multilayer_qg_kernel(Ld, coupling, H)
end

function MultiLayerQGKernel(
        Ld::SVector{M,T}, coupling::SMatrix{N,N,T},
        H::AbstractVector) where {N,M,T<:AbstractFloat}
    length(H) == N || throw(ArgumentError("Expected $N layer thicknesses, got $(length(H))"))
    return _build_multilayer_qg_kernel(Ld, coupling, SVector{N,T}(T.(collect(H))))
end

"""
    nlayers(kernel_or_problem)

Return the number of layers in a [`MultiLayerQGKernel`](@ref) or
[`MultiLayerContourProblem`](@ref).
"""
nlayers(::MultiLayerQGKernel{N}) where {N} = N
