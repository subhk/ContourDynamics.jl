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

Kernel for single-layer quasi-geostrophic dynamics with deformation radius `Ld`.
Uses the modified Bessel function Green's function `K₀(r/Ld)/(2π)`.
"""
struct QGKernel{T<:AbstractFloat} <: AbstractKernel
    Ld::T
    function QGKernel(Ld::T) where {T<:AbstractFloat}
        Ld > zero(T) || throw(ArgumentError("Deformation radius Ld must be positive, got $Ld"))
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
    SQGKernel{T}(delta)

Kernel for surface quasi-geostrophic (SQG) dynamics using the fractional
Laplacian Green's function `G(r) = 1/(2πr)`.

The SQG velocity at a contour boundary is singular (the tangential component
diverges logarithmically), so a **regularization length** `delta > 0` is
required.  The kernel `1/r` is replaced by `1/√(r² + δ²)`.  A typical choice
is `delta ≈ μ`, the minimum segment length used for surgery.

# Sign convention

Positive PV (surface buoyancy) induces counter-clockwise circulation, matching
the `EulerKernel` convention.
"""
struct SQGKernel{T<:AbstractFloat} <: AbstractKernel
    delta::T
    function SQGKernel(delta::T) where {T<:AbstractFloat}
        delta > zero(T) || throw(ArgumentError("Regularization δ must be positive, got $delta"))
        new{T}(delta)
    end
end

"""
    MultiLayerQGKernel{N,M,T}(Ld, coupling)

Kernel for `N`-layer quasi-geostrophic dynamics with `M = N-1` deformation radii `Ld`
and layer coupling matrix `coupling`.  The constructor eigen-decomposes the coupling
matrix for efficient velocity evaluation.

The coupling matrix should already incorporate layer thicknesses (i.e. it is the
full stretching operator, not raw interface stretching coefficients).
It must be symmetric and negative semidefinite, with one barotropic zero mode.

!!! note
    The evolution and energy diagnostics derive modal deformation radii from the
    coupling matrix eigenvalues (`Ld_mode = 1/√|λ|`). The constructor therefore
    validates that the provided `Ld` values match the nonzero modal radii implied
    by `coupling`, preventing silent simulation of a different physical model.
"""
struct MultiLayerQGKernel{N, M, T<:AbstractFloat} <: AbstractKernel
    Ld::SVector{M, T}
    coupling::SMatrix{N, N, T}
    eigenvalues::SVector{N, T}
    eigenvectors::SMatrix{N, N, T}
    eigenvectors_inv::SMatrix{N, N, T}
    function MultiLayerQGKernel(Ld::SVector{M, T}, coupling::SMatrix{N, N, T}) where {N, M, T<:AbstractFloat}
        M == N - 1 || throw(ArgumentError("Number of deformation radii M=$M must equal N-1=$(N-1)"))
        all(>(zero(T)), Ld) || throw(ArgumentError("Deformation radii Ld must all be positive"))
        cmat = Matrix(coupling)
        issymmetric(cmat) || throw(ArgumentError("Coupling matrix must be symmetric; got asymmetry ‖C-Cᵀ‖ = $(norm(cmat - cmat', Inf))"))
        eig = eigen(Symmetric(cmat))
        eigenvalues = SVector{N,T}(eig.values)
        eigenvectors = SMatrix{N,N,T}(eig.vectors)
        # eigen(Symmetric(...)) returns orthonormal eigenvectors, so P⁻¹ = Pᵀ.
        # Using transpose is both faster and more numerically stable than inv().
        eigenvectors_inv = SMatrix{N,N,T}(eig.vectors')

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

        new{N, M, T}(Ld, coupling, eigenvalues, eigenvectors, eigenvectors_inv)
    end
end

function MultiLayerQGKernel(Ld::SVector{M, T}, coupling::SMatrix{N, N, T}, H::SVector{N, T}) where {N, M, T<:AbstractFloat}
    Base.depwarn("MultiLayerQGKernel(Ld, coupling, H) is deprecated; H (layer thicknesses) " *
                 "is not used in dynamics. Use MultiLayerQGKernel(Ld, coupling) instead.", :MultiLayerQGKernel)
    MultiLayerQGKernel(Ld, coupling)
end

"""
    nlayers(kernel_or_problem)

Return the number of layers in a [`MultiLayerQGKernel`](@ref) or
[`MultiLayerContourProblem`](@ref).
"""
nlayers(::MultiLayerQGKernel{N}) where {N} = N
