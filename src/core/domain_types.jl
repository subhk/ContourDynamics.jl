# ── Domains ──────────────────────────────────────────────

"""
    AbstractDomain

Supertype for spatial domains. Domains control boundary handling for velocity
evaluation, remeshing/surgery proximity checks, and node wrapping.
"""
abstract type AbstractDomain end

"""
    UnboundedDomain <: AbstractDomain

An infinite, unbounded two-dimensional domain.
"""
struct UnboundedDomain <: AbstractDomain end

"""
    PeriodicDomain{T}(Lx, Ly)

A doubly-periodic rectangular domain with half-widths `Lx` and `Ly`,
i.e. the domain `[-Lx, Lx) × [-Ly, Ly)`.
"""
struct PeriodicDomain{T<:AbstractFloat} <: AbstractDomain
    Lx::T
    Ly::T
    function PeriodicDomain(Lx::T, Ly::T) where {T<:AbstractFloat}
        (isfinite(Lx) && isfinite(Ly) && Lx > zero(T) && Ly > zero(T)) ||
            throw(ArgumentError("Domain half-widths must be finite and positive"))
        new{T}(Lx, Ly)
    end
end
