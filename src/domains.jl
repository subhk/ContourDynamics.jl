# Domain helper functions

"""Square periodic domain with half-width L."""
PeriodicDomain(L::T) where {T<:AbstractFloat} = PeriodicDomain(L, L)

# Allow non-AbstractFloat Real args (e.g. Irrational like π) by converting to Float64.
# Only triggers when args are NOT already AbstractFloat (avoids ambiguity with inner constructor).
function PeriodicDomain(Lx::Real, Ly::Real)
    Lx_f = Float64(Lx)
    Ly_f = Float64(Ly)
    return PeriodicDomain(Lx_f, Ly_f)
end

PeriodicDomain(L::Real) = PeriodicDomain(Float64(L), Float64(L))
