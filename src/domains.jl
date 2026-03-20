# Domain helper functions

"""Square periodic domain with half-width L."""
PeriodicDomain(L::T) where {T<:AbstractFloat} = PeriodicDomain(L, L)
