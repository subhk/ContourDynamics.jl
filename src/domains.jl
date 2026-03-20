# Domain helper functions

"""Square periodic domain with half-width L."""
PeriodicDomain(L::T) where {T<:Real} = PeriodicDomain(L, L)
