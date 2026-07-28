# API Reference: Periodic & Ewald

## Ewald Summation

`n_fourier` and `n_images` are truncation counts and must be non-negative.
Passing a negative value to `build_ewald_cache` or `setup_ewald_cache!` throws
`ArgumentError`; zero is valid and is useful for deliberately minimal caches in
tests or controlled approximations.

```@docs
EwaldCache
build_ewald_cache
setup_ewald_cache!
clear_ewald_cache!
```

## Periodic Domains

```@docs
wrap_nodes!
```
