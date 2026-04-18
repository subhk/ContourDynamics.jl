# Periodic-domain helper routines shared by diagnostics.

"""
    _eval_ewald_greens(r_vec, cache, domain)

Evaluate the periodic Green's function at separation `r_vec` using Ewald summation.
Returns `G_per(r)` = real-space Ewald sum + Fourier-space sum.

!!! note
    The central-image real-space term (E₁(α²r²)) diverges at `r = 0`.
    This function silently skips that term when `r² < eps(T)`, so the
    returned value is *not* valid at zero separation.  Callers that need
    the self-interaction limit must handle `r = 0` separately (see
    `_energy_contour_pair_euler_periodic` for an example).
"""
function _eval_ewald_greens(r_vec::SVector{2,T}, cache::EwaldCache{T},
                            domain::PeriodicDomain{T}) where {T}
    alpha = cache.alpha
    Lx, Ly = domain.Lx, domain.Ly
    inv4pi = one(T) / (4 * T(π))
    G_val = zero(T)

    # Real-space sum
    for px in -cache.n_images:cache.n_images
        for py in -cache.n_images:cache.n_images
            shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
            rv = r_vec - shift
            r2 = rv[1]^2 + rv[2]^2
            if r2 > eps(T)
                G_val += inv4pi * _expint_e1(alpha^2 * r2)
            end
        end
    end

    # Fourier-space sum
    for (mi, kxi) in enumerate(cache.kx)
        for (ni, kyi) in enumerate(cache.ky)
            coeff = cache.fourier_coeffs[mi, ni]
            abs(coeff) < eps(T) && continue
            phase = kxi * r_vec[1] + kyi * r_vec[2]
            G_val += coeff * cos(phase)
        end
    end

    return G_val
end

function _energy_contour_pair_euler_periodic(ci::PVContour{T}, cj::PVContour{T},
                                              cache::EwaldCache{T},
                                              domain::PeriodicDomain{T};
                                              _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    is_self = ci.nodes === cj.nodes
    # 3-point Gauss-Legendre nodes/weights on [-1,1]
    g_nodes, g_weights = _gl3_nodes_weights(T)
    # Analytical self-segment integral for the log(r²)/2 singularity
    self_seg_const = 4 * log(T(2)) - T(6)

    # Precompute the limit of [-2π G_per(r) - log(r²)/2] as r→0.
    # This is the smooth periodic correction at zero separation, needed for
    # self-segment GL points where both quadrature indices coincide (r=0).
    corr_at_zero = zero(T)
    if is_self
        alpha = cache.alpha
        Lx, Ly = domain.Lx, domain.Ly
        gamma_euler = T(Base.MathConstants.eulergamma)
        # Central image: lim_{r→0} [-(1/2) E₁(α²r²) - log(r²)/2] = (γ + 2ln(α))/2
        corr_at_zero = (gamma_euler + 2 * log(alpha)) / 2
        # Non-central real-space images evaluated at r=0
        for px in -cache.n_images:cache.n_images
            for py in -cache.n_images:cache.n_images
                (px == 0 && py == 0) && continue
                shift_r2 = (2 * Lx * px)^2 + (2 * Ly * py)^2
                corr_at_zero -= _expint_e1(alpha^2 * shift_r2) / 2
            end
        end
        # Fourier-space sum at r=0 (cos(k·0) = 1)
        for (mi, kxi) in enumerate(cache.kx)
            for (ni, kyi) in enumerate(cache.ky)
                coeff = cache.fourier_coeffs[mi, ni]
                abs(coeff) < eps(T) && continue
                corr_at_zero -= 2 * T(π) * coeff
            end
        end
    end

    partial = _partial
    @inbounds for k in 1:nci; partial[k] = zero(T); end
    @_maybe_threads nci >= _THREADING_THRESHOLD for i in 1:nci
        ai = ci.nodes[i]
        bi = next_node(ci, i)
        dsi = bi - ai
        midi = (ai + bi) / 2
        half_dsi = dsi / 2
        local_s = zero(T)
        for j in 1:ncj
            aj = cj.nodes[j]
            bj = next_node(cj, j)
            dsj = bj - aj
            midj = (aj + bj) / 2
            half_dsj = dsj / 2
            dot_ds = dsi[1] * dsj[1] + dsi[2] * dsj[2]

            if is_self && i == j
                # Self-segment: singular subtraction.
                # 1) Analytical integral of log(r²)/2 (same as unbounded)
                half_ds_len = sqrt(half_dsi[1]^2 + half_dsi[2]^2)
                if half_ds_len > eps(T)
                    quad_analytical = self_seg_const + 4 * log(half_ds_len)
                else
                    quad_analytical = zero(T)
                end
                # 2) Smooth periodic correction [-2π G_per(r) - log(r²)/2] via GL
                quad_corr = zero(T)
                for qi in 1:3
                    pi_pt = midi + g_nodes[qi] * half_dsi
                    for qj in 1:3
                        pj_pt = midj + g_nodes[qj] * half_dsj
                        r_vec = SVector{2,T}(pi_pt[1] - pj_pt[1], pi_pt[2] - pj_pt[2])
                        r2 = r_vec[1]^2 + r_vec[2]^2
                        if r2 > eps(T)
                            G_per = _eval_ewald_greens(r_vec, cache, domain)
                            quad_corr += g_weights[qi] * g_weights[qj] * (-2 * T(π) * G_per - log(r2) / 2)
                        else
                            # qi == qj: use precomputed finite limit
                            quad_corr += g_weights[qi] * g_weights[qj] * corr_at_zero
                        end
                    end
                end
                quad = quad_analytical + quad_corr
            else
                quad = zero(T)
                Lx2 = 2 * domain.Lx
                Ly2 = 2 * domain.Ly
                for qi in 1:3
                    pi_pt = midi + g_nodes[qi] * half_dsi
                    for qj in 1:3
                        pj_pt = midj + g_nodes[qj] * half_dsj
                        r_raw = SVector{2,T}(pi_pt[1] - pj_pt[1], pi_pt[2] - pj_pt[2])
                        # Minimum-image wrap for Ewald convergence (matches velocity path)
                        r_vec = SVector{2,T}(
                            r_raw[1] - round(r_raw[1] / Lx2) * Lx2,
                            r_raw[2] - round(r_raw[2] / Ly2) * Ly2)
                        # Replace log(r²)/2 with the periodic equivalent: -2π * G_per
                        # since log(r²)/2 = -2π * G_∞ for unbounded Euler.
                        G_per = _eval_ewald_greens(r_vec, cache, domain)
                        quad += g_weights[qi] * g_weights[qj] * (-2 * T(π) * G_per)
                    end
                end
            end
            local_s += quad / 4 * dot_ds
        end
        partial[i] = local_s
    end
    return sum(@view partial[1:nci])
end
