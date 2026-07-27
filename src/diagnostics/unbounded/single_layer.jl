# Unbounded-domain single-layer diagnostics.

function energy(prob::ContourProblem{EulerKernel, UnboundedDomain, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        E += ci.pv * cj.pv * _energy_contour_pair_euler(ci, cj; _partial=partial)
    end
    return _normalize_energy(E)
end

function _energy_contour_pair_euler(ci::PVContour{T}, cj::PVContour{T};
                                    _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # Clamp r² to eps(T) instead of skipping near-zero separations: adjacent
    # segments share a node where log(r²) diverges, but the integral is finite
    # (integrable singularity). Clamping avoids log(0) while keeping the term.
    Φ = rv -> log(max(rv[1]^2 + rv[2]^2, eps(T))) / 2
    # Coincident segments need the log singularity integrated analytically.
    self_quad = (mid, half_ds, g_nodes, g_weights) -> _log_self_seg_quad(half_ds)
    return _energy_contour_pair(ci, cj, Φ, self_quad; _partial=_partial)
end

function energy(prob::ContourProblem{SQGKernel{T}, UnboundedDomain, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    delta = prob.kernel.delta
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        E += ci.pv * cj.pv * _energy_contour_pair_sqg(ci, cj, delta; _partial=partial)
    end
    # Φδ(r) = sqrt(r²+δ²) - δ log(δ + sqrt(r²+δ²)) satisfies ΔΦδ = 1/sqrt(r²+δ²),
    # matching the regularized SQG kernel.
    return _normalize_energy(E)
end

function _energy_contour_pair_sqg(ci::PVContour{T}, cj::PVContour{T}, delta::T;
                                   _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # No self branch: Φδ(r) is smooth everywhere thanks to the delta regularization.
    Φ = rv -> _sqg_regularized_energy_potential_scalar(rv[1]^2 + rv[2]^2, delta)
    return _energy_contour_pair(ci, cj, Φ; _partial=_partial)
end

function energy(prob::ContourProblem{QGKernel{T}, UnboundedDomain, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    Ld = prob.kernel.Ld
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        E += ci.pv * cj.pv * _energy_contour_pair_qg(ci, cj, Ld; _partial=partial)
    end
    return _normalize_energy(E)
end

function _energy_contour_pair_qg(ci::PVContour{T}, cj::PVContour{T}, Ld::T;
                                  _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # Smooth limit of K₀(r/Ld) + log(r) as r→0
    k0_smooth_at_zero = log(2 * Ld) - T(Base.MathConstants.eulergamma)
    Φ = rv -> begin
        r = sqrt(rv[1]^2 + rv[2]^2)
        # Near-coincident quadrature points contribute nothing (the singular
        # part is handled analytically in the self branch).
        r < eps(T) * Ld ? zero(T) : _besselk0_approx_scalar(r / Ld)
    end
    # Self-segment: singular subtraction. Decompose
    # K₀(r/Ld) = [-log(r)] + [K₀(r/Ld) + log(r)]. The -log(r) part integrates
    # analytically (hence -_log_self_seg_quad); the remainder is smooth and
    # tends to log(2Ld) - γ at r=0, so plain GL quadrature is safe.
    Φ_smooth = rv -> begin
        r2 = rv[1]^2 + rv[2]^2
        r2 < eps(T)^2 && return k0_smooth_at_zero
        r = sqrt(r2)
        return _besselk0_approx_scalar(r / Ld) + log(r)
    end
    self_quad = (mid, half_ds, g_nodes, g_weights) ->
        -_log_self_seg_quad(half_ds) +
        _gl3_pair_quad(mid, half_ds, mid, half_ds, g_nodes, g_weights, Φ_smooth)
    return _energy_contour_pair(ci, cj, Φ, self_quad; _partial=_partial)
end
