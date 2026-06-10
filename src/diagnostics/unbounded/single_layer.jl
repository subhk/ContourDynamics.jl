# Unbounded-domain single-layer diagnostics.

function energy(prob::ContourProblem{EulerKernel, UnboundedDomain, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    inv4pi = one(T) / (4 * T(π))
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        E += ci.pv * cj.pv * _energy_contour_pair_euler(ci, cj; _partial=partial)
    end
    # Factor 1/2: the double sum counts both (i,j) and (j,i) for the symmetric integrand.
    return -inv4pi * E / 2
end

function _energy_contour_pair_euler(ci::PVContour{T}, cj::PVContour{T};
                                    _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    is_self = ci.nodes === cj.nodes  # detect self-interaction
    # 3-point Gauss-Legendre nodes/weights on [-1,1]
    g_nodes, g_weights = _gl3_nodes_weights(T)
    # Analytical self-segment integral:
    # ∫₋₁¹∫₋₁¹ log(|s-t| * |half_ds|) ds dt = 4*log(2) - 6 + 4*log|half_ds|
    self_seg_const = 4 * log(T(2)) - T(6)  # precompute constant part
    # Thread over outer segments, each thread accumulates a partial sum.
    return @_energy_segment_loop partial _partial nci for i in 1:nci
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
                # Self-segment: log singularity requires analytical integration.
                # ∫₋₁¹∫₋₁¹ log(r²)/2 ds dt where r = |s-t|*|half_ds|
                #   = ∫₋₁¹∫₋₁¹ (log|s-t| + log|half_ds|) ds dt
                #   = (4*log(2) - 6) + 4*log|half_ds|
                half_ds_len = sqrt(half_dsi[1]^2 + half_dsi[2]^2)
                if half_ds_len > eps(T)
                    quad = self_seg_const + 4 * log(half_ds_len)
                else
                    quad = zero(T)
                end
            else
                # 3×3 Gauss-Legendre quadrature over both segments.
                # Use max(r2, eps(T)) instead of skipping near-zero r2:
                # adjacent segments share a node where log(r²) diverges,
                # but the integral is finite (integrable singularity).
                # Clamping avoids log(0) while preserving the contribution.
                quad = zero(T)
                for qi in 1:3
                    pi_pt = midi + g_nodes[qi] * half_dsi
                    for qj in 1:3
                        pj_pt = midj + g_nodes[qj] * half_dsj
                        dx = pi_pt[1] - pj_pt[1]
                        dy = pi_pt[2] - pj_pt[2]
                        r2 = max(dx^2 + dy^2, eps(T))
                        quad += g_weights[qi] * g_weights[qj] * log(r2) / 2
                    end
                end
            end
            # Jacobian: each ∫₋₁¹ → ½ ∫₀¹, two of them → ¼
            local_s += quad / 4 * dot_ds
        end
        partial[i] = local_s
    end
end

function energy(prob::ContourProblem{SQGKernel{T}, UnboundedDomain, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    delta = prob.kernel.delta
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        E += ci.pv * cj.pv * _energy_contour_pair_sqg(ci, cj, delta; _partial=partial)
    end
    # E_SQG = -(1/(4π)) × (1/2) × Σ q_i q_j ∮∮ Φδ(r) ds·ds'
    # where Φδ(r) = sqrt(r²+δ²) - δ log(δ + sqrt(r²+δ²)).
    # This satisfies ΔΦδ = 1/sqrt(r²+δ²), matching the regularized SQG kernel.
    inv4pi = one(T) / (4 * T(π))
    return -inv4pi * E / 2
end

function _energy_contour_pair_sqg(ci::PVContour{T}, cj::PVContour{T}, delta::T;
                                   _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    # 3-point Gauss-Legendre nodes/weights on [-1,1]
    g_nodes, g_weights = _gl3_nodes_weights(T)
    return @_energy_segment_loop partial _partial nci for i in 1:nci
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
            # 3×3 Gauss-Legendre quadrature — Φδ(r) is smooth everywhere.
            quad = zero(T)
            for qi in 1:3
                pi_pt = midi + g_nodes[qi] * half_dsi
                for qj in 1:3
                    pj_pt = midj + g_nodes[qj] * half_dsj
                    dx = pi_pt[1] - pj_pt[1]
                    dy = pi_pt[2] - pj_pt[2]
                    quad += g_weights[qi] * g_weights[qj] *
                        _sqg_regularized_energy_potential_scalar(dx^2 + dy^2, delta)
                end
            end
            # Jacobian: each ∫₋₁¹ → ½ ∫₀¹, two of them → ¼
            local_s += quad / 4 * dot_ds
        end
        partial[i] = local_s
    end
end

function energy(prob::ContourProblem{QGKernel{T}, UnboundedDomain, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    Ld = prob.kernel.Ld
    inv4pi = one(T) / (4 * T(π))
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        E += ci.pv * cj.pv * _energy_contour_pair_qg(ci, cj, Ld; _partial=partial)
    end
    return -inv4pi * E / 2
end

function _energy_contour_pair_qg(ci::PVContour{T}, cj::PVContour{T}, Ld::T;
                                  _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    is_self = ci.nodes === cj.nodes  # detect self-interaction
    # 3-point Gauss-Legendre nodes/weights on [-1,1]
    g_nodes, g_weights = _gl3_nodes_weights(T)
    # Analytical self-segment integral for the log(r²)/2 singularity
    # (same formula as Euler self-segment)
    self_seg_const = 4 * log(T(2)) - T(6)
    # Smooth limit of K₀(r/Ld) + log(r) as r→0
    k0_smooth_at_zero = log(2 * Ld) - T(Base.MathConstants.eulergamma)
    return @_energy_segment_loop partial _partial nci for i in 1:nci
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
                # Decompose K₀(r/Ld) = [-log(r)] + [K₀(r/Ld) + log(r)]
                # 1) The -log(r) part has a known analytical integral:
                #    ∫₋₁¹∫₋₁¹ log(|s-t|·|half_ds|) ds dt = self_seg_const + 4·log|half_ds|
                half_ds_len = sqrt(half_dsi[1]^2 + half_dsi[2]^2)
                if half_ds_len > eps(T)
                    quad_log = self_seg_const + 4 * log(half_ds_len)
                else
                    quad_log = zero(T)
                end
                # 2) The smooth remainder K₀(r/Ld) + log(r) → log(2Ld) - γ at r=0
                #    is safe for GL quadrature.
                quad_smooth = zero(T)
                for qi in 1:3
                    pi_pt = midi + g_nodes[qi] * half_dsi
                    for qj in 1:3
                        pj_pt = midj + g_nodes[qj] * half_dsj
                        dx = pi_pt[1] - pj_pt[1]
                        dy = pi_pt[2] - pj_pt[2]
                        r2 = dx^2 + dy^2
                        if r2 < eps(T)^2
                            # qi == qj: use smooth limit
                            quad_smooth += g_weights[qi] * g_weights[qj] * k0_smooth_at_zero
                        else
                            r = sqrt(r2)
                            quad_smooth += g_weights[qi] * g_weights[qj] *
                                (_besselk0_approx_scalar(r / Ld) + log(r))
                        end
                    end
                end
                # Combined: K₀(r/Ld) = [-log(r)] + [K₀(r/Ld) + log(r)]
                # quad_log = ∫∫ log(r) ds dt (positive).
                # The -log(r) part contributes -quad_log.
                # The smooth part contributes +quad_smooth.
                quad = -quad_log + quad_smooth
            else
                # 3×3 Gauss-Legendre quadrature over both segments
                quad = zero(T)
                for qi in 1:3
                    pi_pt = midi + g_nodes[qi] * half_dsi
                    for qj in 1:3
                        pj_pt = midj + g_nodes[qj] * half_dsj
                        dx = pi_pt[1] - pj_pt[1]
                        dy = pi_pt[2] - pj_pt[2]
                        r = sqrt(dx^2 + dy^2)
                        r < eps(T) * Ld && continue
                        quad += g_weights[qi] * g_weights[qj] * _besselk0_approx_scalar(r / Ld)
                    end
                end
            end
            local_s += quad / 4 * dot_ds
        end
        partial[i] = local_s
    end
end
