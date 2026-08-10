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
    # The Euler Hamiltonian is -(1/4π)∫∫ q(x)q(y)log|x-y| dxdy.
    # Two applications of Green's theorem convert this area integral to the
    # smooth contour potential r²(2-log(r²))/4 used here.
    Φ = rv -> _euler_energy_potential_scalar(rv[1]^2 + rv[2]^2)
    return _energy_contour_pair(ci, cj, Φ; _partial=_partial)
end

function energy(prob::ContourProblem{SQGKernel{T}, UnboundedDomain, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    δ = prob.kernel.δ
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        E += ci.pv * cj.pv * _energy_contour_pair_sqg(ci, cj, δ; _partial=partial)
    end
    # φδ(r) = sqrt(r²+δ²) - δ log(δ + sqrt(r²+δ²)) satisfies
    # Δφδ = 1/sqrt(r²+δ²). The pair integrand is 2φδ so the shared
    # normalization returns the positive SQG Hamiltonian.
    return _normalize_energy(E)
end

function _energy_contour_pair_sqg(ci::PVContour{T}, cj::PVContour{T}, δ::T;
                                   _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # No self branch: φδ(r) is smooth everywhere thanks to the δ regularization.
    Φ = rv -> _sqg_regularized_energy_potential_scalar(rv[1]^2 + rv[2]^2, δ)
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
    Φ = rv -> _qg_energy_potential_scalar(rv[1]^2 + rv[2]^2, Ld)
    return _energy_contour_pair(ci, cj, Φ; _partial=_partial)
end
