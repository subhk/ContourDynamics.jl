# remeshing/density.jl — geometry and remeshing stage helpers.

function _prepare_density_sources(sources)
    return map(sources) do source
        (contour=source,
         lengths=arc_lengths(source),
         curvatures=_node_curvatures(source),
         abs_pv=abs(source.pv))
    end
end

function _dritschel_segment_densities(c::PVContour{T}, params::SurgeryParams,
                                      sources=(c,);
                                      _arc_buf::Union{Nothing,Vector{T}}=nothing,
                                      _source_data=nothing) where {T}
    n = nnodes(c)
    densities = Vector{T}(undef, n)
    n == 0 && return densities

    lengths = _arc_buf === nothing ? arc_lengths(c) : _arc_lengths!(_arc_buf, c)
    perimeter = sum(lengths)
    if perimeter <= eps(T)
        fill!(densities, one(T) / T(params.Δ_max))
        return densities
    end

    α = T(2) / T(3)
    δ = T(params.δ)
    μ = T(params.μ)
    Δ_max = T(params.Δ_max)
    L = max(perimeter / T(2π), T(params.Δ_max))
    inv_μL = inv(μ * L)
    sqrt2 = sqrt(T(2))
    d2_floor = eps(T) * max(one(T), L)^2
    source_data = _source_data === nothing ? _prepare_density_sources(sources) : _source_data

    # Dritschel (1988), Eqs. (2a)-(2d).  First form the nonlocal
    # vorticity-weighted curvature K_j at each node of this contour, using all
    # source contours that participate in the same surgery pass.  The resulting
    # transformed node density is averaged onto segments and saturated so the
    # implied spacing cannot fall below δ/sqrt(2).
    node_density_curvatures = Vector{T}(undef, n)
    @inbounds for j in 1:n
        xj = c.nodes[j]
        numerator = zero(T)
        denominator = zero(T)

        for data in source_data
            source = data.contour
            ns = nnodes(source)
            ns == 0 && continue
            source_lengths = data.lengths
            source_curvatures = data.curvatures
            abs_pv = data.abs_pv

            for i in 1:ns
                ei = source_lengths[i]
                ei <= eps(T) && continue
                mid = (source.nodes[i] + next_node(source, i)) / T(2)
                d = xj - mid
                d2 = max(d[1]^2 + d[2]^2, d2_floor)
                weight = ei * abs_pv / d2
                denominator += weight
                numerator += weight * source_curvatures[i]
            end
        end

        K_j = denominator <= eps(T) ? zero(T) : numerator / denominator
        node_density_curvatures[j] = inv_μL * (K_j * L)^α + sqrt2 * K_j
    end

    raw = Vector{T}(undef, n)
    @inbounds for j in 1:n
        κ̃ = (node_density_curvatures[j] + node_density_curvatures[mod1(j + 1, n)]) / T(2)
        raw[j] = κ̃ <= eps(T) ? zero(T) : κ̃ / (one(T) + δ * κ̃ / sqrt2)
    end

    target_intervals = _target_interval_count(perimeter, n, μ, Δ_max)
    _scale_densities!(densities, raw, lengths, target_intervals, μ, Δ_max)
    return densities
end

function _scale_densities!(densities::Vector{T}, raw::Vector{T}, lengths::AbstractVector{T},
                           target_intervals::Int, μ::T, Δ_max::T) where {T}
    # Normalize raw density so its length-weighted integral gives the desired
    # interval count, then clamp to enforce minimum/maximum segment lengths.
    weighted = zero(T)
    @inbounds for i in eachindex(raw, lengths)
        weighted += raw[i] * lengths[i]
    end

    min_density = one(T) / Δ_max
    max_density = one(T) / μ
    scale = weighted <= eps(T) ? min_density : T(target_intervals) / weighted
    @inbounds for i in eachindex(raw)
        densities[i] = clamp(raw[i] * scale, min_density, max_density)
    end
    return densities
end

function _target_interval_count(total_length::T, current::Int, μ::T, Δ_max::T) where {T}
    min_intervals = max(1, ceil(Int, total_length / Δ_max))
    max_intervals = max(min_intervals, floor(Int, total_length / μ))
    return clamp(current, min_intervals, max_intervals)
end
