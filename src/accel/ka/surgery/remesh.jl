# Dritschel weighted remeshing: node density from curvature, cubic
# interpolation onto the new node distribution, corner promotion/demotion, and
# area-preserving correction.

@inline function _flat_target_interval_count(total_length, current::Int, μ, Δ_max)
    min_intervals = max(1, Int(ceil(total_length / Δ_max)))
    max_intervals = max(min_intervals, Int(floor(total_length / μ)))
    return clamp(current, min_intervals, max_intervals)
end

@inline function _flat_closed_remesh_interval_count(total_length, q, μ, Δ_max)
    min_intervals = max(3, Int(ceil(total_length / Δ_max)))
    max_intervals = max(min_intervals, Int(floor(total_length / μ)))
    return clamp(Int(round(q)), min_intervals, max_intervals)
end

@inline function _flat_fixed_span_interval_count(total_length, q, current::Int,
                                                 μ, Δ_max)
    if total_length <= eps(typeof(total_length))
        return max(1, current)
    end
    min_intervals = max(1, Int(ceil(total_length / Δ_max)))
    max_intervals = max(min_intervals, Int(floor(total_length / μ)))
    target_intervals = _flat_target_interval_count(total_length, current, μ, Δ_max)
    n_intervals = clamp(Int(round(q)), min_intervals, max_intervals)
    return max(1, n_intervals, target_intervals)
end

@inline function _flat_signed_node_curvature(x, y, wrapx, wrapy, offsets,
                                             lengths, corners, ci, li)
    n = lengths[ci]
    n < 3 && return zero(eltype(x))
    off = offsets[ci]
    prev_i = li == 1 ? n : li - 1
    next_i = li == n ? 1 : li + 1
    (!iszero(corners[off + prev_i - 1]) ||
     !iszero(corners[off + li - 1]) ||
     !iszero(corners[off + next_i - 1])) && return zero(eltype(x))

    g = off + li - 1
    curr_x = x[g]
    curr_y = y[g]
    prev_x = li == 1 ? x[off + n - 1] - wrapx[ci] : x[g - 1]
    prev_y = li == 1 ? y[off + n - 1] - wrapy[ci] : y[g - 1]
    next_x = li == n ? x[off] + wrapx[ci] : x[g + 1]
    next_y = li == n ? y[off] + wrapy[ci] : y[g + 1]

    ax = curr_x - prev_x
    ay = curr_y - prev_y
    bx = next_x - curr_x
    by = next_y - curr_y
    chord_x = next_x - prev_x
    chord_y = next_y - prev_y
    a_norm = sqrt(ax * ax + ay * ay)
    b_norm = sqrt(bx * bx + by * by)
    chord_norm = sqrt(chord_x * chord_x + chord_y * chord_y)
    denom = a_norm * b_norm * chord_norm
    denom <= eps(typeof(denom)) && return zero(denom)
    return 2 * (ax * by - ay * bx) / denom
end

@kernel function _demote_obtuse_state_corners_kernel!(corners, x, y, wrapx,
                                                      wrapy, offsets, lengths,
                                                      contour_of_node,
                                                      local_index, total_nodes)
    g = @index(Global)
    if g <= total_nodes && !iszero(corners[g])
        ci = contour_of_node[g]
        li = local_index[g]
        n = lengths[ci]
        if n >= 3
            off = offsets[ci]
            curr_x = x[g]
            curr_y = y[g]
            prev_x = li == 1 ? x[off + n - 1] - wrapx[ci] : x[g - 1]
            prev_y = li == 1 ? y[off + n - 1] - wrapy[ci] : y[g - 1]
            next_x = li == n ? x[off] + wrapx[ci] : x[g + 1]
            next_y = li == n ? y[off] + wrapy[ci] : y[g + 1]
            vprev_x = prev_x - curr_x
            vprev_y = prev_y - curr_y
            vnext_x = next_x - curr_x
            vnext_y = next_y - curr_y
            if vprev_x * vnext_x + vprev_y * vnext_y < zero(curr_x)
                corners[g] = UInt8(0)
            end
        end
    end
end

@kernel function _promote_high_curvature_state_corners_kernel!(
    out_corners, corners, x, y, wrapx, wrapy, offsets, lengths,
    δ, ncontours)
    ci = @index(Global)
    if ci <= ncontours
        off = offsets[ci]
        n = lengths[ci]
        @inbounds for li in 1:n
            g = off + li - 1
            out_corners[g] = corners[g]
        end
        if n >= 3 && iszero(wrapx[ci]) && iszero(wrapy[ci])
            @inbounds for li in 1:n
                g = off + li - 1
                iszero(out_corners[g]) || continue
                prev_i = li == 1 ? n : li - 1
                next_i = li == n ? 1 : li + 1
                prev_g = off + prev_i - 1
                next_g = off + next_i - 1
                (iszero(out_corners[prev_g]) && iszero(out_corners[next_g])) || continue
                κ = _flat_signed_node_curvature(x, y, wrapx, wrapy, offsets,
                                                 lengths, corners, ci, li)
                abs(κ) >= inv(δ) || continue
                curr_x = x[g]
                curr_y = y[g]
                prev_x = li == 1 ? x[off + n - 1] - wrapx[ci] : x[g - 1]
                prev_y = li == 1 ? y[off + n - 1] - wrapy[ci] : y[g - 1]
                next_x = li == n ? x[off] + wrapx[ci] : x[g + 1]
                next_y = li == n ? y[off] + wrapy[ci] : y[g + 1]
                v1x = prev_x - curr_x
                v1y = prev_y - curr_y
                v2x = next_x - curr_x
                v2y = next_y - curr_y
                if v1x * v2x + v1y * v2y > zero(curr_x)
                    out_corners[g] = UInt8(1)
                end
            end
        end
    end
end

function _demote_obtuse_corners!(state::DeviceContourState,
                                 dev::AbstractDevice=CPU())
    total_nodes = length(state.x)
    total_nodes == 0 && return state
    @_ka_launch dev total_nodes _demote_obtuse_state_corners_kernel!(
        state.corners, state.x, state.y, state.wrapx, state.wrapy,
        state.offsets, state.lengths, state.contour_of_node,
        state.local_index, total_nodes)
    return state
end

function _promote_high_curvature_corners!(state::DeviceContourState{T}, δ,
                                          dev::AbstractDevice=CPU()) where {T}
    total_nodes = length(state.x)
    out_corners = device_zeros(dev, UInt8, total_nodes)
    ncontours = length(state.lengths)
    if ncontours > 0
        @_ka_launch dev ncontours _promote_high_curvature_state_corners_kernel!(
            out_corners, state.corners, state.x, state.y, state.wrapx,
            state.wrapy, state.offsets, state.lengths, T(δ), ncontours)
    end
    state.corners = out_corners
    return state
end

@kernel function _remesh_input_geometry_kernel!(seg_lengths, signed_curvatures,
                                                abs_curvatures, perimeters,
                                                target_area, x, y, pv, wrapx,
                                                wrapy, offsets, lengths,
                                                contour_of_node, local_index,
                                                corners, total_nodes,
                                                ncontours)
    g = @index(Global)
    if g <= total_nodes
        ci = contour_of_node[g]
        li = local_index[g]
        off = offsets[ci]
        ax = x[g]
        ay = y[g]
        bx = li < lengths[ci] ? x[g + 1] : x[off] + wrapx[ci]
        by = li < lengths[ci] ? y[g + 1] : y[off] + wrapy[ci]
        dx = bx - ax
        dy = by - ay
        seg_lengths[g] = sqrt(dx * dx + dy * dy)
        κ = _flat_signed_node_curvature(x, y, wrapx, wrapy, offsets,
                                         lengths, corners, ci, li)
        signed_curvatures[g] = κ
        abs_curvatures[g] = abs(κ)
    end

    if g <= ncontours
        ci = g
        off = offsets[ci]
        n = lengths[ci]
        perimeter = zero(eltype(perimeters))
        area2 = zero(eltype(perimeters))
        @inbounds for li in 1:n
            gi = off + li - 1
            nx = li < n ? x[gi + 1] : x[off] + wrapx[ci]
            ny = li < n ? y[gi + 1] : y[off] + wrapy[ci]
            dx = nx - x[gi]
            dy = ny - y[gi]
            perimeter += sqrt(dx * dx + dy * dy)
            area2 += x[gi] * ny - nx * y[gi]
        end
        perimeters[ci] = perimeter
        target_area[ci] = area2 / 2
    end
end

@kernel function _remesh_node_density_kernel!(node_density_curvatures, x, y, pv,
                                              wrapx, wrapy, offsets, lengths,
                                              contour_of_node, local_index,
                                              seg_lengths, abs_curvatures,
                                              perimeters, μ, Δ_max,
                                              total_nodes)
    g = @index(Global)
    if g <= total_nodes
        ci = contour_of_node[g]
        xj = x[g]
        yj = y[g]
        L = max(perimeters[ci] / eltype(x)(2π), Δ_max)
        d2_floor = eps(typeof(L)) * max(one(L), L)^2
        numerator = zero(L)
        denominator = zero(L)

        @inbounds for sg in 1:total_nodes
            sc = contour_of_node[sg]
            sli = local_index[sg]
            off = offsets[sc]
            ei = seg_lengths[sg]
            ei <= eps(typeof(ei)) && continue
            mx = sli < lengths[sc] ? (x[sg] + x[sg + 1]) / 2 :
                 (x[sg] + x[off] + wrapx[sc]) / 2
            my = sli < lengths[sc] ? (y[sg] + y[sg + 1]) / 2 :
                 (y[sg] + y[off] + wrapy[sc]) / 2
            dx = xj - mx
            dy = yj - my
            d2 = max(dx * dx + dy * dy, d2_floor)
            weight = ei * abs(pv[sc]) / d2
            denominator += weight
            numerator += weight * abs_curvatures[sg]
        end

        K_j = denominator <= eps(typeof(denominator)) ? zero(denominator) : numerator / denominator
        α = eltype(x)(2) / eltype(x)(3)
        sqrt2 = sqrt(eltype(x)(2))
        node_density_curvatures[g] = inv(μ * L) * (K_j * L)^α + sqrt2 * K_j
    end
end

@kernel function _remesh_raw_density_kernel!(raw_densities, x, offsets, lengths,
                                             contour_of_node, local_index,
                                             node_density_curvatures, δ,
                                             total_nodes)
    g = @index(Global)
    if g <= total_nodes
        ci = contour_of_node[g]
        li = local_index[g]
        next_g = li < lengths[ci] ? g + 1 : offsets[ci]
        sqrt2 = sqrt(eltype(x)(2))
        κ̃ = (node_density_curvatures[g] + node_density_curvatures[next_g]) / 2
        raw_densities[g] = κ̃ <= eps(typeof(κ̃)) ? zero(κ̃) :
                           κ̃ / (one(κ̃) + δ * κ̃ / sqrt2)
    end
end

@kernel function _remesh_density_scale_kernel!(density_scale, x, offsets,
                                               lengths, seg_lengths,
                                               raw_densities, perimeters,
                                               μ, Δ_max, ncontours)
    ci = @index(Global)
    if ci <= ncontours
        off = offsets[ci]
        n = lengths[ci]
        weighted = zero(eltype(x))
        @inbounds for li in 1:n
            gi = off + li - 1
            weighted += raw_densities[gi] * seg_lengths[gi]
        end

        min_density = one(eltype(x)) / Δ_max
        target_intervals = _flat_target_interval_count(perimeters[ci], n, μ, Δ_max)
        density_scale[ci] = weighted <= eps(typeof(weighted)) ?
                            min_density :
                            eltype(x)(target_intervals) / weighted
    end
end

@kernel function _remesh_measure_kernel!(densities, measure_start,
                                         q_measure, out_lengths,
                                         remesh_mode,
                                         out_pv, out_wrapx, out_wrapy,
                                         raw_densities, density_scale,
                                         corners,
                                         seg_lengths, perimeters,
                                         in_pv, in_wrapx, in_wrapy,
                                         offsets, lengths, μ, Δ_max,
                                         ncontours)
    ci = @index(Global)
    if ci <= ncontours
        off = offsets[ci]
        n = lengths[ci]
        min_density = one(eltype(densities)) / Δ_max
        max_density = one(eltype(densities)) / μ
        measure = zero(eltype(densities))
        @inbounds for li in 1:n
            g = off + li - 1
            density = clamp(raw_densities[g] * density_scale[ci],
                            min_density, max_density)
            densities[g] = density
            measure_start[g] = measure
            measure += seg_lengths[g] * density
        end
        q_measure[ci] = measure
        out_pv[ci] = in_pv[ci]
        out_wrapx[ci] = in_wrapx[ci]
        out_wrapy[ci] = in_wrapy[ci]

        corner_count = 0
        @inbounds for li in 1:n
            corner_count += Int(!iszero(corners[off + li - 1]))
        end

        fixed_corners = corner_count > 0 && iszero(in_wrapx[ci]) && iszero(in_wrapy[ci])
        if fixed_corners
            total_out = 0
            first_corner = 0
            prev_corner = 0
            @inbounds for li in 1:n
                if !iszero(corners[off + li - 1])
                    first_corner == 0 && (first_corner = li)
                    if prev_corner != 0
                        span_len = zero(eltype(densities))
                        span_measure = zero(eltype(densities))
                        span_segments = 0
                        seg = prev_corner
                        while seg != li
                            gi = off + seg - 1
                            span_len += seg_lengths[gi]
                            span_measure += seg_lengths[gi] * densities[gi]
                            span_segments += 1
                            seg = seg < n ? seg + 1 : 1
                        end
                        total_out += _flat_fixed_span_interval_count(
                            span_len, span_measure, span_segments, μ, Δ_max)
                    end
                    prev_corner = li
                end
            end
            if prev_corner != 0
                span_len = zero(eltype(densities))
                span_measure = zero(eltype(densities))
                span_segments = 0
                seg = prev_corner
                while seg != first_corner
                    gi = off + seg - 1
                    span_len += seg_lengths[gi]
                    span_measure += seg_lengths[gi] * densities[gi]
                    span_segments += 1
                    seg = seg < n ? seg + 1 : 1
                end
                if prev_corner == first_corner
                    @inbounds for li in 1:n
                        gi = off + li - 1
                        span_len += seg_lengths[gi]
                        span_measure += seg_lengths[gi] * densities[gi]
                    end
                    span_segments = n
                end
                total_out += _flat_fixed_span_interval_count(
                    span_len, span_measure, span_segments, μ, Δ_max)
            end

            if total_out >= 3
                remesh_mode[ci] = UInt8(1)
                out_lengths[ci] = total_out
            else
                remesh_mode[ci] = UInt8(2)
                out_lengths[ci] = n
            end
        else
            remesh_mode[ci] = UInt8(0)
            out_lengths[ci] = _flat_closed_remesh_interval_count(perimeters[ci],
                                                                 measure, μ, Δ_max)
        end
    end
end

@inline function _flat_cubic_point_xy(ax, ay, bx, by, κa, κb, p)
    tx = bx - ax
    ty = by - ay
    e = sqrt(tx * tx + ty * ty)
    e <= eps(typeof(e)) && return ax, ay
    nx = -ty
    ny = tx
    α = -e * (2 * κa + κb) / 6
    β = e * κa / 2
    γ = e * (κb - κa) / 6
    η = p * (α + p * (β + p * γ))
    return ax + p * tx + η * nx, ay + p * ty + η * ny
end

@kernel function _materialize_remesh_outputs_kernel!(out_x, out_y, out_corners,
                                                     out_offsets, out_lengths,
                                                     out_node_contour, x, y,
                                                     wrapx, wrapy, offsets,
                                                     lengths, seg_lengths,
                                                     signed_curvatures,
                                                     densities, measure_start,
                                                     q_measure, remesh_mode,
                                                     corners, μ, Δ_max,
                                                     total_out_nodes)
    g = @index(Global)
    if g <= total_out_nodes
        ci = out_node_contour[g]
        out_local = g - out_offsets[ci] + 1
        off = offsets[ci]
        n = lengths[ci]
        ox = zero(eltype(out_x))
        oy = zero(eltype(out_y))
        corner = UInt8(0)
        written = false

        if remesh_mode[ci] == UInt8(2)
            in_g = off + out_local - 1
            ox = x[in_g]
            oy = y[in_g]
            corner = corners[in_g]
            written = true
        elseif remesh_mode[ci] == UInt8(1)
            remaining = out_local
            first_corner = 0
            prev_corner = 0
            chosen_start = 0
            chosen_intervals = 0
            chosen_segments = 0
            chosen_measure = zero(eltype(out_x))

            @inbounds for li in 1:n
                if !iszero(corners[off + li - 1])
                    first_corner == 0 && (first_corner = li)
                    if prev_corner != 0 && chosen_start == 0
                        span_len = zero(eltype(out_x))
                        span_measure = zero(eltype(out_x))
                        span_segments = 0
                        seg = prev_corner
                        while seg != li
                            gi = off + seg - 1
                            span_len += seg_lengths[gi]
                            span_measure += seg_lengths[gi] * densities[gi]
                            span_segments += 1
                            seg = seg < n ? seg + 1 : 1
                        end
                        n_intervals = _flat_fixed_span_interval_count(
                            span_len, span_measure, span_segments, μ, Δ_max)
                        if remaining > n_intervals
                            remaining -= n_intervals
                        else
                            chosen_start = prev_corner
                            chosen_intervals = n_intervals
                            chosen_segments = span_segments
                            chosen_measure = span_measure
                        end
                    end
                    prev_corner = li
                end
            end

            if chosen_start == 0
                span_len = zero(eltype(out_x))
                span_measure = zero(eltype(out_x))
                span_segments = 0
                seg = prev_corner
                while seg != first_corner
                    gi = off + seg - 1
                    span_len += seg_lengths[gi]
                    span_measure += seg_lengths[gi] * densities[gi]
                    span_segments += 1
                    seg = seg < n ? seg + 1 : 1
                end
                if prev_corner == first_corner
                    @inbounds for li in 1:n
                        gi = off + li - 1
                        span_len += seg_lengths[gi]
                        span_measure += seg_lengths[gi] * densities[gi]
                    end
                    span_segments = n
                end
                chosen_start = prev_corner
                chosen_intervals = _flat_fixed_span_interval_count(
                    span_len, span_measure, span_segments, μ, Δ_max)
                chosen_segments = span_segments
                chosen_measure = span_measure
            end

            if remaining == 1
                in_g = off + chosen_start - 1
                ox = x[in_g]
                oy = y[in_g]
                corner = UInt8(1)
                written = true
            else
                s_target = chosen_measure * (remaining - 1) / chosen_intervals
                span_measure = zero(eltype(out_x))
                seg = chosen_start
                @inbounds for count in 1:chosen_segments
                    in_g = off + seg - 1
                    next_measure = span_measure + seg_lengths[in_g] * densities[in_g]
                    if s_target <= next_measure || count == chosen_segments
                        seg_measure = next_measure - span_measure
                        p = seg_measure <= eps(typeof(seg_measure)) ? zero(seg_measure) :
                            (s_target - span_measure) / seg_measure
                        p = min(max(p, zero(p)), one(p))
                        next_li = seg < n ? seg + 1 : 1
                        next_g = off + next_li - 1
                        ox, oy = _flat_cubic_point_xy(x[in_g], y[in_g],
                                                      x[next_g], y[next_g],
                                                      signed_curvatures[in_g],
                                                      signed_curvatures[next_g], p)
                        corner = UInt8(0)
                        written = true
                        break
                    end
                    span_measure = next_measure
                    seg = seg < n ? seg + 1 : 1
                end
            end
        end

        if !written
            nout = out_lengths[ci]
            s_target = q_measure[ci] * (out_local - 1) / nout
            seg = 1
            @inbounds for li in 1:n
                gi = off + li - 1
                next_measure = measure_start[gi] + seg_lengths[gi] * densities[gi]
                if s_target <= next_measure || li == n
                    seg = li
                    break
                end
            end

            in_g = off + seg - 1
            seg_measure = seg_lengths[in_g] * densities[in_g]
            p = seg_measure <= eps(typeof(seg_measure)) ? zero(seg_measure) :
                (s_target - measure_start[in_g]) / seg_measure
            p = min(max(p, zero(p)), one(p))
            next_g = seg < n ? in_g + 1 : off
            ax = x[in_g]
            ay = y[in_g]
            bx = seg < n ? x[next_g] : x[off] + wrapx[ci]
            by = seg < n ? y[next_g] : y[off] + wrapy[ci]
            ox, oy = _flat_cubic_point_xy(ax, ay, bx, by,
                                          signed_curvatures[in_g],
                                          signed_curvatures[next_g], p)
            corner = UInt8(0)
        end

        out_x[g] = ox
        out_y[g] = oy
        out_corners[g] = corner
    end
end

@kernel function _remesh_output_moments_kernel!(out_area, out_centroid_x,
                                                out_centroid_y, out_x, out_y,
                                                out_offsets, out_lengths,
                                                ncontours)
    ci = @index(Global)
    if ci <= ncontours
        off = out_offsets[ci]
        n = out_lengths[ci]
        area2 = zero(eltype(out_area))
        cx_num = zero(eltype(out_area))
        cy_num = zero(eltype(out_area))
        sx = zero(eltype(out_area))
        sy = zero(eltype(out_area))
        @inbounds for li in 1:n
            g = off + li - 1
            ng = li < n ? g + 1 : off
            cross = out_x[g] * out_y[ng] - out_x[ng] * out_y[g]
            area2 += cross
            cx_num += (out_x[g] + out_x[ng]) * cross
            cy_num += (out_y[g] + out_y[ng]) * cross
            sx += out_x[g]
            sy += out_y[g]
        end

        area = area2 / 2
        out_area[ci] = area
        if abs(area) <= eps(typeof(area))
            out_centroid_x[ci] = sx / n
            out_centroid_y[ci] = sy / n
        else
            inv6A = one(area) / (6 * area)
            out_centroid_x[ci] = cx_num * inv6A
            out_centroid_y[ci] = cy_num * inv6A
        end
    end
end

@kernel function _preserve_remesh_area_kernel!(out_x, out_y, out_node_contour,
                                               out_area, out_centroid_x,
                                               out_centroid_y, target_area,
                                               wrapx, wrapy, remesh_mode,
                                               total_out_nodes)
    g = @index(Global)
    if g <= total_out_nodes
        ci = out_node_contour[g]
        if remesh_mode[ci] == UInt8(0) && iszero(wrapx[ci]) && iszero(wrapy[ci])
            target = target_area[ci]
            current = out_area[ci]
            if abs(target) > eps(typeof(target)) && abs(current) > eps(typeof(current)) &&
               ((target > zero(target)) == (current > zero(current)))
                scale = sqrt(abs(target / current))
                if abs(scale - one(scale)) > sqrt(eps(typeof(scale)))
                    cx = out_centroid_x[ci]
                    cy = out_centroid_y[ci]
                    out_x[g] = cx + scale * (out_x[g] - cx)
                    out_y[g] = cy + scale * (out_y[g] - cy)
                end
            end
        end
    end
end

# Smallest-magnitude real root of a t² + b t + c = 0, device-friendly twin of
# `_smallest_quadratic_root`. Returns `(t, ok)`; `ok=false` means no usable root.
@inline function _device_smallest_quadratic_root(a::T, b::T, c::T) where {T}
    if abs(a) <= eps(T)
        abs(b) <= eps(T) && return (zero(T), false)
        return (-c / b, true)
    end
    disc = b * b - 4 * a * c
    disc < zero(T) && return (zero(T), false)
    sd = sqrt(disc)
    q = -(b + (b >= zero(T) ? sd : -sd)) / 2
    r1 = q / a
    abs(q) <= eps(T) && return (r1, true)
    r2 = c / q
    return (abs(r1) <= abs(r2) ? r1 : r2, true)
end

# Corner-mode area preservation. The corner-free path (`_preserve_remesh_area_kernel!`,
# remesh_mode 0) rescales about the centroid, but that moves fixed corners. For
# the fixed-corner modes (1 and 2) restore the area by displacing only the free
# (non-corner) nodes along d = (p - centroid); the signed area is quadratic in
# the scalar step `t`. This kernel computes that `t` per contour (corners pinned),
# mirroring the CPU `_preserve_closed_area_fixed_corners!`.
@kernel function _remesh_corner_area_step_kernel!(step, target_area, out_area,
                                                  out_centroid_x, out_centroid_y,
                                                  out_x, out_y, out_corners,
                                                  out_offsets, out_lengths,
                                                  wrapx, wrapy, remesh_mode, ncontours)
    ci = @index(Global)
    if ci <= ncontours
        T = eltype(out_x)
        step[ci] = zero(T)
        mode = remesh_mode[ci]
        if (mode == UInt8(1) || mode == UInt8(2)) &&
           iszero(wrapx[ci]) && iszero(wrapy[ci])
            target = target_area[ci]
            A0 = out_area[ci]
            if abs(target) > eps(T) && abs(A0) > eps(T) &&
               ((target > zero(target)) == (A0 > zero(A0)))
                rhs = target - A0
                if abs(rhs) > sqrt(eps(T)) * abs(target)
                    cx = out_centroid_x[ci]
                    cy = out_centroid_y[ci]
                    off = out_offsets[ci]
                    n = out_lengths[ci]
                    B = zero(T)
                    C = zero(T)
                    @inbounds for li in 1:n
                        g = off + li - 1
                        ng = li < n ? g + 1 : off
                        pix = out_x[g]
                        piy = out_y[g]
                        pjx = out_x[ng]
                        pjy = out_y[ng]
                        dix = iszero(out_corners[g]) ? pix - cx : zero(T)
                        diy = iszero(out_corners[g]) ? piy - cy : zero(T)
                        djx = iszero(out_corners[ng]) ? pjx - cx : zero(T)
                        djy = iszero(out_corners[ng]) ? pjy - cy : zero(T)
                        B += (pix * djy - djx * piy) + (dix * pjy - pjx * diy)
                        C += dix * djy - djx * diy
                    end
                    B /= 2
                    C /= 2
                    t, ok = _device_smallest_quadratic_root(C, B, -rhs)
                    if ok && isfinite(t) && abs(t) <= T(1) / 2
                        step[ci] = t
                    end
                end
            end
        end
    end
end

@kernel function _apply_corner_area_step_kernel!(out_x, out_y, out_node_contour,
                                                 out_centroid_x, out_centroid_y,
                                                 out_corners, step, total_out_nodes)
    g = @index(Global)
    if g <= total_out_nodes
        ci = out_node_contour[g]
        t = step[ci]
        if !iszero(t) && iszero(out_corners[g])
            cx = out_centroid_x[ci]
            cy = out_centroid_y[ci]
            out_x[g] = out_x[g] + t * (out_x[g] - cx)
            out_y[g] = out_y[g] + t * (out_y[g] - cy)
        end
    end
end

function _device_remesh_supported(contours::Vector{PVContour{T}}) where {T}
    @inbounds for c in contours
        nnodes(c) < 3 && return false
    end
    return true
end

function _device_remesh_outputs(flat::FlatContourTopology{T},
                                params::SurgeryParams,
                                dev::AbstractDevice=CPU()) where {T}
    ncontours = _flat_ncontours(flat)
    total_nodes = _flat_nnodes(flat)
    if ncontours == 0 || total_nodes == 0
        empty_t = device_zeros(dev, T, 0)
        empty_i = device_zeros(dev, Int, 0)
        empty_b = device_zeros(dev, UInt8, 0)
        return DeviceRewriteOutputs(empty_t, empty_t, empty_t, empty_t,
                                    empty_t, empty_i, empty_i, empty_b)
    end

    seg_lengths = device_zeros(dev, T, total_nodes)
    signed_curvatures = device_zeros(dev, T, total_nodes)
    abs_curvatures = device_zeros(dev, T, total_nodes)
    perimeters = device_zeros(dev, T, ncontours)
    target_area = device_zeros(dev, T, ncontours)
    @_ka_launch dev max(total_nodes, ncontours) _remesh_input_geometry_kernel!(
        seg_lengths, signed_curvatures, abs_curvatures, perimeters,
        target_area, flat.x, flat.y, flat.pv, flat.wrapx, flat.wrapy,
        flat.offsets, flat.lengths, flat.contour_of_node, flat.local_index,
        flat.corners, total_nodes, ncontours)

    node_density_curvatures = device_zeros(dev, T, total_nodes)
    @_ka_launch dev total_nodes _remesh_node_density_kernel!(
        node_density_curvatures, flat.x, flat.y, flat.pv, flat.wrapx,
        flat.wrapy, flat.offsets, flat.lengths, flat.contour_of_node,
        flat.local_index, seg_lengths, abs_curvatures, perimeters,
        T(params.μ), T(params.Δ_max), total_nodes)

    raw_densities = device_zeros(dev, T, total_nodes)
    density_scale = device_zeros(dev, T, ncontours)
    @_ka_launch dev total_nodes _remesh_raw_density_kernel!(
        raw_densities, flat.x, flat.offsets, flat.lengths,
        flat.contour_of_node, flat.local_index, node_density_curvatures,
        T(params.δ), total_nodes)
    @_ka_launch dev ncontours _remesh_density_scale_kernel!(
        density_scale, flat.x, flat.offsets, flat.lengths, seg_lengths,
        raw_densities, perimeters, T(params.μ), T(params.Δ_max), ncontours)

    densities = device_zeros(dev, T, total_nodes)
    measure_start = device_zeros(dev, T, total_nodes)
    q_measure = device_zeros(dev, T, ncontours)
    out_lengths = device_zeros(dev, Int, ncontours)
    remesh_mode = device_zeros(dev, UInt8, ncontours)
    out_pv = device_zeros(dev, T, ncontours)
    out_wrapx = device_zeros(dev, T, ncontours)
    out_wrapy = device_zeros(dev, T, ncontours)
    @_ka_launch dev ncontours _remesh_measure_kernel!(
        densities, measure_start, q_measure, out_lengths, remesh_mode,
        out_pv, out_wrapx, out_wrapy, raw_densities, density_scale,
        flat.corners, seg_lengths, perimeters, flat.pv, flat.wrapx,
        flat.wrapy, flat.offsets, flat.lengths, T(params.μ),
        T(params.Δ_max), ncontours)

    out_offsets = device_zeros(dev, Int, ncontours)
    total_store = device_zeros(dev, Int, 1)
    @_ka_launch dev ncontours _prefix_lengths_kernel!(
        out_offsets, total_store, out_lengths, ncontours)
    total_out_nodes = to_cpu(total_store)[1]

    out_node_contour = device_zeros(dev, Int, total_out_nodes)
    @_ka_launch dev ncontours _out_node_contour_kernel!(
        out_node_contour, out_offsets, out_lengths, ncontours)

    out_x = device_zeros(dev, T, total_out_nodes)
    out_y = device_zeros(dev, T, total_out_nodes)
    out_corners = device_zeros(dev, UInt8, total_out_nodes)
    @_ka_launch dev total_out_nodes _materialize_remesh_outputs_kernel!(
        out_x, out_y, out_corners, out_offsets, out_lengths,
        out_node_contour, flat.x, flat.y, flat.wrapx, flat.wrapy,
        flat.offsets, flat.lengths, seg_lengths, signed_curvatures,
        densities, measure_start, q_measure, remesh_mode, flat.corners,
        T(params.μ), T(params.Δ_max), total_out_nodes)

    out_area = device_zeros(dev, T, ncontours)
    out_centroid_x = device_zeros(dev, T, ncontours)
    out_centroid_y = device_zeros(dev, T, ncontours)
    @_ka_launch dev ncontours _remesh_output_moments_kernel!(
        out_area, out_centroid_x, out_centroid_y, out_x, out_y,
        out_offsets, out_lengths, ncontours)
    @_ka_launch dev total_out_nodes _preserve_remesh_area_kernel!(
        out_x, out_y, out_node_contour, out_area, out_centroid_x,
        out_centroid_y, target_area, out_wrapx, out_wrapy, remesh_mode,
        total_out_nodes)

    # Fixed-corner modes (1, 2) preserve area by moving only free nodes.
    corner_area_step = device_zeros(dev, T, ncontours)
    @_ka_launch dev ncontours _remesh_corner_area_step_kernel!(
        corner_area_step, target_area, out_area, out_centroid_x, out_centroid_y,
        out_x, out_y, out_corners, out_offsets, out_lengths,
        out_wrapx, out_wrapy, remesh_mode, ncontours)
    @_ka_launch dev total_out_nodes _apply_corner_area_step_kernel!(
        out_x, out_y, out_node_contour, out_centroid_x, out_centroid_y,
        out_corners, corner_area_step, total_out_nodes)

    return DeviceRewriteOutputs(out_x, out_y, out_pv, out_wrapx, out_wrapy,
                                out_offsets, out_lengths, out_corners)
end

function _device_remesh_outputs(contours::Vector{PVContour{T}},
                                params::SurgeryParams,
                                dev::AbstractDevice=CPU()) where {T}
    _device_remesh_supported(contours) || return nothing
    return _device_remesh_outputs(_pack_flat_topology(contours, dev), params, dev)
end

function _device_remesh_outputs(state::DeviceContourState{T},
                                params::SurgeryParams,
                                dev::AbstractDevice=CPU()) where {T}
    return _device_remesh_outputs(_flat_topology(state, dev), params, dev)
end

function _device_remesh_contours(contours::Vector{PVContour{T}},
                                 params::SurgeryParams,
                                 dev::AbstractDevice=CPU()) where {T}
    outputs = _device_remesh_outputs(contours, params, dev)
    outputs === nothing && return nothing
    return _unpack_rewrite_outputs(outputs)
end

