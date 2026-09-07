# surgery/driver.jl — CPU surgery stage.

# Remesh every contour in `contours` (mutating the vector in place) using a
# single shared `density_sources` snapshot, then demote obtuse corners. Taking
# `contours` and the scratch buffers as explicit arguments — rather than
# capturing them in a closure that reassigns `density_sources` — keeps this
# allocation-free and type-stable (no `Core.Box`).
function _remesh_all!(contours::Vector{PVContour{T}}, params::SurgeryParams,
                      remesh_buf::Vector{SVector{2,T}}, arc_buf::Vector{T},
                      vnodes_buf::Vector{SVector{2,T}}) where {T}
    density_sources = copy(contours)
    density_source_data = _prepare_density_sources(density_sources)

    for i in eachindex(contours)
        contours[i] = remesh(contours[i], params;
                             _buf=remesh_buf, _arc_buf=arc_buf,
                             _vnodes_buf=vnodes_buf, _density_sources=density_sources,
                             _density_source_data=density_source_data)
    end

    _demote_obtuse_corners!(contours)
    return contours
end

# One full Dritschel surgery pass over a single vector of contours: pre-clean →
# corner labelling → remesh → reconnect close compatible parts until exhausted
# (with stall detection) → post-remesh → filament removal → spanning-proximity
# check. Shared by single-layer `surgery!` and each layer of the multi-layer
# `surgery!`; `layer_label` (e.g. " layer 2") is interpolated into warnings.
# Shared stall-detection reconnection loop (Dritschel surgery step 2). The CPU
# and device surgery passes differ only in how they find close pairs, apply a
# reconnection, and clean up afterwards; the termination policy — stop after 3
# consecutive close-pair-count increases or 6 iterations without a new minimum,
# retry once after a cleanup pass, and warn on large stalls — lives here so
# every path terminates under the same rules.
#
# `find_pairs()` returns the current close-pair collection, `npairs(pairs)`
# its size, `reconnect_step!(pairs)` applies one reconnection round (returning
# `false` to abort the loop), and `stall_cleanup!()` runs the cleanup used for
# the one retry before declaring a stall.
function _reconnect_until_exhausted!(find_pairs::F, npairs::N, reconnect_step!::R,
                                     stall_cleanup!::C,
                                     warn_label::AbstractString) where {F,N,R,C}
    reconnected = false
    max_reconnect_iter = 100
    stall_warning_pairs = 100

    prev_n_pairs = typemax(Int)
    min_n_pairs = typemax(Int)
    stall_count = 0
    no_improve_count = 0

    for iter in 1:max_reconnect_iter
        pairs = find_pairs()
        n_pairs = npairs(pairs)
        n_pairs == 0 && break
        if n_pairs > prev_n_pairs
            stall_count += 1
        else
            stall_count = 0
        end
        if n_pairs < min_n_pairs
            min_n_pairs = n_pairs
            no_improve_count = 0
        else
            no_improve_count += 1
        end
        if stall_count >= 3 || no_improve_count >= 6
            # Reconnection creates near-duplicate stitch nodes by construction.
            # Before warning about a stall, run the cleanup pass that follows a
            # successful reconnect and re-check proximity.
            if reconnected
                stall_cleanup!()
                pairs = find_pairs()
                remeshed_n_pairs = npairs(pairs)
                remeshed_n_pairs == 0 && break
                if remeshed_n_pairs < n_pairs
                    prev_n_pairs = remeshed_n_pairs
                    min_n_pairs = min(min_n_pairs, remeshed_n_pairs)
                    stall_count = 0
                    no_improve_count = 0
                    continue
                end
                n_pairs = remeshed_n_pairs
            end
            if n_pairs >= stall_warning_pairs
                @warn "surgery!:$(warn_label) reconnection stalled ($n_pairs close pairs, min seen: $min_n_pairs) — stopping early"
            end
            break
        end
        prev_n_pairs = n_pairs
        reconnect_step!(pairs) || break
        reconnected = true
        if iter == max_reconnect_iter
            @warn "surgery!:$(warn_label) reconnection iteration limit ($max_reconnect_iter) reached with $n_pairs close pairs remaining"
        end
    end
    return reconnected
end

function _surgery_pass!(contours::Vector{PVContour{T}}, domain::AbstractDomain,
                        params::SurgeryParams, remesh_buf::Vector{SVector{2,T}},
                        arc_buf::Vector{T}, vnodes_buf::Vector{SVector{2,T}};
                        layer_label::AbstractString="") where {T}

    remove_filaments!(contours, params.area_min, params.μ)
    _demote_obtuse_corners!(contours)
    _promote_high_curvature_corners!(contours, params.δ)

    # 1. Remesh all contours (Dritschel node redistribution).
    _remesh_all!(contours, params, remesh_buf, arc_buf, vnodes_buf)

    reconnection_bin_size = max(T(params.δ), T(params.μ))

    # 2. Reconnection — iterate until no more close pairs remain, under the
    #    shared stall policy in `_reconnect_until_exhausted!` (catches both
    #    monotone growth and oscillation, e.g. 10→12→10→12).
    reconnected = _reconnect_until_exhausted!(
        () -> begin
            idx = build_spatial_index(contours, reconnection_bin_size, domain)
            find_close_segments(contours, idx, params.δ, domain)
        end,
        length,
        close_pairs -> begin
            reconnect!(contours, close_pairs, domain)
            remove_filaments!(contours, params.area_min, params.μ)
            _remesh_all!(contours, params, remesh_buf, arc_buf, vnodes_buf)
            remove_filaments!(contours, params.area_min, params.μ)
            true
        end,
        () -> begin
            _remesh_all!(contours, params, remesh_buf, arc_buf, vnodes_buf)
            remove_filaments!(contours, params.area_min, params.μ)
        end,
        layer_label)

    # 3. Re-remesh after reconnection to clean up short/long stitch segments.
    if reconnected
        _remesh_all!(contours, params, remesh_buf, arc_buf, vnodes_buf)
    end

    # 4. Remove filaments.
    remove_filaments!(contours, params.area_min, params.μ)

    # 5. Warn if closed contours are too close to spanning contours.
    _check_spanning_proximity(contours, params.δ, domain)
    return contours
end

"""
    surgery!(prob::ContourProblem, params::SurgeryParams)

Dritschel-style contour-surgery pass:
pre-clean unresolved debris → update corner labels → remesh → reconnect close
compatible contour parts until exhausted → post-remesh → remove unresolved
filaments.

This implements the standard topological surgery loop from Dritschel's contour
surgery algorithm (Dritschel 1988, Table III): identify admissible close contour
parts enclosing the same interior vorticity, split or merge them, repeat until
exhausted, and redistribute nodes afterward. Reconnection creates labelled
corner nodes that remain fixed during remeshing until they become obtuse.
Remeshing uses the Dritschel nonlocal node-density rule with cubic interpolation
arcs; the velocity paths use the same signed-curvature geometry when evaluating
curved segments.
Mutates `_host_contours(prob)` in place.
"""
function surgery!(prob::ContourProblem{<:AbstractKernel, <:AbstractDomain, T}, params::SurgeryParams) where {T}
    # Pre-allocate workspace buffers once to avoid per-contour allocations in
    # remesh, then run the shared Dritschel surgery pass over `_host_contours(prob)`.
    scratch = execution_workspace(prob).surgery
    remesh_buf, arc_buf, vnodes_buf = scratch.nodes, scratch.arcs, scratch.virtual_nodes
    _surgery_pass!(_host_contours(prob), prob.domain, params, remesh_buf, arc_buf, vnodes_buf)
    return prob
end
