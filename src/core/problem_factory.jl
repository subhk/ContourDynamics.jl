# Builder helpers for the keyword-based `Problem(...)` constructor. Keeping
# them here separates the user-facing wrapper type from argument parsing.

const _SURGERY_PRESETS = Dict{Symbol, NamedTuple}(
    :standard     => (δ=0.005, μ=0.02,  Δ_max=0.2,  area_min=1e-6,  n_surgery=5),
    :conservative => (δ=0.002, μ=0.01,  Δ_max=0.15, area_min=1e-8,  n_surgery=10),
    :aggressive   => (δ=0.01,  μ=0.04,  Δ_max=0.3,  area_min=1e-4,  n_surgery=3),
)

@inline _convert_contour_precision(::Type{T}, c::PVContour{T}) where {T<:AbstractFloat} = c

function _convert_contour_precision(::Type{T}, c::PVContour) where {T<:AbstractFloat}
    nodes = Vector{SVector{2,T}}(undef, nnodes(c))
    @inbounds for i in eachindex(nodes)
        node = c.nodes[i]
        nodes[i] = SVector{2,T}(T(node[1]), T(node[2]))
    end
    wrap = SVector{2,T}(T(c.wrap[1]), T(c.wrap[2]))
    return PVContour(nodes, T(c.pv), wrap, copy(c.corners))
end

@inline _convert_contours_precision(::Type{T},
                                    contours::Vector{PVContour{T}}) where {T<:AbstractFloat} =
    contours

function _convert_contours_precision(::Type{T},
                                     contours::AbstractVector{<:PVContour}) where {T<:AbstractFloat}
    converted = Vector{PVContour{T}}(undef, length(contours))
    @inbounds for i in eachindex(contours)
        converted[i] = _convert_contour_precision(T, contours[i])
    end
    return converted
end

function _convert_layers_precision(::Type{T}, layers::Tuple) where {T<:AbstractFloat}
    return map(layer -> _convert_contours_precision(T, layer), layers)
end

function _validate_problem_layout(kernel::Symbol, contours, layers)
    # Single-layer problems own a flat Vector{PVContour}; multi-layer problems
    # own a tuple of per-layer contour vectors. Reject mixed inputs early so the
    # lower-level constructors do not fail with less helpful type errors.
    is_multilayer = kernel === :multilayer_qg
    if is_multilayer
        layers !== nothing || throw(ArgumentError(
            "kernel=:multilayer_qg requires `layers` (tuple of contour vectors), not `contours`."))
        contours === nothing || throw(ArgumentError(
            "`contours` and `layers` are mutually exclusive. Use `layers` for multi-layer problems."))
    else
        contours !== nothing || throw(ArgumentError(
            "Required keyword `contours` not provided. Pass a Vector{PVContour}."))
        layers === nothing || throw(ArgumentError(
            "`layers` requires `kernel=:multilayer_qg`."))
    end
    return is_multilayer
end

Base.@constprop :aggressive function _build_kernel(::Type{T}, kernel::Symbol, Ld, coupling,
                                                   layer_thicknesses,
                                                   δ_sqg::Real, beta::Real) where {T}
    # Normalize all numeric inputs to the problem float type here. Downstream
    # kernel constructors deliberately enforce type consistency with contours.
    layer_thicknesses !== nothing && kernel !== :multilayer_qg && throw(ArgumentError(
        "`layer_thicknesses` is only valid with kernel=:multilayer_qg."))
    if kernel === :euler
        return EulerKernel()
    elseif kernel === :qg
        Ld === nothing && throw(ArgumentError("kernel=:qg requires `Ld` (deformation radius)."))
        return QGKernel(T(Ld))
    elseif kernel === :beta_plane_qg
        Ld === nothing && throw(ArgumentError("kernel=:beta_plane_qg requires `Ld` (deformation radius)."))
        isnan(beta) && throw(ArgumentError("kernel=:beta_plane_qg requires `beta`."))
        return BetaPlaneQGKernel(T(beta), T(Ld), PVContour{T}[])
    elseif kernel === :sqg
        isnan(δ_sqg) && throw(ArgumentError("kernel=:sqg requires `δ_sqg` (regularization length)."))
        return SQGKernel(T(δ_sqg))
    elseif kernel === :multilayer_qg
        Ld === nothing && throw(ArgumentError("kernel=:multilayer_qg requires `Ld` (deformation radii)."))
        coupling === nothing && throw(ArgumentError("kernel=:multilayer_qg requires `coupling` (layer coupling matrix)."))
        M = length(Ld)
        size(coupling, 1) == size(coupling, 2) || throw(ArgumentError("`coupling` must be square; got size $(size(coupling))."))
        Nlay = size(coupling, 1)
        # StaticArrays keep small layer-coupling matrices stack-allocated and
        # make modal decomposition type-stable in the velocity path.
        Ld_s = Ld isa SVector ? convert(SVector{M, T}, Ld) : SVector{M, T}(T.(collect(Ld)))
        coupling_s = coupling isa SMatrix ? convert(SMatrix{Nlay, Nlay, T}, coupling) : SMatrix{Nlay, Nlay, T}(T.(coupling))
        if layer_thicknesses === nothing
            return MultiLayerQGKernel(Ld_s, coupling_s)
        end
        length(layer_thicknesses) == Nlay || throw(ArgumentError(
            "Expected $Nlay layer thicknesses, got $(length(layer_thicknesses))"))
        H_s = layer_thicknesses isa SVector ?
              convert(SVector{Nlay,T}, layer_thicknesses) :
              SVector{Nlay,T}(T.(collect(layer_thicknesses)))
        return MultiLayerQGKernel(Ld_s, coupling_s, H_s)
    end
    throw(ArgumentError("Unknown kernel :$kernel. Use :euler, :qg, :beta_plane_qg, :sqg, or :multilayer_qg."))
end

Base.@constprop :aggressive function _build_domain(::Type{T}, domain::Symbol,
                                                   Lx::Real, Ly::Real) where {T}
    if domain === :unbounded
        return UnboundedDomain()
    elseif domain === :periodic
        (isnan(Lx) || isnan(Ly)) && throw(ArgumentError(
            "domain=:periodic requires `Lx` and `Ly` (domain half-widths)."))
        return PeriodicDomain(T(Lx), T(Ly))
    end
    throw(ArgumentError("Unknown domain :$domain. Use :unbounded or :periodic."))
end

function _build_device(dev)
    dev isa AbstractDevice || throw(ArgumentError(
        "`dev` must be `CPU()` or `GPU()`, got $(repr(dev))."))
    return dev
end

Base.@constprop :aggressive function _build_contour_problem(is_multilayer::Bool, kernel,
                                                            domain, contours, layers, dev)
    if is_multilayer
        return MultiLayerContourProblem(kernel, domain, layers; dev=dev)
    elseif kernel isa BetaPlaneQGKernel
        domain isa PeriodicDomain || throw(ArgumentError(
            "kernel=:beta_plane_qg currently requires domain=:periodic."))
        return ContourProblem(_attach_beta_plane_reference(kernel, contours),
                              domain, contours; dev=dev)
    else
        return ContourProblem(kernel, domain, contours; dev=dev)
    end
end

Base.@constprop :aggressive function _build_stepper(::Type{T}, stepper::Symbol,
                                                    dt::Real, prob, dev) where {T}
    # Stepper buffers are sized from the current node count. Surgery may change
    # this later, in which case `resize_buffers!` is called by `evolve!`.
    N = total_nodes(prob)
    if stepper === :RK4
        return RK4Stepper(T(dt), N; dev=dev)
    end
    throw(ArgumentError("Unknown stepper :$stepper. Only :RK4 is supported."))
end

Base.@constprop :aggressive function _build_surgery(surgery, ::Type{T}) where {T}
    # Accept either an explicit SurgeryParams or a named preset. Presets are
    # intentionally conservative defaults for examples, not universal settings.
    surgery isa SurgeryParams && return surgery
    surgery === :none && return nothing
    if surgery isa Symbol
        haskey(_SURGERY_PRESETS, surgery) || throw(ArgumentError(
            "Unknown surgery preset :$surgery. Use :standard, :conservative, :aggressive, :none, or a SurgeryParams."))
        preset = _SURGERY_PRESETS[surgery]
        return SurgeryParams(T(preset.δ), T(preset.μ), T(preset.Δ_max), T(preset.area_min), preset.n_surgery)
    end
    throw(ArgumentError("surgery must be a Symbol preset or SurgeryParams, got $(typeof(surgery))"))
end

"""
    Problem(; contours, dt, kernel=:euler, domain=:unbounded, stepper=:RK4, surgery=:standard, dev=CPU(), T=Float64, ...)

Convenience constructor that builds a [`ContourProblem`](@ref),
time stepper, and [`SurgeryParams`](@ref) from keyword arguments.

Use `contours=Vector{PVContour}` for single-layer Euler/QG/SQG problems, or
`layers=(layer1, layer2, ...)` with `kernel=:multilayer_qg` for multi-layer QG.
Periodic domains require `domain=:periodic, Lx=..., Ly=...`. QG kernels require
`Ld`; beta-plane QG also requires `beta` and spanning contours from
[`beta_staircase`](@ref); SQG kernels require `δ_sqg`; multi-layer kernels
require both `Ld` and `coupling`. For unequal-depth multi-layer QG, pass
`layer_thicknesses`; a connected physical coupling matrix also allows their
relative values to be inferred.

`surgery` may be a [`SurgeryParams`](@ref), one of `:standard`,
`:conservative`, `:aggressive`, or `:none`.
"""
Base.@constprop :aggressive function Problem(;
    contours=nothing,
    dt::Real,
    layers=nothing,
    coupling=nothing,
    layer_thicknesses=nothing,
    kernel::Symbol=:euler,
    Ld=nothing,
    beta::Real=NaN,
    δ_sqg::Real=NaN,
    delta_sqg::Real=NaN,
    domain::Symbol=:unbounded,
    Lx::Real=NaN,
    Ly::Real=NaN,
    stepper::Symbol=:RK4,
    surgery=:standard,
    dev=CPU(),
    T::Type{<:AbstractFloat}=Float64,
)
    # Build in dependency order: layout determines problem type, then kernel and
    # domain are normalized, then the concrete problem sizes the stepper buffers.
    is_multilayer = _validate_problem_layout(kernel, contours, layers)
    typed_contours = is_multilayer ? nothing : _convert_contours_precision(T, contours)
    typed_layers = is_multilayer ? _convert_layers_precision(T, layers) : nothing
    isnan(δ_sqg) || isnan(delta_sqg) || throw(ArgumentError(
        "Specify only one of `δ_sqg` and the legacy `delta_sqg`."))
    sqg_regularization = isnan(δ_sqg) ? delta_sqg : δ_sqg
    built_kernel = _build_kernel(T, kernel, Ld, coupling, layer_thicknesses,
                                 sqg_regularization, beta)
    built_domain = _build_domain(T, domain, Lx, Ly)
    device = _build_device(dev)
    contour_problem = _build_contour_problem(is_multilayer, built_kernel, built_domain,
                                             typed_contours, typed_layers, device)
    time_stepper = _build_stepper(T, stepper, dt, contour_problem, device)
    surgery_params = _build_surgery(surgery, T)
    return Problem(contour_problem, time_stepper, surgery_params)
end
