# Reusable computational buffers have an explicit owner, independent of geometry.
mutable struct _VelocityScratch{T<:AbstractFloat}
    contour_curvatures::Vector{Vector{T}}
    reference_curvatures::Vector{Vector{T}}
    layer_curvatures::Vector{Vector{Vector{T}}}
    offsets::Vector{Int}
    target_nodes::Vector{SVector{2,T}}
    mode_vel::Vector{SVector{2,T}}
    to_physical::Matrix{T}
    to_modal::Matrix{T}
    energy_partial::Vector{T}
    modal_kernel::Any
end

function _VelocityScratch{T}() where {T<:AbstractFloat}
    return _VelocityScratch{T}(Vector{T}[],
                               Vector{T}[],
                               Vector{Vector{T}}[],
                               Int[],
                               SVector{2,T}[],
                               SVector{2,T}[],
                               Matrix{T}(undef, 0, 0),
                               Matrix{T}(undef, 0, 0),
                               T[], nothing)
end


mutable struct _SurgeryWorkspace{T}
    nodes::Vector{SVector{2,T}}
    arcs::Vector{T}
    virtual_nodes::Vector{SVector{2,T}}
end
_SurgeryWorkspace{T}() where {T} = _SurgeryWorkspace(SVector{2,T}[], T[], SVector{2,T}[])

"""
    ExecutionWorkspace(T=Float64)

Reusable velocity, energy, and surgery scratch for a problem. Constructors
allocate one by default; pass `workspace=ws` to explicitly control its lifetime.
Use a separate workspace for each concurrently evaluated problem. Workspaces
contain no physical state and may be cleared between calls. RK4 stage buffers
remain owned by their stepper, whose lifetime defines an integration session.
"""
mutable struct ExecutionWorkspace{T<:AbstractFloat}
    cpu::_VelocityScratch{T}
    surgery::_SurgeryWorkspace{T}
    buffers::Dict{Tuple{Symbol,DataType,DataType},Any}
end
ExecutionWorkspace(::Type{T}=Float64) where {T<:AbstractFloat} =
    ExecutionWorkspace{T}(_VelocityScratch{T}(), _SurgeryWorkspace{T}(),
                          Dict{Tuple{Symbol,DataType,DataType},Any}())

# Compatibility default for standalone internal state evaluators. Public problem
# paths always pass their own workspace explicitly, so unrelated problems do
# not evict each other's device buffers when their node counts differ.
const _EXECUTION_WS_TLS_KEY = :contourdynamics_execution_workspace
function _default_execution_workspace(::Type{T}) where {T}
    get!(() -> ExecutionWorkspace(T), task_local_storage(), (_EXECUTION_WS_TLS_KEY, T))
end

"""
    clear_state_workspace_cache!(workspace::ExecutionWorkspace)
    clear_state_workspace_cache!()

Release cached computational buffers owned by `workspace`. With no argument,
clear only the calling task's compatibility workspaces for standalone state
operations. Problem-owned workspaces are unaffected by the no-argument form.
"""
function clear_state_workspace_cache!(ws::ExecutionWorkspace{T}) where {T}
    empty!(ws.buffers)
    ws.cpu = _VelocityScratch{T}()
    ws.surgery = _SurgeryWorkspace{T}()
    return nothing
end
function clear_state_workspace_cache!()
    store = task_local_storage()
    for key in collect(keys(store))
        key isa Tuple && length(key) == 2 && key[1] === _EXECUTION_WS_TLS_KEY && delete!(store, key)
    end
    return nothing
end
