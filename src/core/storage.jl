# The storage object is the single authority for geometry and topology.
abstract type _AbstractContourStorage end
struct _HostContourStorage{C} <: _AbstractContourStorage
    data::C
end
struct _DeviceContourStorage{S} <: _AbstractContourStorage
    data::S
end
_storage(data, ::CPU) = _HostContourStorage(data)
_storage(data::Vector{PVContour{T}}, dev::GPU) where {T} =
    _DeviceContourStorage(DeviceContourState(data, dev))
_storage(data::Tuple, dev::GPU) =
    _DeviceContourStorage(map(layer -> DeviceContourState(layer, dev), data))

_storage_data(storage::_AbstractContourStorage) = storage.data
_borrow_contours(storage::_HostContourStorage) = storage.data
_borrow_contours(::_DeviceContourStorage) = error(
    "Device contours cannot be borrowed on the host; use snapshot_contours(prob).")
_materialize_storage(storage::_HostContourStorage) = storage.data
_materialize_storage(storage::_DeviceContourStorage{<:DeviceContourState}) = materialize_contours(storage.data)
_materialize_storage(storage::_DeviceContourStorage{<:Tuple}) = map(materialize_contours, storage.data)
_snapshot_storage(storage::_HostContourStorage) = deepcopy(storage.data)
_snapshot_storage(storage::_DeviceContourStorage) = _materialize_storage(storage)
_device_storage(storage::_DeviceContourStorage) = storage.data
_device_storage(::_HostContourStorage) = nothing
