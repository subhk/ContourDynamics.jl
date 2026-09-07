# Pure cubic interpolation shared by host geometry and device kernels.

@inline function _cubic_point_tangent_scalar(ax::T, ay::T, bx::T, by::T,
                                             κa::T, κb::T, p::T) where {T}
    # Return both point and tangent on the cubic segment in scalar form, avoiding
    # SVector allocation inside GPU kernels.
    dsx = bx - ax
    dsy = by - ay
    ds_len = sqrt(dsx^2 + dsy^2)
    nx = -dsy
    ny = dsx
    α = -ds_len * (T(2) * κa + κb) / T(6)
    β = ds_len * κa / T(2)
    γ = ds_len * (κb - κa) / T(6)
    η = p * (α + p * (β + p * γ))
    η′ = α + T(2) * β * p + T(3) * γ * p^2
    return ax + p * dsx + η * nx,
           ay + p * dsy + η * ny,
           dsx + η′ * nx,
           dsy + η′ * ny
end
