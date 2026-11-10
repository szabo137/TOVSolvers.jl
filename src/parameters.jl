# TODO:
# - implement Base.show
# - consider Base.@kw_only
struct TOVParameters{T} <: AbstractTOVParameters
  omega::T
  x::T

  function TOVParameters(omega::T, x::T) where {T<:Real}
    isfinite(omega) || throw(
                             ArgumentError(
                                           "omega must be finite, got $omega"
                                          )
                            )
    omega > 0 || throw(
                       ArgumentError(
                                     "omega must be positive, got $omega"
                                    )
                      )
    isfinite(x) ||  throw(
                          ArgumentError(
                                        "x must be finite, got $x"
                                       )
                         )
    x > 0 ||  throw(
                    ArgumentError(
                                  "x must be positive, got $x"
                                 )
                   )

    new{T}(omega,x)
  end
end

TOVParameters(omega,x) = TOVParameters(promote(omega,x)...)
parameter_tuple(p::TOVParameters) = (p.omega, p.x)
