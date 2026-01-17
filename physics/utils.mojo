fn abs[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Absolute value."""
    return x if x >= 0.0 else -x


fn clamp[
    dtype: DType
](x: Scalar[dtype], low: Scalar[dtype], high: Scalar[dtype]) -> Scalar[dtype]:
    """Clamp value to range."""
    if x < low:
        return low
    if x > high:
        return high
    return x
