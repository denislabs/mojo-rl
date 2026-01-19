# ===== Particle for engine flames =====


@register_passable("trivial")
struct Particle[dtype: DType]:
    """Simple particle for engine flame effects."""

    var x: Scalar[Self.dtype]
    var y: Scalar[Self.dtype]
    var vx: Scalar[Self.dtype]
    var vy: Scalar[Self.dtype]
    var ttl: Scalar[Self.dtype]  # Time to live in seconds

    fn __init__(
        out self,
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
        vx: Scalar[Self.dtype],
        vy: Scalar[Self.dtype],
        ttl: Scalar[Self.dtype],
    ):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ttl = ttl
