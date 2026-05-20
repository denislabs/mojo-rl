"""box_muller_normal — fill a buffer with i.i.d. N(0, 1) samples.

Box-Muller transform: given two independent U(0, 1) samples u1, u2,
    z = sqrt(-2 ln u1) · cos(2π u2)
is N(0, 1). We clamp u1 ≥ 1e-10 to avoid log(0).

The PPO and SAC examples both used a copy of this in-file; Phase 8.1
extracts it to nn2/random/. Phase 7 used `std.random.random_float64`
as the entropy source — same here. RNG seeding is the caller's
responsibility (`from std.random import seed`).
"""

from std.math import cos as fcos, log as flog, sqrt as fsqrt, pi
from std.random import random_float64

from ..constants import DT


def box_muller_normal(out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int):
    """Fill out_ptr[0:n] with iid N(0, 1) samples via Box-Muller.

    Each lane is drawn independently — no batched-pair optimization (the
    classic Box-Muller produces *two* normals per cos/sin pair; we only
    keep the cos branch for simplicity since correlations don't matter
    here). If you need the second normal, call again — RNG advances.
    """
    for i in range(n):
        var u1 = random_float64()
        if u1 < 1e-10:
            u1 = 1e-10
        var u2 = random_float64()
        out_ptr[i] = fsqrt(Scalar[DT](-2.0) * flog(Scalar[DT](u1))) * fcos(
            Scalar[DT](2.0 * pi) * Scalar[DT](u2)
        )
