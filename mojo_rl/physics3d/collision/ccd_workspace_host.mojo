"""Host-side allocation of a CCD workspace row.

Split from `ccd_workspace` so that module stays a leaf: it is imported by
`gjk.mojo` and therefore by every collision kernel, and pulling `TensorImpl`
in there would drag the tensor package into the Metal compile for the sake of
a helper no kernel can call.

The engine never uses this — `Data.ccd_ws` is the engine's workspace, one row
per env. This is for the gates and probes in `tests/physics3d`, which collide
a single pair at a time and bind `L_CCD_WS1` with `wrow = 0`.
"""

from mojo_rl.nn.core.tensor import TensorImpl

from .ccd_workspace import CCD_WS_SIZE


def ccd_ws_alloc[DTYPE: DType]() raises -> TensorImpl[DTYPE]:
    """One CCD workspace row. Uninitialised on purpose: EPA writes every slot
    it reads within a single call, so a caller never has to clear it."""
    return TensorImpl[DTYPE].alloc(CCD_WS_SIZE)
