"""Concurrency backbone — one OS thread, a bounded byte ring, atomic cells.

See `docs/CONCURRENCY_BACKBONE.md` for why this exists and what it is for.
Mojo has no `async`/`await` and `runtime.asyncrt` is not in the standalone
package; none of that is needed, and this package is what replaces it.

⚠ START WITH `SharedRing` AND `SharedBlock`, not `SpscRing` / `ControlBlock`.
The bare owners are freed at their LAST MENTION, which is usually the `view()`
call that built the worker — leaving a live thread reading freed memory. The
refcounted wrappers are references the compiler tracks, so the hazard goes
away instead of becoming a rule to remember.
"""

from .thread import (
    OpaquePtr,
    ThreadHandle,
    null_opaque,
    opaque_from_address,
    sleep_us,
)
from .block import (
    CELLS_PER_LINE,
    ControlBlock,
    ControlBlockView,
    SharedBlock,
)
from .ring import (
    PopClaim,
    PushClaim,
    SharedRing,
    SpscRing,
    SpscRingView,
    cells_for,
)
from .worker import (
    POLL_DID_WORK,
    POLL_DONE,
    POLL_IDLE,
    BackgroundThread,
    BackgroundWorker,
    WorkerCtl,
)
