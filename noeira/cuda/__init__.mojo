"""Device-graph record/replay for Mojo GPU kernels.

Two mechanisms, one being retired:

`device_graph` — MAX's own `DeviceGraph`. No interceptor, no `LD_PRELOAD`, no
borrowed stream. **Prefer it.** Its `STEP` takes a `DeviceContext` argument
instead of capturing one, and its fallback is a runtime latch rather than a
comptime no-op (it links everywhere and raises off CUDA/HIP).

`graph` — the legacy `LD_PRELOAD` + `cuStreamBeginCapture` shim. Compile-time
no-op off NVIDIA. Being migrated away from call site by call site; see
`device_graph.mojo`'s header for what it costs and what replaces it.
"""

from .graph import CUDAGraph, maybe_capture_replay
from .device_graph import GraphSlot, maybe_record_replay
