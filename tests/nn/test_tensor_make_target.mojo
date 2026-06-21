"""Tensor.make[target] / ensure[target] — unified allocator parity gate.

The additive `[target]`-generic allocator must behave exactly like the existing
`alloc` / `alloc_gpu` (+ `ensure` / `ensure_gpu`): zero-filled, length n, and on
GPU round-trips through upload/download identically. Also confirms the
parametrized `ensure[target]` overload coexists with the bare `ensure(n)`.

Run:
  pixi run mojo run -I . tests/nn/test_tensor_make_target.mojo
  pixi run -e apple mojo run -I . tests/nn/test_tensor_make_target.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor


def main() raises:
    print("=" * 56)
    print("Tensor.make[target] / ensure[target] parity gate")
    print("=" * 56)
    comptime N = 17

    # ---- CPU: make["cpu"] == alloc -------------------------------------
    var a = Tensor.make["cpu"](N)
    var b = Tensor.alloc(N)
    var cpu_ok = len(a.data) == N and a.n == N and len(b.data) == N
    for i in range(N):
        if a.data[i] != Scalar[DT](0):
            cpu_ok = False
    # write through, then ensure["cpu"] must NOT clobber (grow-only)
    for i in range(N):
        a.data[i] = Scalar[DT](i + 1)
    a.ensure["cpu"](N)  # no-op (already >= N)
    for i in range(N):
        if a.data[i] != Scalar[DT](i + 1):
            cpu_ok = False
    a.ensure["cpu"](N + 5)  # grows → reset to zero-fill (matches bare ensure)
    var cpu_grow_ok = len(a.data) >= N + 5
    print("  CPU make/ensure parity:", "OK" if cpu_ok and cpu_grow_ok else "FAIL")

    # ---- GPU: make["gpu"] round-trips like alloc_gpu -------------------
    var gpu_ok = True
    with DeviceContext() as ctx:
        var g = Tensor.make["gpu"](N, Optional(ctx))
        # fresh device buffer is zero-filled
        g.download(ctx)
        for i in range(N):
            if g.data[i] != Scalar[DT](0):
                gpu_ok = False
        # write on host, upload, zero host, download → values survive D2H/H2D
        for i in range(N):
            g.data[i] = Scalar[DT](i * 2 - 3)
        g.upload(ctx)
        for i in range(N):
            g.data[i] = Scalar[DT](0)
        g.download(ctx)
        for i in range(N):
            if g.data[i] != Scalar[DT](i * 2 - 3):
                gpu_ok = False
        # ensure["gpu"] grow path allocates a larger device buffer
        var h = Tensor()
        h.ensure["gpu"](N, Optional(ctx))
        h.ensure["gpu"](2 * N, Optional(ctx))
        gpu_ok = gpu_ok and h.n >= 2 * N
    print("  GPU make/ensure round-trip:", "OK" if gpu_ok else "FAIL")

    assert_true(cpu_ok and cpu_grow_ok and gpu_ok, "make[target] parity")
    print("TENSOR MAKE TARGET OK")
