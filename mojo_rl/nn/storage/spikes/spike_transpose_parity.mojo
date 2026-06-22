"""Transpose2D parity: forward + vjp vs an INDEPENDENT analytic permutation
reference. CPU + GPU. Validates the shared-mem tiled GPU rewrite (B1).

Pure data movement (no arithmetic) → bit-exact on both paths, no TF32 caveat.

Run: pixi run -e apple mojo run -I . mojo_rl/nn/storage/spikes/spike_transpose_parity.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.primitives.transpose_2d import Transpose2D
from mojo_rl.nn.storage.core.initializer import Deterministic


def _run[
    target: StaticString, A: Int, B: Int, BATCH: Int
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime AB = A * B
    var m = Transpose2D[A, B].make[target, Deterministic](ctx)

    # Distinct value per (b,k) so any wrong index is caught.
    var x = Tensor.alloc(BATCH * AB)
    for bb in range(BATCH):
        for k in range(AB):
            x.data[bb * AB + k] = Scalar[DT](bb * 1000 + k) * 0.001
    var go = Tensor.alloc(BATCH * AB)
    for i in range(BATCH * AB):
        go.data[i] = Scalar[DT]((i % 13) - 6) * 0.07

    var out = Tensor.alloc(BATCH * AB)
    var gin = Tensor.alloc(BATCH * AB)

    comptime if target == "cpu":
        m.forward["cpu", BATCH](TensorRefs[1](x), out, None)
        m.vjp["cpu", BATCH](TensorRefs[1](x), go, TensorRefs[1](gin), None)
    else:
        var c = ctx.value()
        x.upload(c)
        go.upload(c)
        m.forward["gpu", BATCH](TensorRefs[1](x), out, ctx)
        m.vjp["gpu", BATCH](TensorRefs[1](x), go, TensorRefs[1](gin), ctx)
        out.download(c)
        gin.download(c)

    # analytic reference: out[b, j*A+i] = x[b, i*B+j]; gin[b, i*B+j] = go[b, j*A+i]
    var bad_f = 0
    var bad_b = 0
    for bb in range(BATCH):
        for i in range(A):
            for j in range(B):
                var wf = x.data[bb * AB + i * B + j]
                if out.data[bb * AB + j * A + i] != wf:
                    bad_f += 1
                var wb = go.data[bb * AB + j * A + i]
                if gin.data[bb * AB + i * B + j] != wb:
                    bad_b += 1
    print("  ", target, " A=", A, " B=", B, " BATCH=", BATCH,
          " fwd_bad=", bad_f, " bwd_bad=", bad_b)
    return bad_f == 0 and bad_b == 0


def main() raises:
    var ctx = DeviceContext()
    print("Transpose2D parity (analytic permutation oracle) — B1")
    print("=" * 60)
    var ok = True
    # tile-edge / non-multiple-of-32 shapes + ViT-like
    ok = _run["cpu", 37, 50, 3](None) and ok
    ok = _run["gpu", 37, 50, 3](ctx) and ok
    ok = _run["cpu", 192, 196, 8](None) and ok
    ok = _run["gpu", 192, 196, 8](ctx) and ok
    ok = _run["cpu", 65, 33, 4](None) and ok
    ok = _run["gpu", 65, 33, 4](ctx) and ok
    print("=" * 60)
    if ok:
        print("ALL PASSED")
    else:
        print("FAILED")
