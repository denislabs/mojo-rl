"""SpaceTimeTranspose parity: forward + vjp vs an INDEPENDENT analytic
permutation reference. CPU + GPU. Validates the vectorized run-copy GPU
rewrite (B2): D%4==0 takes the vec path, D%4!=0 the scalar fallback.

Pure data movement (no arithmetic) → bit-exact on both paths, no TF32 caveat.

Run: pixi run -e apple mojo run -I . mojo_rl/nn/storage/spikes/spike_stt_parity.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.space_time_transpose import SpaceTimeTranspose
from mojo_rl.nn.core.initializer import Deterministic


def _run[
    target: StaticString, T: Int, S: Int, D: Int, BATCH: Int
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TSD = T * S * D
    var m = SpaceTimeTranspose[T, S, D].make[target, Deterministic](ctx)

    var x = Tensor.alloc(BATCH * TSD)
    for i in range(BATCH * TSD):
        x.data[i] = Scalar[DT](i % 9973) * 0.001
    var go = Tensor.alloc(BATCH * TSD)
    for i in range(BATCH * TSD):
        go.data[i] = Scalar[DT]((i % 13) - 6) * 0.07

    var out = Tensor.alloc(BATCH * TSD)
    var gin = Tensor.alloc(BATCH * TSD)

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

    # analytic: out[b,(s*T+t)*D+d] = x[b,(t*S+s)*D+d];
    #           gin[b,(t*S+s)*D+d] = go[b,(s*T+t)*D+d]
    var bad_f = 0
    var bad_b = 0
    for b in range(BATCH):
        for t in range(T):
            for s in range(S):
                for d in range(D):
                    var src_off = b * TSD + (t * S + s) * D + d
                    var dst_off = b * TSD + (s * T + t) * D + d
                    if out.data[dst_off] != x.data[src_off]:
                        bad_f += 1
                    if gin.data[src_off] != go.data[dst_off]:
                        bad_b += 1
    print("  ", target, " T=", T, " S=", S, " D=", D, " BATCH=", BATCH,
          " fwd_bad=", bad_f, " bwd_bad=", bad_b)
    return bad_f == 0 and bad_b == 0


def main() raises:
    var ctx = DeviceContext()
    print("SpaceTimeTranspose parity (analytic permutation oracle) — B2")
    print("=" * 62)
    var ok = True
    # D%4==0 → vec path
    ok = _run["cpu", 5, 7, 16, 3](None) and ok
    ok = _run["gpu", 5, 7, 16, 3](ctx) and ok
    ok = _run["cpu", 8, 6, 64, 4](None) and ok
    ok = _run["gpu", 8, 6, 64, 4](ctx) and ok
    # D%4!=0 → scalar fallback
    ok = _run["cpu", 4, 5, 6, 2](None) and ok
    ok = _run["gpu", 4, 5, 6, 2](ctx) and ok
    print("=" * 62)
    if ok:
        print("ALL PASSED")
    else:
        print("FAILED")
