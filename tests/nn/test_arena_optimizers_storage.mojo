"""Arena-mode parity for SGD + grad-clip (GPU), on the shared ParamArena.

1. SGD: adopt (arena, 1 kernel) vs per-param SGD over K identical steps →
   param values BIT-IDENTICAL (same elementwise math).
2. grad-clip: arena reduction+scale (opt.clip_grads, adopted) vs per-param
   clip_grad_norm over identical grads → same pre-clip norm + same clipped grads
   (within fp tolerance — the arena regroups the reduction).

Run: pixi run -e apple mojo run -I . tests/nn/test_arena_optimizers_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.param import ParamVisitor
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.optimizer.sgd import SGD
from mojo_rl.nn.storage.optimizer.grad_clip import clip_grad_norm


comptime D = 4
comptime H = 6
comptime O = 3
comptime B = 5
comptime NET = Sequential[Linear[D, H], Linear[H, O]]


struct _Capture(ParamVisitor):
    var vals: List[Scalar[DT]]
    var read_grad: Bool

    def __init__(out self, read_grad: Bool):
        self.vals = List[Scalar[DT]]()
        self.read_grad = read_grad

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if self.read_grad:
            grad.download(ctx.value())
            for i in range(N):
                self.vals.append(grad.data[i])
        else:
            param.download(ctx.value())
            for i in range(N):
                self.vals.append(param.data[i])


def _feed(mut x: Tensor, mut go: Tensor, step: Int, ctx: DeviceContext) raises:
    for i in range(B * D):
        x.data[i] = Scalar[DT](((i + step) % 5) - 2) * 0.3
    for i in range(B * O):
        go.data[i] = Scalar[DT](((i * 3 + step) % 7) - 3) * 0.4
    x.upload(ctx)
    go.upload(ctx)


def _populate_grads(mut net: NET, ctx: DeviceContext) raises:
    var x = Tensor.alloc(B * D); var go = Tensor.alloc(B * O)
    _feed(x, go, 0, ctx)
    var out = Tensor.alloc(B * O); var gi = Tensor.alloc(B * D)
    net.forward["gpu", B](TensorRefs[1](x), out, Optional(ctx))
    net.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(ctx))


def main() raises:
    var c = DeviceContext()
    print("Arena optimizers parity (SGD + grad-clip)")

    # ---- 1. SGD: per-param vs arena, K steps, bit-identical -------------
    comptime K = 4
    var a = NET.make["gpu", Deterministic](Optional(c))
    var optA = SGD(lr=1e-2)
    var b = NET.make["gpu", Deterministic](Optional(c))
    var optB = SGD(lr=1e-2)
    optB.adopt["gpu"](b, Optional(c))
    for step in range(K):
        var xa = Tensor.alloc(B * D); var ga = Tensor.alloc(B * O)
        _feed(xa, ga, step, c)
        var oa = Tensor.alloc(B * O); var gia = Tensor.alloc(B * D)
        a.zero_grad["gpu"](Optional(c))
        a.forward["gpu", B](TensorRefs[1](xa), oa, Optional(c))
        a.vjp["gpu", B](TensorRefs[1](xa), ga, TensorRefs[1](gia), Optional(c))
        optA.step["gpu"](a, Optional(c))

        var xb = Tensor.alloc(B * D); var gb = Tensor.alloc(B * O)
        _feed(xb, gb, step, c)
        var ob = Tensor.alloc(B * O); var gib = Tensor.alloc(B * D)
        optB.zero_grad["gpu"](b, Optional(c))
        b.forward["gpu", B](TensorRefs[1](xb), ob, Optional(c))
        b.vjp["gpu", B](TensorRefs[1](xb), gb, TensorRefs[1](gib), Optional(c))
        optB.step["gpu"](b, Optional(c))

    var ca = _Capture(False); a.for_each_param["gpu"](ca, Optional(c))
    var cb = _Capture(False); b.for_each_param["gpu"](cb, Optional(c))
    var sgd_max = Scalar[DT](0.0)
    for i in range(len(ca.vals)):
        var d = abs(ca.vals[i] - cb.vals[i])
        if d > sgd_max:
            sgd_max = d
    print("  SGD arena vs per-param max|A-B| =", sgd_max)
    var sgd_ok = sgd_max < Scalar[DT](1e-6)

    # ---- 2. grad-clip: arena vs per-param, identical grads --------------
    var p = NET.make["gpu", Deterministic](Optional(c))
    var optP = Adam(lr=1e-2)
    optP.adopt["gpu"](p, Optional(c))
    _populate_grads(p, c)
    var q = NET.make["gpu", Deterministic](Optional(c))
    _populate_grads(q, c)

    var normP = optP.clip_grads["gpu"](p, Scalar[DT](1e9), Optional(c))  # no clip
    var normQ = clip_grad_norm["gpu"](q, Scalar[DT](1e9), Optional(c))
    var norm_ok = abs(normP - normQ) < Scalar[DT](1e-3) and normP > Scalar[DT](0.0)
    print("  clip norm: arena =", normP, " per-param =", normQ)

    # Now actually clip both to half the norm, compare scaled grads.
    _ = optP.clip_grads["gpu"](p, normP * Scalar[DT](0.5), Optional(c))
    _ = clip_grad_norm["gpu"](q, normP * Scalar[DT](0.5), Optional(c))
    var gp = _Capture(True); p.for_each_param["gpu"](gp, Optional(c))
    var gq = _Capture(True); q.for_each_param["gpu"](gq, Optional(c))
    var clip_max = Scalar[DT](0.0)
    for i in range(len(gp.vals)):
        var d = abs(gp.vals[i] - gq.vals[i])
        if d > clip_max:
            clip_max = d
    print("  clipped-grad max|arena-perparam| =", clip_max)
    var clip_ok = norm_ok and clip_max < Scalar[DT](1e-3)

    # ---- 3. clip_grads_device (capture-safe) == clip_grads (non-capture) -
    # Same arena kernels; clip_grads_device just uses persistent scratch and
    # skips the norm D2H (read separately via read_clip_norm). Bit-identical.
    var r = NET.make["gpu", Deterministic](Optional(c))
    var optR = Adam(lr=1e-2)
    optR.adopt["gpu"](r, Optional(c))
    _populate_grads(r, c)
    var s = NET.make["gpu", Deterministic](Optional(c))
    var optS = Adam(lr=1e-2)
    optS.adopt["gpu"](s, Optional(c))
    _populate_grads(s, c)

    var thr = normP * Scalar[DT](0.5)  # force clipping (same grads as p)
    var normR = optR.clip_grads["gpu"](r, thr, Optional(c))  # non-capture path
    optS.clip_grads_device["gpu"](s, thr, Optional(c))       # capture-safe path
    var normS = optS.read_clip_norm(c)
    var gr = _Capture(True); r.for_each_param["gpu"](gr, Optional(c))
    var gs = _Capture(True); s.for_each_param["gpu"](gs, Optional(c))
    var dev_max = Scalar[DT](0.0)
    for i in range(len(gr.vals)):
        var d = abs(gr.vals[i] - gs.vals[i])
        if d > dev_max:
            dev_max = d
    print(
        "  clip_grads_device vs clip_grads: max|grad diff| =", dev_max,
        " norm diff =", abs(normR - normS),
    )
    var dev_ok = (
        dev_max < Scalar[DT](1e-6)
        and abs(normR - normS) < Scalar[DT](1e-4)
        and normR > Scalar[DT](0.0)
    )

    # ---- 4. on-device LR warmup == host LinearWarmup applied each step ---
    # `attach_warmup_schedule` runs the ramp on-device (capture-safe); the
    # reference sets `opt.lr = target·min(k/warmup,1)` on the host each step.
    # Same grads + same LR sequence + same bias correction → identical params.
    comptime WUP = 5
    var tgt = Scalar[DT](1e-2)
    var u = NET.make["gpu", Deterministic](Optional(c))
    var optU = Adam(lr=tgt)
    optU.adopt["gpu"](u, Optional(c))
    optU.attach_warmup_schedule(tgt, WUP)               # device schedule
    var w = NET.make["gpu", Deterministic](Optional(c))
    var optW = Adam(lr=tgt)
    optW.adopt["gpu"](w, Optional(c))                   # host-driven reference
    for k in range(8):
        optU.zero_grad["gpu"](u, Optional(c)); _populate_grads(u, c)
        optU.step["gpu"](u, Optional(c))
        var ratio = Scalar[DT](k) / Scalar[DT](WUP)
        optW.lr = tgt * ratio if k < WUP else tgt   # lr_at(k)
        optW.zero_grad["gpu"](w, Optional(c)); _populate_grads(w, c)
        optW.step["gpu"](w, Optional(c))
    var pu = _Capture(False); u.for_each_param["gpu"](pu, Optional(c))
    var pw = _Capture(False); w.for_each_param["gpu"](pw, Optional(c))
    var sched_max = Scalar[DT](0.0)
    for i in range(len(pu.vals)):
        var d = abs(pu.vals[i] - pw.vals[i])
        if d > sched_max:
            sched_max = d
    print("  device-warmup vs host-warmup: max|param diff| =", sched_max)
    var sched_ok = sched_max < Scalar[DT](1e-6)

    print(
        "  SGD:", "OK" if sgd_ok else "FAIL",
        " clip:", "OK" if clip_ok else "FAIL",
        " clip_device:", "OK" if dev_ok else "FAIL",
        " sched:", "OK" if sched_ok else "FAIL",
    )
    assert_true(
        sgd_ok and clip_ok and dev_ok and sched_ok, "arena optimizers parity"
    )
    print("ARENA OPTIMIZERS OK")
