"""SGD optimizer test — analytic CPU correctness + GPU finite smoke.

Verifies the SGD update matches PyTorch's `torch.optim.SGD` (momentum +
L2-COUPLED weight decay, nesterov=False) to float tolerance, on a tiny
`Linear[2,2]` (weight: decay=True; bias: decay=False). Grads are held constant
across N steps (no zero_grad), so the per-element trajectory is closed-form:

    d_p = g + (wd*p if decay else 0);  v = mom*v + d_p;  p = p - lr*v

Run:
    pixi run -e apple mojo run -I . tests/nn/test_sgd_optimizer.mojo
"""

from std.math import abs, isnan, isinf
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.core import ParamVisitor
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.optimizer.sgd import SGD


comptime IN = 2
comptime OUT = 2
comptime LR = Scalar[DT](0.1)
comptime MOM = Scalar[DT](0.9)
comptime WD = Scalar[DT](0.01)
comptime G = Scalar[DT](0.5)      # constant grad on every element
comptime N_STEPS = 3


# Set every grad element to a constant.
@fieldwise_init
struct SetGrad(ParamVisitor):
    var val: Scalar[DT]

    def visit(
        mut self, name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        var g = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        for i in range(n_elems):
            g[i] = self.val


# Capture the first element of the decay param and the no-decay param.
struct Capture(ParamVisitor):
    var p_decay: Scalar[DT]
    var p_nodecay: Scalar[DT]

    def __init__(out self):
        self.p_decay = Scalar[DT](0.0)
        self.p_nodecay = Scalar[DT](0.0)

    def visit(
        mut self, name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        var p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        if apply_decay:
            self.p_decay = p[0]
        else:
            self.p_nodecay = p[0]


def _ref(p0: Scalar[DT], decay: Bool) -> Scalar[DT]:
    """Closed-form PyTorch SGD trajectory for one element, constant grad G."""
    var p = p0
    var v = Scalar[DT](0.0)
    var wd: Scalar[DT] = WD if decay else Scalar[DT](0.0)
    for _ in range(N_STEPS):
        var d_p = G + wd * p
        v = MOM * v + d_p
        p = p - LR * v
    return p


def main() raises:
    print("=" * 70)
    print("SGD optimizer test (analytic CPU + GPU finite smoke)")
    print("=" * 70)

    # ── CPU analytic correctness ────────────────────────────────────
    var m = Linear[IN, OUT].make[target="cpu", INIT=Kaiming]()
    var pre = Capture()
    m.for_each_param["cpu", Capture]("m", pre)
    var w0 = pre.p_decay
    var b0 = pre.p_nodecay

    var opt = SGD.make[target="cpu", M = Linear[IN, OUT]](m)
    opt.lr = LR
    opt.momentum = MOM
    opt.weight_decay = WD

    var setg = SetGrad(G)
    for _ in range(N_STEPS):
        # grads held constant (no zero_grad) → matches the closed form
        m.for_each_param["cpu", SetGrad]("m", setg)
        opt.step["cpu", M = Linear[IN, OUT]](m)

    var post = Capture()
    m.for_each_param["cpu", Capture]("m", post)

    var w_ref = _ref(w0, True)
    var b_ref = _ref(b0, False)
    print("  weight: got", post.p_decay, " ref", w_ref)
    print("  bias  : got", post.p_nodecay, " ref", b_ref)
    assert_true(abs(post.p_decay - w_ref) < Scalar[DT](1e-5),
                "SGD weight (decay) matches PyTorch trajectory")
    assert_true(abs(post.p_nodecay - b_ref) < Scalar[DT](1e-5),
                "SGD bias (no decay) matches PyTorch trajectory")
    assert_true(abs(w_ref - b_ref) > Scalar[DT](1e-9) or abs(w0 - b0) < Scalar[DT](1e-9),
                "weight-decay path actually differs from no-decay (when w0!=b0)")
    _ = m^
    _ = opt^
    print("  ✓ CPU update is exact PyTorch SGD")

    # ── GPU finite smoke (real forward→vjp→step; output read via host) ──
    comptime B = 4
    with DeviceContext() as ctx:
        var gm = Linear[IN, OUT].make[target="gpu", INIT=Kaiming](ctx)
        var gopt = SGD.make[target="gpu", M = Linear[IN, OUT]](gm, ctx)
        gopt.lr = LR
        gopt.momentum = MOM
        gopt.weight_decay = WD
        gopt.max_grad_norm = Scalar[DT](5.0)   # exercise the clip pipeline

        var xin = ctx.enqueue_create_buffer[DT](B * IN)
        var yout = ctx.enqueue_create_buffer[DT](B * OUT)
        var go = ctx.enqueue_create_buffer[DT](B * OUT)
        var gx = ctx.enqueue_create_buffer[DT](B * IN)
        var xin_h = ctx.enqueue_create_host_buffer[DT](B * IN)
        var go_h = ctx.enqueue_create_host_buffer[DT](B * OUT)
        var y_h = ctx.enqueue_create_host_buffer[DT](B * OUT)
        ctx.synchronize()
        for i in range(B * IN):
            xin_h.unsafe_ptr()[i] = Scalar[DT](0.3)
        for i in range(B * OUT):
            go_h.unsafe_ptr()[i] = Scalar[DT](1.0)
        ctx.enqueue_copy(xin, xin_h)
        ctx.enqueue_copy(go, go_h)
        ctx.synchronize()

        var xin_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](xin.unsafe_ptr()),
            row_major[B, IN]())
        var yout_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](yout.unsafe_ptr()),
            row_major[B, OUT]())
        var go_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go.unsafe_ptr()),
            row_major[B, OUT]())
        var gx_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gx.unsafe_ptr()),
            row_major[B, IN]())

        # initial output
        gm.forward["gpu", B](xin_t, output=yout_t)
        ctx.enqueue_copy(y_h, yout)
        ctx.synchronize()
        var y0 = Float64(y_h.unsafe_ptr()[0])

        # N steps of forward→vjp→step (real device grads)
        for _ in range(N_STEPS):
            gm.forward["gpu", B](xin_t, output=yout_t)
            gopt.zero_grad["gpu", M = Linear[IN, OUT]](gm)
            gm.vjp["gpu", B](go_t, gx_t)
            gopt.step["gpu", M = Linear[IN, OUT]](gm)

        gm.forward["gpu", B](xin_t, output=yout_t)
        ctx.enqueue_copy(y_h, yout)
        ctx.synchronize()
        var y1 = Float64(y_h.unsafe_ptr()[0])
        print("  GPU out[0]:", y0, "→", y1)
        assert_true(y1 == y1 and y1 < 1e30 and y1 > -1e30, "GPU output finite")
        assert_true(abs(Scalar[DT](y1) - Scalar[DT](y0)) > Scalar[DT](1e-7),
                    "GPU params moved (output changed)")
        print("  ✓ GPU step finite + params moved (clip on)")
        _ = gm^
        _ = gopt^

    print("=" * 70)
    print("PASSED")
    print("=" * 70)
