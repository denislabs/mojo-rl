# +--------------------------------------------------------------------------+ #
# | BatchNorm2D frozen mode — vs torchvision's FrozenBatchNorm2d
# +--------------------------------------------------------------------------+ #
"""`set_attr["frozen"]` must reproduce what both ACT implementations wrap their
ResNet backbone in.

    pixi run -e act-ref python tools/act/dump_act_reference.py \\
        --out /tmp/act_ref --only frozen_bn
    pixi run -e apple mojo run -I . tests/nn/test_batch_norm_2d_frozen.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_batch_norm_2d_frozen.mojo

## What frozen has to mean, and why each part needs its own check

`FrozenBatchNorm2d` (`references/act-main/detr/models/backbone.py:21`, and
LeRobot imports torchvision's) registers weight, bias, running_mean and
running_var as BUFFERS. All four are constants; none takes a gradient.

⚠ **That is not `training = False`.** Eval mode already uses the running
statistics and already stops the EMA — but it still lets the optimizer walk
gamma and beta. Three of the four tensors would keep moving. So this file
checks the parts separately:

  1. the forward matches torchvision's, IN TRAINING MODE — frozen has to
     override `training`, or a `train_mode(True)` after the freeze silently
     unfreezes the backbone;
  2. the running statistics do not move across a training-mode forward;
  3. gamma/beta gradients come back EXACTLY zero;
  4. `grad_input` still matches torch's autograd. A frozen BatchNorm is a fixed
     affine map and must still PASS gradient to the convolutions beneath it —
     a "freeze" that also blocked that would train nothing below the layer and
     read as a learning-rate problem.

Each of 1-4 is paired with the same measurement taken with `frozen = False`, so
a gate that passed because nothing ran at all would fail. That contrast is the
whole point: "the statistics did not move" is trivially true of a forward that
never happened.

⚠ The dump uses `running_var` in [0.76, 2.43] and a non-zero `running_mean` on
purpose. At the init values (mean 0, var 1) frozen BatchNorm is near-identity
and an implementation that ignored the statistics entirely would pass.
"""

from std.python import Python, PythonObject

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.batch_norm_2d import BatchNorm2D
from mojo_rl.deep_agents.act.refload import LoadRefParams, RefDump


comptime B = 2
comptime C = 6
comptime H = 5
comptime W = 4
comptime N = B * C * H * W
comptime TOL = 1e-5

comptime BN = BatchNorm2D[C, H, W]


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def ref_dir() raises -> String:
    var os = Python.import_module("os")
    var env = String(
        os.environ.get(PythonObject("ACT_REF_DIR"), PythonObject(""))
    )
    return env if env.byte_length() > 0 else String("/tmp/act_ref")


def make_loaded(d_dir: String) raises -> BN:
    """A BatchNorm2D carrying the dump's weights AND its running statistics."""
    var bn = BN.make["cpu", Kaiming]()
    var wl = LoadRefParams["fbn."](RefDump(String(d_dir)))
    bn.for_each_param["cpu"](wl, None, String(""))
    var sl = LoadRefParams["fbn."](RefDump(String(d_dir)))
    bn.for_each_state["cpu"](sl, None, String(""))
    if len(wl.missing) > 0 or len(sl.missing) > 0:
        raise Error("gate: the dump does not carry every BN tensor")
    return bn^


def run_once(
    mut bn: BN, ref d: RefDump, mut out: Tensor, mut gin: Tensor
) raises:
    """One training-mode forward + backward on the dump's input and grad_out."""
    var xp = TensorPack[1]()
    var xr = d.get(String("fbn_x"))
    xp[0].ensure(N)
    for i in range(N):
        xp[0].data[i] = xr[i]
    bn.forward["cpu", B](TensorRefs[1, MutAnyOrigin](xp[0]), out)

    var go = Tensor()
    var gr = d.get(String("fbn_go"))
    go.ensure(N)
    for i in range(N):
        go.data[i] = gr[i]
    var gp = TensorPack[1]()
    gp[0].ensure(N)
    bn.zero_grad["cpu"](None)
    bn.vjp["cpu", B](
        TensorRefs[1, MutAnyOrigin](xp[0]),
        go,
        TensorRefs[1, MutAnyOrigin](gp[0]),
    )
    gin.ensure(N)
    for i in range(N):
        gin.data[i] = gp[0].data[i]


def worst(ref a: Tensor, ref b: List[Scalar[DT]], n: Int) -> Float64:
    var w = Float64(0.0)
    for i in range(n):
        w = max(w, abs(Float64(a.data[i]) - Float64(b[i])))
    return w


def run_gpu(mut fails: Int, d_dir: String, ref d: RefDump) raises:
    """The same four checks on device.

    ⚠ NOT redundant with the CPU leg. The frozen reset is a different
    mechanism there — `enqueue_fill` on the gradient buffer rather than a host
    loop — and the eval-mode GPU backward it runs on top of is the path that
    was silently dropping gamma/beta entirely until `8860a831`. An
    implementation meant for a GPU training run cannot be gated only on CPU.
    """
    var ctx = DeviceContext()
    print("")
    print("  GPU leg — device: " + String(ctx.name()))

    # Weights are loaded on the HOST and uploaded: `LoadRefParams` fills
    # `param.val.data` and then uploads when a context is passed.
    var bn = BatchNorm2D[C, H, W].make["gpu", Kaiming](ctx)
    var wl = LoadRefParams["fbn."](RefDump(String(d_dir)))
    bn.for_each_param["gpu"](wl, ctx, String(""))
    var sl = LoadRefParams["fbn."](RefDump(String(d_dir)))
    bn.for_each_state["gpu"](sl, ctx, String(""))
    if len(wl.missing) > 0 or len(sl.missing) > 0:
        raise Error("gate: the dump does not carry every BN tensor (gpu)")

    bn.set_attr["frozen"](Scalar[DT](1.0))
    bn.set_attr["training"](Scalar[DT](1.0))

    var xp = TensorPack[1]()
    var xr = d.get(String("fbn_x"))
    xp[0].ensure(N)
    for i in range(N):
        xp[0].data[i] = xr[i]
    xp[0].upload(ctx)

    var out = Tensor()
    bn.forward["gpu", B](TensorRefs[1, MutAnyOrigin](xp[0]), out, ctx)
    ctx.synchronize()
    out.download(ctx)
    check(
        fails,
        "gpu: forward vs torchvision FrozenBatchNorm2d",
        worst(out, d.get(String("fbn_out")), N) < TOL,
        "max|diff| = " + String(worst(out, d.get(String("fbn_out")), N)),
    )

    var go = Tensor()
    var gr = d.get(String("fbn_go"))
    go.ensure(N)
    for i in range(N):
        go.data[i] = gr[i]
    go.upload(ctx)
    var gp = TensorPack[1]()
    gp[0].ensure_gpu(ctx, N)
    bn.zero_grad["gpu"](ctx)
    bn.vjp["gpu", B](
        TensorRefs[1, MutAnyOrigin](xp[0]),
        go,
        TensorRefs[1, MutAnyOrigin](gp[0]),
        ctx,
    )
    ctx.synchronize()
    gp[0].download(ctx)
    check(
        fails,
        "gpu: grad_input vs torch autograd",
        worst(gp[0], d.get(String("fbn_gin")), N) < TOL,
        "max|diff| = " + String(worst(gp[0], d.get(String("fbn_gin")), N)),
    )

    bn.gamma.grd.download(ctx)
    bn.beta.grd.download(ctx)
    var dg = Float64(0.0)
    var db = Float64(0.0)
    for c in range(C):
        dg = max(dg, abs(Float64(bn.gamma.grd.data[c])))
        db = max(db, abs(Float64(bn.beta.grd.data[c])))
    check(
        fails,
        "gpu: gamma/beta gradients are exactly zero",
        dg == 0.0 and db == 0.0,
        "max|dgamma| = " + String(dg) + ", max|dbeta| = " + String(db),
    )

    bn.running_mean.t.download(ctx)
    var rm_moved = Float64(0.0)
    var rmr = d.get(String("fbn.running_mean"))
    for c in range(C):
        rm_moved = max(
            rm_moved,
            abs(Float64(bn.running_mean.t.data[c]) - Float64(rmr[c])),
        )
    check(
        fails,
        "gpu: running statistics did not move",
        rm_moved == 0.0,
        "mean moved " + String(rm_moved),
    )


def main() raises:
    var fails = 0
    var d_dir = ref_dir()
    var os = Python.import_module("os")
    if not Bool(os.path.exists(PythonObject(d_dir + "/fbn_out.bin"))):
        print("MISSING DUMP: " + d_dir + "/fbn_out.bin")
        print(
            "  pixi run -e act-ref python tools/act/dump_act_reference.py"
            " --out /tmp/act_ref --only frozen_bn"
        )
        raise Error("frozen-BN dump not found")

    print("BatchNorm2D frozen-mode gate")
    print("  dump  " + d_dir)
    print("")
    var d = RefDump(String(d_dir))

    # ── frozen, in TRAINING mode ─────────────────────────────────────────
    # The order is deliberate: freeze, THEN ask for training mode. If `frozen`
    # did not override `training`, this is the call that would undo it.
    var bn = make_loaded(d_dir)
    bn.set_attr["frozen"](Scalar[DT](1.0))
    bn.set_attr["training"](Scalar[DT](1.0))

    var rm_before = List[Float64]()
    var rv_before = List[Float64]()
    for c in range(C):
        rm_before.append(Float64(bn.running_mean.t.data[c]))
        rv_before.append(Float64(bn.running_var.t.data[c]))

    var out = Tensor()
    var gin = Tensor()
    run_once(bn, d, out, gin)

    check(
        fails,
        "forward vs torchvision FrozenBatchNorm2d (in TRAINING mode)",
        worst(out, d.get(String("fbn_out")), N) < TOL,
        "max|diff| = " + String(worst(out, d.get(String("fbn_out")), N)),
    )
    check(
        fails,
        "grad_input vs torch autograd (frozen still passes gradient)",
        worst(gin, d.get(String("fbn_gin")), N) < TOL,
        "max|diff| = " + String(worst(gin, d.get(String("fbn_gin")), N)),
    )

    var rm_moved = Float64(0.0)
    var rv_moved = Float64(0.0)
    for c in range(C):
        rm_moved = max(
            rm_moved, abs(Float64(bn.running_mean.t.data[c]) - rm_before[c])
        )
        rv_moved = max(
            rv_moved, abs(Float64(bn.running_var.t.data[c]) - rv_before[c])
        )
    check(
        fails,
        "running statistics did not move",
        rm_moved == 0.0 and rv_moved == 0.0,
        "mean moved " + String(rm_moved) + ", var moved " + String(rv_moved),
    )

    var dg = Float64(0.0)
    var db = Float64(0.0)
    for c in range(C):
        dg = max(dg, abs(Float64(bn.gamma.grd.data[c])))
        db = max(db, abs(Float64(bn.beta.grd.data[c])))
    check(
        fails,
        "gamma/beta gradients are exactly zero",
        dg == 0.0 and db == 0.0,
        "max|dgamma| = " + String(dg) + ", max|dbeta| = " + String(db),
    )

    # ── the SAME measurements with frozen = False ────────────────────────
    # Without this, every check above passes on a layer that did nothing at
    # all: statistics that never moved and gradients that were never written
    # are indistinguishable from statistics that were held and gradients that
    # were zeroed.
    print("")
    print("  contrast — the same layer with frozen = False:")
    var bn2 = make_loaded(d_dir)
    bn2.set_attr["training"](Scalar[DT](1.0))
    var out2 = Tensor()
    var gin2 = Tensor()
    run_once(bn2, d, out2, gin2)

    var rm_moved2 = Float64(0.0)
    for c in range(C):
        rm_moved2 = max(
            rm_moved2, abs(Float64(bn2.running_mean.t.data[c]) - rm_before[c])
        )
    check(
        fails,
        "unfrozen: running statistics DO move",
        rm_moved2 > 1e-3,
        "mean moved " + String(rm_moved2),
    )
    var dg2 = Float64(0.0)
    for c in range(C):
        dg2 = max(dg2, abs(Float64(bn2.gamma.grd.data[c])))
    check(
        fails,
        "unfrozen: gamma DOES take gradient",
        dg2 > 1e-3,
        "max|dgamma| = " + String(dg2),
    )
    var fwd_differs = worst(out2, d.get(String("fbn_out")), N)
    check(
        fails,
        "unfrozen: the forward DIFFERS from the frozen reference",
        fwd_differs > 1e-3,
        "max|diff| = " + String(fwd_differs),
    )

    run_gpu(fails, d_dir, d)

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("frozen batch-norm gate failed")
