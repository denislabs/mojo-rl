"""EZv2 PER train-step IS-weighting (6c-2) — GPU, tiny MLP nets.

Verifies the importance-sampling-weight path added to
`ezv2_unroll_train_step_gpu`:

  1. **All-ones IS weights ≡ no weights** — same reported loss AND bit-identical
     parameter after the step (w_b = 1 scales every grad row by 1 → no-op).
  2. **Varied IS weights change the update** — the post-step parameter differs
     from the all-ones run (gradients are actually weighted per sample).
  3. **`out_prio` is populated** — finite, non-negative (it is a soft-CE).

The reported loss is the UNWEIGHTED mean (IS weights touch only gradients), so
checks 1/2 compare the post-step parameter, captured via a `ParamVisitor`.
Uses small MLP nets (fast compile); the PER logic is net-agnostic.

Run (GPU env required):
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_per_train_step.mojo
"""

from std.memory import alloc
from std.math import abs, isnan, isinf
from std.random import seed
from std.gpu import global_idx
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext
from std.testing import assert_true
from layout import Layout, LayoutTensor, TileTensor

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.nn.storage.optimizer.sgd import SGD
from mojo_rl.nn.core import ParamVisitor
from mojo_rl.deep_agents.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, MZPredNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents.efficient_zero_v2.blocks import (
    ezv2_unroll_train_step_gpu,
)
from mojo_rl.deep_agents.efficient_zero_v2.unroll_scratch import EZV2UnrollScratch


comptime OBS = 4
comptime ACT = 2
comptime LATENT = 16
comptime BINS = 11
comptime H = 32
comptime PROJ = 16
comptime PROJ_HID = 16
comptime BOTTLENECK = 8
comptime B = 4
comptime K = 2

comptime Rep = MZRepNet[OBS, LATENT, H]
comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _grab_k(
    dst: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < 1:
        dst[i] = rebind[Scalar[DT]](src[i])


# Capture the device pointer of the first param (for a 1-elem D2H grab).
struct _FirstAddr(ParamVisitor):
    var seen: Bool
    var ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    def __init__(out self):
        self.seen = False
        self.ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=Int(0)
        )

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
        if not self.seen and n_elems > 0:
            self.ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
            self.seen = True


def _read_first_pred_param(ctx: DeviceContext, mut pred: Pred) raises -> Float64:
    """D2H the first element of pred's first param via a 1-thread grab kernel."""
    var probe = _FirstAddr()
    pred.for_each_param["gpu", _FirstAddr](String(""), probe)
    var d_one = ctx.enqueue_create_buffer[DT](1)
    var h_one = ctx.enqueue_create_host_buffer[DT](1)
    var dst = LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin](
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](d_one.unsafe_ptr())
    )
    var src = LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin](probe.ptr)
    ctx.enqueue_function[_grab_k](dst, src, grid_dim=1, block_dim=1)
    ctx.enqueue_copy(h_one, d_one)
    ctx.synchronize()
    return Float64(h_one.unsafe_ptr()[0])


def _fill_batch(
    obs_seq: UnsafePointer[Scalar[DT], MutAnyOrigin],
    actions: UnsafePointer[Scalar[DT], MutAnyOrigin],
    pol: UnsafePointer[Scalar[DT], MutAnyOrigin],
    val: UnsafePointer[Scalar[DT], MutAnyOrigin],
    rew: UnsafePointer[Scalar[DT], MutAnyOrigin],
    cm: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    for i in range((K + 1) * B * OBS):
        obs_seq[i] = Scalar[DT]((Float64((i * 31) % 100) / 100.0) - 0.5)
    for i in range(K * B):
        actions[i] = Scalar[DT](i % ACT)
        rew[i] = Scalar[DT](0.1)
        cm[i] = Scalar[DT](1.0)
    for i in range((K + 1) * B):
        val[i] = Scalar[DT](0.2)
    for i in range((K + 1) * B * ACT):
        pol[i] = Scalar[DT](1.0) / Scalar[DT](ACT)


def _run_once(
    ctx: DeviceContext,
    obs_seq: UnsafePointer[Scalar[DT], MutAnyOrigin],
    actions: UnsafePointer[Scalar[DT], MutAnyOrigin],
    pol: UnsafePointer[Scalar[DT], MutAnyOrigin],
    val: UnsafePointer[Scalar[DT], MutAnyOrigin],
    rew: UnsafePointer[Scalar[DT], MutAnyOrigin],
    cm: UnsafePointer[Scalar[DT], MutAnyOrigin],
    isw: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]],
    out_prio: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]],
) raises -> Tuple[Float64, Float64]:
    """Build fresh (deterministic-init) nets, run ONE train step, return
    (reported_loss, post-step first-param-of-pred). Re-seeds the global RNG so
    all calls start from BIT-IDENTICAL parameters (init uses random_float64)."""
    seed(12345)
    var rep = Rep.make["gpu", Kaiming](Optional(ctx))
    var dyn = Dyn.make["gpu", Kaiming](Optional(ctx))
    var pred = Pred.make["gpu", Kaiming](Optional(ctx))
    var proj = Proj.make["gpu", Kaiming](Optional(ctx))
    var predh = Predh.make["gpu", Kaiming](Optional(ctx))
    # SGD (not Adam): Adam's first-step update ≈ lr·sign(g) is invariant to
    # positive gradient scaling, which would mask the IS-weight effect; SGD's
    # lr·g update reflects it directly.
    var orep = SGD(lr=Scalar[DT](0.1))
    var odyn = SGD(lr=Scalar[DT](0.1))
    var opred = SGD(lr=Scalar[DT](0.1))
    var oproj = SGD(lr=Scalar[DT](0.1))
    var opredh = SGD(lr=Scalar[DT](0.1))
    var scratch = EZV2UnrollScratch[B, K, OBS, ACT, LATENT, BINS, PROJ].make(ctx)

    var loss = Float64(
        ezv2_unroll_train_step_gpu[Rep, Dyn, Pred, Proj, Predh,
            B, K, OBS, ACT, LATENT, BINS,
        ](
            ctx, scratch, rep, dyn, pred, proj, predh,
            orep, odyn, opred, oproj, opredh,
            obs_seq, actions, pol, val, rew,
            Scalar[DT](-10.0), Scalar[DT](10.0),
            value_coef=Scalar[DT](0.25), consistency_coef=Scalar[DT](2.0),
            cons_mask=cm, is_weights=isw, out_prio=out_prio,
        )
    )
    ctx.synchronize()
    # Read pred's first param value off the device through a 1-elem D2H grab.
    var firstv = _read_first_pred_param(ctx, pred)
    _ = rep^; _ = dyn^; _ = pred^; _ = proj^; _ = predh^
    _ = orep^; _ = odyn^; _ = opred^; _ = oproj^; _ = opredh^
    _ = scratch^
    return (loss, firstv)


def main() raises:
    print("=" * 70)
    print("EZv2 PER train-step IS-weighting (6c-2) — GPU")
    print("=" * 70)
    var ctx = DeviceContext()

    var obs_seq = _a((K + 1) * B * OBS)
    var actions = _a(K * B)
    var pol = _a((K + 1) * B * ACT)
    var val = _a((K + 1) * B)
    var rew = _a(K * B)
    var cm = _a(K * B)
    _fill_batch(obs_seq, actions, pol, val, rew, cm)

    var ones = _a(B)
    var varied = _a(B)
    var prio = _a(B)
    for b in range(B):
        ones[b] = Scalar[DT](1.0)
        prio[b] = Scalar[DT](-1.0)
    varied[0] = Scalar[DT](2.0); varied[1] = Scalar[DT](0.5)
    varied[2] = Scalar[DT](1.0); varied[3] = Scalar[DT](0.3)

    # (A) no weights
    var ra = _run_once(ctx, obs_seq, actions, pol, val, rew, cm, None, None)
    # (B) all-ones weights + out_prio
    var rb = _run_once(ctx, obs_seq, actions, pol, val, rew, cm, ones, prio)
    # (C) varied weights
    var rc = _run_once(ctx, obs_seq, actions, pol, val, rew, cm, varied, None)

    print("  loss  none/ones/varied:", ra[0], rb[0], rc[0])
    print("  param none/ones/varied:", ra[1], rb[1], rc[1])

    # 1. all-ones ≡ none (loss + post-step param bit-identical)
    assert_true(ra[0] == rb[0], "reported loss: ones == none")
    assert_true(ra[1] == rb[1], "post-step param: ones == none (w=1 no-op)")
    # 2. varied weights change the update
    assert_true(abs(rc[1] - rb[1]) > Float64(1e-9),
                "varied IS weights change the post-step parameter")
    # 3. out_prio populated finite + non-negative (soft-CE)
    var prio_ok = True
    for b in range(B):
        var p = prio[b]
        if isnan(p) or isinf(p) or p < Scalar[DT](0.0):
            prio_ok = False
    print("  out_prio:", prio[0], prio[1], prio[2], prio[3])
    assert_true(prio_ok, "out_prio finite + non-negative")

    obs_seq.free(); actions.free(); pol.free(); val.free(); rew.free()
    cm.free(); ones.free(); varied.free(); prio.free()
    print("=" * 70)
    print("PASSED")
    print("=" * 70)
