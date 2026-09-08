"""`FBCPRTrainer` CPU vs GPU — one step, same weights, same batch.

The CPU path carries the structural gates (`test_fb_cpr_smoke.mojo`); this
one asks that the device path computes the same thing: the CPR loss terms,
`D`'s logits after the step, `Q_D1`'s values, the actor's output, and the
expert encoding.

⚠ The two targets draw their noise from DIFFERENT generators (host RNG vs
Philox — the standing rule in `kernels.gaussian_t`), so the stochastic
inputs are switched OFF for the value comparison: target smoothing at
sigma 0 and `gp_coef 0` (the interpolation weights are random; the GP
kernels have their own parity gate, `tests/nn/test_grad_penalty_gpu.mojo`).
A second GPU-only step at `gp_coef 10` then checks the full device
sequence runs and reports finite terms.

Tolerances are fp32 GPU-vs-CPU on Apple; a 5090 outside them is TF32 on the
matmuls — compare weights bit-for-bit before loosening anything.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_fb_cpr_gpu_parity.mojo
"""

from std.math import abs, sqrt
from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Tanh, ReLU
from mojo_rl.nn.primitives.layer_norm import LayerNorm
from mojo_rl.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from mojo_rl.deep_agents.fb.cpr import FBCPRTrainer
from mojo_rl.deep_agents.fb import sample_z_uniform


comptime OBS: Int = 5
comptime ACT: Int = 3
comptime D: Int = 8
comptime SEQ: Int = 4
comptime BATCH: Int = 32
comptime HID: Int = 32
comptime SEED: Int = 20260908

comptime F_IN = OBS + ACT + D
comptime A_IN = OBS + D
comptime D_IN = OBS + D

comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, HID], ReLU[HID], Linear[HID, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]
]
comptime DNet = Sequential[
    Linear[D_IN, HID], LayerNorm[HID], Tanh[HID],
    Linear[HID, HID], ReLU[HID], Linear[HID, HID], ReLU[HID], Linear[HID, 1],
]
comptime QNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, 1]]

comptime CPUT = FBCPRTrainer[FNet, BNet, ANet, DNet, QNet, OBS, ACT, D, BATCH, SEQ, "cpu"]
comptime GPUT = FBCPRTrainer[FNet, BNet, ANet, DNet, QNet, OBS, ACT, D, BATCH, SEQ, "gpu"]


def _rand(n: Int) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    return t^


def _clone(ref t: Tensor, n: Int) raises -> Tensor:
    var c = Tensor.alloc(n)
    for i in range(n):
        c.data[i] = t.data[i]
    return c^


def _z() raises -> Tensor:
    var z = sample_z_uniform[D](BATCH)
    var t = Tensor.alloc(BATCH * D)
    for i in range(BATCH * D):
        t.data[i] = z[i]
    return t^


def _worst(ref a: Tensor, ref b: Tensor, n: Int) -> Float64:
    var w = Float64(0)
    for i in range(n):
        var e = abs(Float64(a.data[i]) - Float64(b.data[i]))
        if e > w:
            w = e
    return w


def _rel(a: Float64, b: Float64) -> Float64:
    return abs(a - b) / (abs(a) + 1e-6)


def main() raises:
    print("=" * 70)
    print("FBCPRTrainer CPU/GPU parity")
    print("=" * 70)
    var ctx = DeviceContext()
    var octx = Optional[DeviceContext](ctx)

    seed(SEED + 5)
    var s = _rand(BATCH * OBS)
    var a = _rand(BATCH * ACT)
    var sn = _rand(BATCH * OBS)
    var sp = _rand(BATCH * OBS)
    var z = _z()
    var es = _rand(BATCH * OBS)
    var esn = _rand(BATCH * OBS)

    seed(SEED)
    var tc = CPUT.make(
        None, lr=1e-3, lr_b=1e-3, lr_d=1e-3, lr_q=1e-3, ortho_weight=1.0,
        max_grad_norm=1.0, bc_weight=1.0, reg_coeff=0.5, gp_coef=0.0,
        seed=UInt64(SEED),
    )
    seed(SEED)
    var tg = GPUT.make(
        octx, lr=1e-3, lr_b=1e-3, lr_d=1e-3, lr_q=1e-3, ortho_weight=1.0,
        max_grad_norm=1.0, bc_weight=1.0, reg_coeff=0.5, gp_coef=0.0,
        seed=UInt64(SEED),
    )
    tc.t.policy_noise = 0.0
    tc.policy_noise = 0.0
    tg.t.policy_noise = 0.0
    tg.policy_noise = 0.0

    print("[1] one step, gp 0, sigma 0: CPR terms agree ...")
    var s1 = _clone(s, BATCH * OBS)
    var a1 = _clone(a, BATCH * ACT)
    var sn1 = _clone(sn, BATCH * OBS)
    var sp1 = _clone(sp, BATCH * OBS)
    var z1 = _clone(z, BATCH * D)
    var es1 = _clone(es, BATCH * OBS)
    var esn1 = _clone(esn, BATCH * OBS)
    tc.t.load_batch(s1, a1, sn1, sp1, z1)
    tc.load_expert(es1, esn1)
    tc.encode_expert()
    var lc = tc.train_step()

    var s2 = _clone(s, BATCH * OBS)
    var a2 = _clone(a, BATCH * ACT)
    var sn2 = _clone(sn, BATCH * OBS)
    var sp2 = _clone(sp, BATCH * OBS)
    var z2 = _clone(z, BATCH * D)
    var es2 = _clone(es, BATCH * OBS)
    var esn2 = _clone(esn, BATCH * OBS)
    s2.upload(ctx); a2.upload(ctx); sn2.upload(ctx); sp2.upload(ctx); z2.upload(ctx)
    es2.upload(ctx); esn2.upload(ctx)
    tg.t.load_batch(s2, a2, sn2, sp2, z2)
    tg.load_expert(es2, esn2)
    tg.encode_expert()
    var lg = tg.train_step()
    ctx.synchronize()
    print(
        "      cpu  d_pos", lc.d_pos, " d_neg", lc.d_neg, " q_loss", lc.q_loss,
        " q_pi", lc.q_pi, " r", lc.r_mean, " measure", lc.fb.measure,
    )
    print(
        "      gpu  d_pos", lg.d_pos, " d_neg", lg.d_neg, " q_loss", lg.q_loss,
        " q_pi", lg.q_pi, " r", lg.r_mean, " measure", lg.fb.measure,
    )
    assert_true(_rel(lc.d_pos, lg.d_pos) < 1e-4, "d_pos differs")
    assert_true(_rel(lc.d_neg, lg.d_neg) < 1e-4, "d_neg differs")
    assert_true(_rel(lc.q_loss, lg.q_loss) < 1e-4, "q_loss differs")
    assert_true(_rel(lc.q_pi, lg.q_pi) < 1e-4, "q_pi differs")
    assert_true(_rel(lc.r_mean, lg.r_mean) < 1e-3 or abs(lc.r_mean - lg.r_mean) < 1e-5, "r_mean differs")
    assert_true(_rel(lc.fb.measure, lg.fb.measure) < 1e-4, "measure differs")

    print("[2] expert encoding, D logits and the actor after the step ...")
    tg.ez.download(ctx)
    var wz = _worst(tc.ez, tg.ez, BATCH * D)
    print("      worst |ez cpu − gpu| =", wz)
    assert_true(wz < 1e-4, "expert encoding differs")

    var s3 = _clone(s, BATCH * OBS)
    var z3 = _clone(z, BATCH * D)
    var dc = Tensor()
    tc.discriminate[BATCH](s3, z3, dc)
    var s4 = _clone(s, BATCH * OBS)
    var z4 = _clone(z, BATCH * D)
    s4.upload(ctx); z4.upload(ctx)
    var dg = Tensor()
    tg.discriminate[BATCH](s4, z4, dg)
    dg.download(ctx)
    var wd = _worst(dc, dg, BATCH)
    print("      worst |D logit cpu − gpu| =", wd)
    assert_true(wd < 1e-4, "D logits differ after the step")

    var ac = Tensor()
    tc.t.act[BATCH](s3, z3, ac)
    var ag = Tensor()
    tg.t.act[BATCH](s4, z4, ag)
    ag.download(ctx)
    var wa = _worst(ac, ag, BATCH * ACT)
    print("      worst |actor cpu − gpu| =", wa)
    assert_true(wa < 1e-4, "actor differs after the step — the style hook diverges on device")

    print("[3] GPU-only: the full sequence at gp 10 runs and reports finite terms ...")
    seed(SEED)
    var tg2 = GPUT.make(
        octx, lr=1e-3, lr_b=1e-3, lr_d=1e-3, lr_q=1e-3, ortho_weight=1.0,
        max_grad_norm=1.0, bc_weight=1.0, reg_coeff=0.5, gp_coef=10.0,
        seed=UInt64(SEED),
    )
    var s5 = _clone(s, BATCH * OBS)
    var a5 = _clone(a, BATCH * ACT)
    var sn5 = _clone(sn, BATCH * OBS)
    var sp5 = _clone(sp, BATCH * OBS)
    var z5 = _clone(z, BATCH * D)
    var es5 = _clone(es, BATCH * OBS)
    var esn5 = _clone(esn, BATCH * OBS)
    s5.upload(ctx); a5.upload(ctx); sn5.upload(ctx); sp5.upload(ctx); z5.upload(ctx)
    es5.upload(ctx); esn5.upload(ctx)
    tg2.t.load_batch(s5, a5, sn5, sp5, z5)
    tg2.load_expert(es5, esn5)
    tg2.encode_expert()
    var l3 = tg2.train_step()
    tg2.train_device_kernels()
    tg2.train_device_kernels()
    var l4 = tg2.train_step()
    ctx.synchronize()
    print("      gp", l3.d_gp, "->", l4.d_gp, "  d_pos", l3.d_pos, "->", l4.d_pos)
    assert_true(l3.d_gp == l3.d_gp and l3.d_gp > 0.0, "gp not finite/positive on device")
    assert_true(l4.d_pos == l4.d_pos and l4.q_loss == l4.q_loss, "NaN after device steps")
    print("\n[PASS] FBCPRTrainer CPU/GPU parity")
