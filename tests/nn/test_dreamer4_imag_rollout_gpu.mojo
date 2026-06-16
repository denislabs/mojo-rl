"""Dreamer 4 imagination rollout — CPU↔GPU parity (Phase 4.6).

    pixi run -e apple mojo run -I . tests/nn/test_dreamer4_imag_rollout_gpu.mojo

`imagine_rollout[FWD="gpu"]` runs the (frozen) dynamics transformer forward on
the device — the heavy compute — while the heads + all orchestration stay on
host. This checks it matches the pure-CPU rollout: same dynamics params (seeded)
+ the SAME host-side heads + identical action-sampling uniforms ⇒ the generated
trajectory (h, sampled actions, rewards, values) must agree to fp32 transformer
parity (~1e-5). Actions must match EXACTLY (the uniforms sit mid-bin, so a
~1e-6 logit difference can't flip the categorical sample).
"""

from std.memory import alloc
from std.math import abs
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.deep_agents.dreamer4.dynamics import Dreamer4Dynamics
from mojo_rl.deep_agents.dreamer4.heads import (
    Dreamer4PolicyHead, Dreamer4ValueHead, Dreamer4RewardHead,
)
from mojo_rl.deep_agents.dreamer4.imag_rollout import imagine_rollout
from mojo_rl.deep_agents.dreamerv3.twohot import symexp_twohot_bins


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


comptime DSP = 4
comptime NSP = 4
comptime D = 8
comptime NH = 2
comptime T = 4
comptime NREG = 2
comptime HID = 16
comptime DEPTH = 2
comptime KMAX = 4
comptime K = 2
comptime NCTX = 1
comptime NAGENT = 1
comptime HHID = 16
comptime NACT = 3
comptime NBINS = 41
comptime NMTP = 1
comptime B = 2
comptime AGD = NAGENT * D
comptime ADIM = NACT
comptime AHID = 2 * D
comptime ND = NSP * DSP

comptime DYN = Dreamer4Dynamics[
    DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX, True, ADIM, AHID, NAGENT
]
comptime PH = Dreamer4PolicyHead[AGD, HHID, NACT, NMTP]
comptime VH = Dreamer4ValueHead[AGD, HHID, NBINS]
comptime RH = Dreamer4RewardHead[AGD, HHID, NBINS, NMTP]


def main() raises:
    print("=" * 70)
    print("Dreamer 4 imagination rollout — CPU↔GPU parity (Phase 4.6)")
    print("=" * 70)

    var ctx = DeviceContext()

    # identical dynamics params on CPU and GPU (seeded Xavier draws)
    seed(7)
    var dcpu = DYN.make[target="cpu", INIT=Xavier]()
    seed(7)
    var dgpu = DYN.make[target="gpu", INIT=Xavier](ctx)
    # heads are host-side in BOTH paths → build once, share
    seed(11)
    var ph = PH.make[target="cpu", INIT=Xavier]()
    seed(13)
    var vh = VH.make[target="cpu", INIT=Xavier]()
    seed(17)
    var rh = RH.make[target="cpu", INIT=Xavier]()

    var bins = _alloc(NBINS)
    symexp_twohot_bins[NBINS](bins, lo=Scalar[DT](-9.0))

    var ctxf = _alloc(B * NCTX * ND)
    for i in range(B * NCTX * ND):
        ctxf[i] = Scalar[DT](0.3)
    var agent_in = _alloc(B * T * AGD)
    for i in range(B * T * AGD):
        agent_in[i] = Scalar[DT](0.1)
    var u01 = _alloc(B * T)
    for i in range(B * T):
        u01[i] = Scalar[DT](0.15 + 0.2 * Float64(i % 4))   # mid-bin
    var znoise = _alloc(B * T * ND)
    for i in range(B * T * ND):
        znoise[i] = Scalar[DT](0.5)

    var hC = _alloc(B * T * AGD)
    var aC = _alloc(B * (T - 1))
    var rC = _alloc(B * T)
    var vC = _alloc(B * T)
    var hG = _alloc(B * T * AGD)
    var aG = _alloc(B * (T - 1))
    var rG = _alloc(B * T)
    var vG = _alloc(B * T)

    imagine_rollout[
        DYN, PH, VH, RH, B, T, NSP, DSP, KMAX, K, NCTX, AGD, NACT, NBINS, NMTP,
        "cpu",
    ](dcpu, ph, vh, rh, ctxf, agent_in, u01, znoise, bins, hC, aC, rC, vC)

    imagine_rollout[
        DYN, PH, VH, RH, B, T, NSP, DSP, KMAX, K, NCTX, AGD, NACT, NBINS, NMTP,
        "gpu",
    ](dgpu, ph, vh, rh, ctxf, agent_in, u01, znoise, bins, hG, aG, rG, vG,
      dctx=ctx)

    var max_h = Float64(0.0)
    for i in range(B * T * AGD):
        var d = abs(Float64(hC[i]) - Float64(hG[i]))
        if d > max_h:
            max_h = d
    var max_rv = Float64(0.0)
    for i in range(B * T):
        var dr = abs(Float64(rC[i]) - Float64(rG[i]))
        var dv = abs(Float64(vC[i]) - Float64(vG[i]))
        if dr > max_rv:
            max_rv = dr
        if dv > max_rv:
            max_rv = dv
    var act_match = True
    for i in range(B * (T - 1)):
        if Int(Float64(aC[i]) + 0.5) != Int(Float64(aG[i]) + 0.5):
            act_match = False

    print("   max|Δ h|        =", max_h)
    print("   max|Δ rew/val|  =", max_rv)
    print("   actions match   =", act_match)

    assert_true(act_match, "sampled actions must match CPU↔GPU")
    assert_true(max_h < 1e-3, "agent tokens h must match CPU↔GPU")
    assert_true(max_rv < 1e-3, "rewards/values must match CPU↔GPU")

    print("=" * 70)
    print("ALL PASSED — imagination rollout CPU↔GPU parity (Phase 4.6)")
    print("=" * 70)
