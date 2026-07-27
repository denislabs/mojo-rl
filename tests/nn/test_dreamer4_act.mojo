"""Dreamer4Agent.act_from_latents — single-step acting smoke (CPU).

Validates the online acting path: a window of CLEAN latents → one frozen-dynamics
forward → policy head → action class. With a RANDOM (untrained) agent the action
VALUE is arbitrary, so this is a SHAPE / VALIDITY + DETERMINISM smoke, not a
policy-correctness test:
  • greedy (explore=False) returns an Int in [0, NACT) and is deterministic,
  • exploratory (explore=True) returns a valid index in [0, NACT) for several u01.

Run: pixi run mojo run -I . tests/nn/test_dreamer4_act.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents.dreamer4.shortcut_loss import _mao


def main() raises:
    print("Dreamer4Agent.act_from_latents smoke (CPU)")
    comptime DSP = 4
    comptime NSP = 2
    comptime D = 8
    comptime NH = 2
    comptime T = 3
    comptime NREG = 1
    comptime HID = 8
    comptime DEPTH = 1
    comptime KMAX = 4
    comptime NAGENT = 1
    comptime NTASK = 2
    comptime HHID = 8
    comptime NACT = 3
    comptime NBINS = 5
    comptime NMTP = 2
    comptime B = 2
    comptime B_SELF = 1
    comptime NCTX = 1
    comptime ND = NSP * DSP

    comptime AG = Dreamer4Agent[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
        NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
        True, NACT, 0, 0, NCTX,   # USE_MAX, ADIM=NACT, AHID, K_IMAG, NCTX
    ]
    var ag = AG.make["cpu", Deterministic](None)

    comptime NC = T   # use a full window
    var zw = List[Scalar[DT]](length=NC * ND, fill=Scalar[DT](0))
    for i in range(NC * ND):
        zw[i] = Scalar[DT]((i % 7) - 3) * 0.1
    var ahist = List[Scalar[DT]](length=T * NACT, fill=Scalar[DT](0))

    var ok = True

    # greedy: valid + deterministic
    var g0 = ag.act_from_latents(
        _mao(zw.unsafe_ptr()), NC, _mao(ahist.unsafe_ptr()), 0, False, 0.0
    )
    var g1 = ag.act_from_latents(
        _mao(zw.unsafe_ptr()), NC, _mao(ahist.unsafe_ptr()), 0, False, 0.0
    )
    print("  greedy action =", g0, " (repeat", g1, ")")
    if not (g0 >= 0 and g0 < NACT):
        ok = False
        print("  FAIL: greedy out of range")
    if g0 != g1:
        ok = False
        print("  FAIL: greedy not deterministic")

    # explore: valid index for several u01
    var us = List[Float64]()
    us.append(0.01)
    us.append(0.33)
    us.append(0.5)
    us.append(0.77)
    us.append(0.99)
    for k in range(len(us)):
        var a = ag.act_from_latents(
            _mao(zw.unsafe_ptr()), NC, _mao(ahist.unsafe_ptr()), 0, True, us[k]
        )
        print("  explore u01 =", us[k], " action =", a)
        if not (a >= 0 and a < NACT):
            ok = False
            print("  FAIL: explore out of range")

    # partial window (n_ctx < T) should still produce a valid action
    var ap = ag.act_from_latents(
        _mao(zw.unsafe_ptr()), 2, _mao(ahist.unsafe_ptr()), 1, False, 0.0
    )
    print("  partial-window (n_ctx=2) greedy action =", ap)
    if not (ap >= 0 and ap < NACT):
        ok = False
        print("  FAIL: partial-window out of range")

    print("  act_from_latents valid + deterministic:", "OK" if ok else "FAIL")
    assert_true(ok, "act_from_latents shape/validity/determinism smoke")
    print("DREAMER4 ACT GATE OK")
