"""Dreamer4Agent.imag_train_step — CPU↔GPU facade parity (Phase 4.6).

    pixi run -e apple mojo run -I . tests/nn2/test_dreamer4_imag_agent_gpu.mojo

A `DYN_TARGET="gpu"` agent puts the dynamics transformer on the device (the
heavy imagination compute) while the task embedder, heads, and value head stay
on host — so `imag_train_step` runs the rollout's transformer forwards on GPU
and everything else (λ-returns, PMPO, value TD, head vjps) on CPU, unchanged.

This checks the device facade against the pure-CPU agent: identical (seeded)
params on both + identical rollout inputs ⇒ the returned value / policy losses
must agree to fp32 transformer parity (the GPU rollout itself is bit-checked in
test_dreamer4_imag_rollout_gpu; here we confirm the facade plumbs it correctly).
"""

from std.memory import alloc
from std.math import abs
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.deep_agents2.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents2.dreamerv3.twohot import symexp_twohot_bins


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
comptime NAGENT = 1
comptime NTASK = 2
comptime HHID = 16
comptime NACT = 3
comptime NBINS = 41
comptime NMTP = 1
comptime B = 2
comptime B_SELF = 1
comptime ADIM = NACT
comptime AHID = 2 * D
comptime K_IMAG = 2
comptime NCTX = 1
comptime ND = NSP * DSP

comptime ACPU = Dreamer4Agent[
    DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
    NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
    True, ADIM, AHID, K_IMAG, NCTX, "cpu",
]
comptime AGPU = Dreamer4Agent[
    DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
    NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
    True, ADIM, AHID, K_IMAG, NCTX, "gpu",
]


def main() raises:
    print("=" * 70)
    print("Dreamer4Agent.imag_train_step — CPU↔GPU facade parity (Phase 4.6)")
    print("=" * 70)

    var ctx = DeviceContext()

    # identical params: seed before each make so Xavier draws the same sequence
    # for dyn/te/ph/rh/vh/ph_prior (the GPU agent only differs in WHERE dyn lives)
    seed(31)
    var acpu = ACPU.make[target="cpu", INIT=Xavier]()
    seed(31)
    var agpu = AGPU.make[target="cpu", INIT=Xavier](ctx)
    acpu.snapshot_prior()
    agpu.snapshot_prior()

    var bins = _alloc(NBINS)
    symexp_twohot_bins[NBINS](bins, lo=Scalar[DT](-9.0))

    var ctxf = _alloc(B * NCTX * ND)
    for i in range(B * NCTX * ND):
        ctxf[i] = Scalar[DT](0.2)
    var u01 = _alloc(B * T)
    for i in range(B * T):
        u01[i] = Scalar[DT](0.15 + 0.2 * Float64(i % 4))
    var znoise = _alloc(B * T * ND)
    for i in range(B * T * ND):
        znoise[i] = Scalar[DT](0.1)
    var task_ids = _alloc(B)
    for b in range(B):
        task_ids[b] = Scalar[DT](Float64(b % NTASK))

    var lc = acpu.imag_train_step(ctxf, u01, znoise, task_ids, bins)
    var lg = agpu.imag_train_step(
        ctxf, u01, znoise, task_ids, bins, dctx=ctx
    )

    print("   value loss   cpu =", lc[0], " gpu =", lg[0])
    print("   policy loss  cpu =", lc[1], " gpu =", lg[1])
    var dv = abs(lc[0] - lg[0])
    var dp = abs(lc[1] - lg[1])
    print("   |Δ value| =", dv, "  |Δ policy| =", dp)

    assert_true(dv < 1e-2, "value loss must match CPU↔GPU")
    assert_true(dp < 1e-2, "policy loss must match CPU↔GPU")

    print("=" * 70)
    print("ALL PASSED — imag_train_step CPU↔GPU facade parity (Phase 4.6)")
    print("=" * 70)
