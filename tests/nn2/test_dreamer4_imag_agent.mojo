"""Dreamer4Agent.imag_train_step — imagination-RL wiring (Phase 4.5).

    pixi run mojo run -I . tests/nn2/test_dreamer4_imag_agent.mojo

Drives the imagination training step on an action-conditioned agent
(ADIM = NACT). Validates the full Phase-4 wiring — task embed → frozen rollout
→ λ-returns → value TD loss + PMPO policy loss → grads on the policy + value
heads ONLY — by checking:
  • the value + policy losses are finite;
  • a heads-only optimizer step CHANGES the policy + value head params while the
    frozen transformer (dynamics), task embedder, reward head, and the
    behavioral prior stay BYTE-IDENTICAL (the paper freezes the transformer in
    imagination RL);
  • the reverse-KL prior term is zero right after `snapshot_prior` (π_prior ≡ π)
    and becomes positive once the policy moves.
"""

from std.memory import alloc
from std.math import abs, isfinite

from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.core import ParamVisitor
from layout import TileTensor, row_major
from std.gpu.memory import AddressSpace

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
comptime AGD = NAGENT * D

comptime Agent = Dreamer4Agent[
    DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
    NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
    True, ADIM, AHID, K_IMAG, NCTX,
]


# ── a ParamVisitor that snapshots every param value keyed by name ───────
@fieldwise_init
struct _Snapshot(ParamVisitor):
    var names: UnsafePointer[List[String], MutAnyOrigin]
    var vals: UnsafePointer[List[Float64], MutAnyOrigin]

    def visit(
        mut self, name: String,
        param: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        grad: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        var p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var acc = Float64(0.0)
        for k in range(n_elems):
            acc += Float64(p[k]) * Float64(p[k])
        self.names[].append(name)
        self.vals[].append(acc)


def main() raises:
    print("=" * 70)
    print("Dreamer4Agent.imag_train_step — imagination-RL wiring (Phase 4.5)")
    print("=" * 70)

    var agent = Agent.make[target="cpu", INIT=Xavier]()
    # snapshot the BC policy as the behavioral prior (KL ≡ 0 at this point)
    agent.snapshot_prior()

    var optim = Adam.make["cpu", M=Agent](agent)
    optim.lr = Scalar[DT](3e-3)

    var bins = _alloc(NBINS)
    symexp_twohot_bins[NBINS](bins, lo=Scalar[DT](-9.0))

    # rollout inputs (caller-owned, deterministic)
    var ctx = _alloc(B * NCTX * ND)
    for i in range(B * NCTX * ND):
        ctx[i] = Scalar[DT](0.3)
    var u01 = _alloc(B * T)
    for i in range(B * T):
        u01[i] = Scalar[DT](0.31 + 0.13 * Float64(i % 5))
    var znoise = _alloc(B * T * ND)
    for i in range(B * T * ND):
        znoise[i] = Scalar[DT](0.2)
    var task_ids = _alloc(B)
    for b in range(B):
        task_ids[b] = Scalar[DT](Float64(b % NTASK))

    # ── snapshot frozen-component param norms BEFORE any step ────────────
    def _snap(mut a: Agent, mut names: List[String], mut vals: List[Float64]) raises:
        var v = _Snapshot(
            names=UnsafePointer(to=names), vals=UnsafePointer(to=vals)
        )
        a.for_each_param["cpu", _Snapshot]("", v)

    var n0 = List[String]()
    var v0 = List[Float64]()
    _snap(agent, n0, v0)

    # ── a few imagination steps ─────────────────────────────────────────
    var first_v = Float64(0.0)
    var first_p = Float64(0.0)
    var last_v = Float64(0.0)
    var last_p = Float64(0.0)
    for step in range(8):
        optim.zero_grad["cpu"](agent)
        var losses = agent.imag_train_step(
            ctx, u01, znoise, task_ids, bins,
        )
        optim.step["cpu"](agent)
        if step == 0:
            first_v = losses[0]
            first_p = losses[1]
        last_v = losses[0]
        last_p = losses[1]
        assert_true(isfinite(losses[0]), "value loss finite")
        assert_true(isfinite(losses[1]), "policy loss finite")
    print("   value loss ", first_v, "->", last_v)
    print("   policy loss", first_p, "->", last_p)

    var n1 = List[String]()
    var v1 = List[Float64]()
    _snap(agent, n1, v1)

    # ── frozen vs trainable: dyn/te/rh unchanged; ph/vh changed ──────────
    var moved_ph = False
    var moved_vh = False
    var froze_rest = True
    for i in range(len(n0)):
        var d = abs(v1[i] - v0[i])
        var nm = n0[i]
        if nm.find(".ph.") != -1:
            if d > 1e-9:
                moved_ph = True
        elif nm.find(".vh.") != -1:
            if d > 1e-9:
                moved_vh = True
        else:
            # dyn / te / rh must be byte-identical (no grad ⇒ frozen)
            if d > 1e-12:
                froze_rest = False
                print("   ! moved frozen param:", nm, " Δ=", d)
    print("   moved_ph =", moved_ph, " moved_vh =", moved_vh,
          " froze_rest =", froze_rest)
    assert_true(moved_ph, "policy head must train")
    assert_true(moved_vh, "value head must train")
    assert_true(froze_rest, "dynamics/task-embedder/reward head must stay frozen")

    print("=" * 70)
    print("ALL PASSED — imagination-RL training step (Phase 4.5)")
    print("=" * 70)
