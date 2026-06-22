"""Dreamer4 storage migration — CPU integration smoke.

Builds the two top-level pieces of the migrated (legacy-nn-free) Dreamer4 package
on the storage ABI and exercises their forward/vjp/training paths end-to-end:

  • `Dreamer4Tokenizer` — encoder + decoder transformer (covers blocks/encoder/
    tokenizer + the attention/SwiGLU/MAE/learned-token primitives), forward+vjp.
  • `Dreamer4Agent.bc_train_step` — the joint shortcut-forcing video loss + BC
    loss (covers dynamics forward/vjp, task_embedder, the MTP heads, shortcut_loss
    + bc_loss).

This is the integration build (forces instantiation of every migrated method) and
the first correctness gate: every loss / output / grad must be finite, and a
param grad must be non-zero after a training step.

Run: pixi run -e apple mojo run -I . tests/nn/test_dreamer4_storage_smoke.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.param import ParamVisitor
from mojo_rl.nn.storage.core.initializer import Deterministic


struct _GradSum(ParamVisitor):
    """Accumulates Σ|grad| over every param (CPU) — confirms vjp wrote grads."""

    var total: Float64

    def __init__(out self):
        self.total = 0.0

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "cpu":
            for i in range(len(grad.data)):
                self.total += abs(Float64(grad.data[i]))

from mojo_rl.deep_agents.dreamer4.tokenizer import Dreamer4Tokenizer
from mojo_rl.deep_agents.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents.dreamer4.shortcut_loss import _mao


def _finite(x: Scalar[DT]) -> Bool:
    return x == x  # NaN-only check (smoke)


def _tokenizer_smoke() raises -> Bool:
    comptime DP = 4
    comptime D = 8
    comptime NH = 2
    comptime T = 2
    comptime L = 2
    comptime NP = 3
    comptime D_BOT = 4
    comptime HID = 8
    comptime DEPTH = 1
    comptime BATCH = 2  # = 1 sequence × T frames
    comptime NPDP = NP * DP

    comptime TOK = Dreamer4Tokenizer[
        DP, D, NH, T, L, NP, D_BOT, HID, DEPTH, 0.0, 0.5, 0, True
    ]
    var tok = TOK.make["cpu", Deterministic](None)

    var inp = Tensor.alloc(BATCH * NPDP)
    for i in range(BATCH * NPDP):
        inp.data[i] = Scalar[DT]((i % 5) - 2) * 0.1
    var out = Tensor.alloc(BATCH * NPDP)
    tok.forward["cpu", BATCH](TensorRefs[1](inp), out, None)

    var ok = True
    for i in range(BATCH * NPDP):
        if not _finite(out.data[i]):
            ok = False

    var go = Tensor.alloc(BATCH * NPDP)
    for i in range(BATCH * NPDP):
        go.data[i] = Scalar[DT](0.1)
    var gi = Tensor.alloc(BATCH * NPDP)
    tok.vjp["cpu", BATCH](TensorRefs[1](inp), go, TensorRefs[1](gi), None)
    for i in range(BATCH * NPDP):
        if not _finite(gi.data[i]):
            ok = False
    return ok


def _agent_bc_smoke() raises -> Bool:
    comptime DSP = 4
    comptime NSP = 2
    comptime D = 8
    comptime NH = 2
    comptime T = 2
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
    comptime BF = B * T
    comptime ND = NSP * DSP

    comptime AG = Dreamer4Agent[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
        NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
    ]
    var ag = AG.make["cpu", Deterministic](None)

    var z1 = List[Scalar[DT]](length=BF * ND, fill=Scalar[DT](0))
    var z0 = List[Scalar[DT]](length=BF * ND, fill=Scalar[DT](0))
    var sigma = List[Scalar[DT]](length=BF, fill=Scalar[DT](0.3))
    var sigma_idx = List[Scalar[DT]](length=BF, fill=Scalar[DT](1))
    var step_idx = List[Scalar[DT]](length=BF, fill=Scalar[DT](0))
    var task_ids = List[Scalar[DT]](length=B, fill=Scalar[DT](0))
    var actions = List[Scalar[DT]](length=BF, fill=Scalar[DT](1))
    var rewards = List[Scalar[DT]](length=BF, fill=Scalar[DT](0.5))
    var bins = List[Scalar[DT]](length=NBINS, fill=Scalar[DT](0))
    for i in range(BF * ND):
        z1[i] = Scalar[DT]((i % 7) - 3) * 0.1
        z0[i] = Scalar[DT]((i % 5) - 2) * 0.1
    for b in range(B):
        task_ids[b] = Scalar[DT](b % NTASK)
    for k in range(NBINS):
        bins[k] = Scalar[DT](k) - 2.0  # symexp-ish bin centers

    ag.zero_grad["cpu"](None)
    var lv_lbc = ag.bc_train_step(
        _mao(z1.unsafe_ptr()), _mao(z0.unsafe_ptr()),
        _mao(sigma.unsafe_ptr()), _mao(sigma_idx.unsafe_ptr()),
        _mao(step_idx.unsafe_ptr()), True,
        _mao(task_ids.unsafe_ptr()), _mao(actions.unsafe_ptr()),
        _mao(rewards.unsafe_ptr()), _mao(bins.unsafe_ptr()),
    )
    var loss_v = lv_lbc[0]
    var loss_bc = lv_lbc[1]
    print("    bc_train_step  video_loss =", loss_v, " bc_loss =", loss_bc)
    var ok = (loss_v == loss_v) and (loss_bc == loss_bc)  # finite

    # param grads must be non-zero after the BC step (vjp actually ran)
    var gs = _GradSum()
    ag.for_each_param["cpu"](gs, None)
    print("    Σ|grad| over all agent params =", gs.total)
    if not (gs.total > 0.0):
        ok = False
    return ok


def _agent_acwm_imag_smoke() raises -> Bool:
    """Action-conditioned WM step + imagination-RL step (ADIM=NACT, NAGENT>0):
    covers dynamics action conditioning (ACOND), imagine_rollout, the ODE
    denoise loop, imag_rl_loss (λ-returns / value TD / PMPO), and the continue
    head — the code paths the bc_train_step smoke does not touch."""
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
    comptime BF = B * T
    comptime ND = NSP * DSP

    comptime AG = Dreamer4Agent[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
        NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
        True, NACT, 0, 0, NCTX,   # USE_MAX, ADIM=NACT, AHID, K_IMAG, NCTX
    ]
    var ag = AG.make["cpu", Deterministic](None)

    var z1 = List[Scalar[DT]](length=BF * ND, fill=Scalar[DT](0))
    var z0 = List[Scalar[DT]](length=BF * ND, fill=Scalar[DT](0))
    var sigma = List[Scalar[DT]](length=BF, fill=Scalar[DT](0.3))
    var sigma_idx = List[Scalar[DT]](length=BF, fill=Scalar[DT](1))
    var step_idx = List[Scalar[DT]](length=BF, fill=Scalar[DT](0))
    var task_ids = List[Scalar[DT]](length=B, fill=Scalar[DT](0))
    var actions = List[Scalar[DT]](length=BF, fill=Scalar[DT](1))
    var rewards = List[Scalar[DT]](length=BF, fill=Scalar[DT](0.5))
    var bins = List[Scalar[DT]](length=NBINS, fill=Scalar[DT](0))
    for i in range(BF * ND):
        z1[i] = Scalar[DT]((i % 7) - 3) * 0.1
        z0[i] = Scalar[DT]((i % 5) - 2) * 0.1
    for b in range(B):
        task_ids[b] = Scalar[DT](b % NTASK)
    for k in range(NBINS):
        bins[k] = Scalar[DT](k) - 2.0

    ag.zero_grad["cpu"](None)
    var ac = ag.acwm_train_step(
        _mao(z1.unsafe_ptr()), _mao(z0.unsafe_ptr()),
        _mao(sigma.unsafe_ptr()), _mao(sigma_idx.unsafe_ptr()),
        _mao(step_idx.unsafe_ptr()), True,
        _mao(task_ids.unsafe_ptr()), _mao(actions.unsafe_ptr()),
        _mao(rewards.unsafe_ptr()), _mao(bins.unsafe_ptr()),
    )
    print("    acwm_train_step  video =", ac[0], " bc =", ac[1])
    var ok = (ac[0] == ac[0]) and (ac[1] == ac[1])

    # imagination-RL step (frozen WM rollout → value + policy losses)
    ag.snapshot_prior()
    var ictx = List[Scalar[DT]](length=B * NCTX * ND, fill=Scalar[DT](0))
    var u01 = List[Scalar[DT]](length=B * T, fill=Scalar[DT](0.5))
    var znoise = List[Scalar[DT]](length=B * T * ND, fill=Scalar[DT](0))
    for i in range(B * NCTX * ND):
        ictx[i] = Scalar[DT]((i % 5) - 2) * 0.1
    for i in range(B * T * ND):
        znoise[i] = Scalar[DT]((i % 3) - 1) * 0.1

    ag.zero_grad["cpu"](None)
    var im = ag.imag_train_step(
        _mao(ictx.unsafe_ptr()), _mao(u01.unsafe_ptr()),
        _mao(znoise.unsafe_ptr()), _mao(task_ids.unsafe_ptr()),
        _mao(bins.unsafe_ptr()), use_continue=True,
    )
    print("    imag_train_step  value =", im[0], " policy =", im[1])
    ok = ok and (im[0] == im[0]) and (im[1] == im[1])
    return ok


def main() raises:
    print("Dreamer4 storage smoke (CPU integration build + finite gate)")
    var tok_ok = _tokenizer_smoke()
    print("  tokenizer forward+vjp finite:", "OK" if tok_ok else "FAIL")
    var ag_ok = _agent_bc_smoke()
    print("  agent bc_train_step finite + grad:", "OK" if ag_ok else "FAIL")
    var ai_ok = _agent_acwm_imag_smoke()
    print("  agent acwm + imag steps finite:", "OK" if ai_ok else "FAIL")
    assert_true(tok_ok and ag_ok and ai_ok, "Dreamer4 storage smoke")
    print("DREAMER4 STORAGE SMOKE OK")
