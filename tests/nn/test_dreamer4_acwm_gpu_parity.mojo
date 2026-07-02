"""Dreamer4 acwm_train_step CPU↔GPU parity gate.

Builds two identical (`Deterministic` init) agents — one DYN_TARGET="cpu", one
"gpu" — runs `acwm_train_step` (CPU) and `acwm_train_step_gpu` (GPU dynamics) on
the SAME synthetic inputs, and checks that (a) the returned (video, bc) losses and
(b) every param grad (dynamics on device, heads + task-embedder on host) agree to
a tight tolerance. This is the first coverage of the action-conditioned (ADIM>0)
GPU dynamics train path (the storage parity test uses ADIM=0).

Uses `do_boot=True` — the full shortcut-forcing objective, including the bootstrap
half-passes at the smaller self-row batch `BS=B_SELF*T`. This exercises the
dynamics GPU staging across a batch change (BS→BF); `acwm_train_step_gpu`
pre-sizes the dynamics GPU scratch to the full batch BF up front so the (grow-only)
staging buffers never overflow a later BF copy. CPU↔GPU grads match to ~3e-8.

Run: pixi run -e apple  mojo run -I . tests/nn/test_dreamer4_acwm_gpu_parity.mojo
     pixi run -e nvidia mojo run -I . tests/nn/test_dreamer4_acwm_gpu_parity.mojo
"""

from std.math import abs
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import Deterministic

from mojo_rl.deep_agents.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents.dreamer4.shortcut_loss import _mao


struct _GradCollect(ParamVisitor):
    """Collect every param's grad into a flat host list (walk order). Downloads
    device grads first when target=="gpu"."""

    var vals: List[Float64]

    def __init__(out self):
        self.vals = List[Float64]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            grad.download(ctx.value())
        for i in range(len(grad.data)):
            self.vals.append(Float64(grad.data[i]))


def main() raises:
    print("Dreamer4 acwm_train_step CPU↔GPU parity gate")
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
    comptime BF = B * T
    comptime ND = NSP * DSP

    var c = DeviceContext()

    comptime CPU_AG = Dreamer4Agent[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
        NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
        True, NACT, 0, 0, 1, "cpu",
    ]
    comptime GPU_AG = Dreamer4Agent[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX,
        NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
        True, NACT, 0, 0, 1, "gpu",
    ]
    var ac = CPU_AG.make["cpu", Deterministic](None)
    var ag = GPU_AG.make["cpu", Deterministic](Optional(c))

    # identical synthetic inputs (mirror the storage smoke)
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

    # zero grads (whole-agent on CPU; submodules on the GPU agent)
    ac.zero_grad["cpu"](None)
    ag.dyn.zero_grad["gpu"](Optional(c))
    ag.ph.zero_grad["cpu"](None)
    ag.rh.zero_grad["cpu"](None)
    ag.te.zero_grad["cpu"](None)

    var lc = ac.acwm_train_step(
        _mao(z1.unsafe_ptr()), _mao(z0.unsafe_ptr()),
        _mao(sigma.unsafe_ptr()), _mao(sigma_idx.unsafe_ptr()),
        _mao(step_idx.unsafe_ptr()), True,
        _mao(task_ids.unsafe_ptr()), _mao(actions.unsafe_ptr()),
        _mao(rewards.unsafe_ptr()), _mao(bins.unsafe_ptr()),
    )
    var lg = ag.acwm_train_step_gpu(
        _mao(z1.unsafe_ptr()), _mao(z0.unsafe_ptr()),
        _mao(sigma.unsafe_ptr()), _mao(sigma_idx.unsafe_ptr()),
        _mao(step_idx.unsafe_ptr()), True,
        _mao(task_ids.unsafe_ptr()), _mao(actions.unsafe_ptr()),
        _mao(rewards.unsafe_ptr()), _mao(bins.unsafe_ptr()), c,
    )

    var dlv = abs(lc[0] - lg[0])
    var dlb = abs(lc[1] - lg[1])
    print("  video loss  cpu=", lc[0], " gpu=", lg[0], " Δ=", dlv)
    print("  bc loss     cpu=", lc[1], " gpu=", lg[1], " Δ=", dlb)

    var vc = _GradCollect()
    ac.dyn.for_each_param["cpu"](vc, None)
    ac.ph.for_each_param["cpu"](vc, None)
    ac.rh.for_each_param["cpu"](vc, None)
    ac.te.for_each_param["cpu"](vc, None)

    var vg = _GradCollect()
    ag.dyn.for_each_param["gpu"](vg, Optional(c))
    ag.ph.for_each_param["cpu"](vg, None)
    ag.rh.for_each_param["cpu"](vg, None)
    ag.te.for_each_param["cpu"](vg, None)

    var same_len = len(vc.vals) == len(vg.vals)
    var maxd: Float64 = 0.0
    if same_len:
        for i in range(len(vc.vals)):
            var d = abs(vc.vals[i] - vg.vals[i])
            if d > maxd:
                maxd = d
    print("  params compared:", len(vc.vals), " (gpu:", len(vg.vals), ")")
    print("  max |Δ grad| (dyn + heads + te) =", maxd)

    var ok = same_len and (dlv < 1.0e-4) and (dlb < 1.0e-4) and (maxd < 2.0e-3)
    print("  CPU↔GPU acwm parity:", "OK" if ok else "FAIL")
    assert_true(ok, "acwm_train_step CPU/GPU parity")
    print("DREAMER4 ACWM GPU PARITY OK")
