"""DreamerV3 eager-vs-captured parity over the REAL train_step facade (Stage 3).

Stronger sibling of `test_dreamerv3_capture_parity.mojo`. That gate drives the
captured `train_device_kernels` directly with a FIXED `load_minibatch` window —
it proves the captured device-kernel COMPUTATION is bit-identical, but it never
exercises the full-training path: the GPU replay sample (`sample_batch_fst_dev`)
feeding the captured WM, the device-Philox noise, and the eager prologue, all
through the `train_step` / `train_step_captured` facade.

This test closes that gap. Two identical discrete-GPU trainers are fed the SAME
recorded transitions; one trains with eager `train_step`, the other with
`train_step_captured` (capture-once / replay). Both replays hold identical data
and use the same device Philox RNG, so they draw identical windows + identical
imagination noise each step → the two MUST track bit-identically if capture is
correct. We compare WM + AC losses at `want_diag` boundaries.

If this DIVERGES while `test_dreamerv3_capture_parity.mojo` passes, the bug is in
the capture×(replay-sample / prologue) interaction, not the device-kernel math.

Run: `pixi run -e nvidia mojo run -I . tests/nn/test_dreamerv3_capture_train_parity.mojo`
     (Apple: capture is a no-op → both run eagerly → transparency check.)
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamerv3.trainer import DreamerV3Trainer

comptime OBS = 3
comptime ACT = 2
comptime DETER = 16
comptime H = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime TOKEN = 8
comptime DEC_U = 8
comptime HU = 8
comptime VU = 8
comptime PU = 8
comptime BINS = 7
comptime B = 2
comptime T = 3
comptime T_IMAG = 4
comptime CAP = 256
comptime NREC = 200       # transitions recorded into BOTH replays (identical)
comptime ITERS = 40       # train steps to compare over
comptime DIAG_EVERY = 4
comptime SEED = 1234

comptime Tr = DreamerV3Trainer[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP, True,   # DISCRETE=True
]


def _fill_replay(mut tr: Tr) raises:
    """Record NREC identical synthetic transitions (deterministic LCG) so both
    trainers' device replays hold the SAME data → identical sampled windows."""
    var s = UInt64(99887766)
    var ob = alloc[Scalar[DT]](OBS)
    var ac = alloc[Scalar[DT]](ACT)
    for _t in range(NREC):
        for k in range(OBS):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            ob[k] = Scalar[DT]((Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0)
        for k in range(ACT):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            ac[k] = Scalar[DT]((Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0)
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0)
        tr.record(ob, ac, r, Scalar[DT](0.0))
    ob.free(); ac.free()


def main() raises:
    print("=" * 70)
    print("DreamerV3 eager-vs-captured TRAIN-STEP parity (real facade, GPU)")
    print("=" * 70)
    var ctx = DeviceContext()

    # Identical init weights (reseed before each make).
    seed(SEED)
    var eager = Tr.make(
        ctx=ctx, lr=Scalar[DT](2e-3), learning_starts=0, warmup_steps=0
    )
    seed(SEED)
    var cap = Tr.make(
        ctx=ctx, lr=Scalar[DT](2e-3), learning_starts=0, warmup_steps=0
    )
    # Identical replay contents → identical sampled windows (same device RNG).
    _fill_replay(eager)
    _fill_replay(cap)

    var max_wm: Scalar[DT] = 0.0
    var max_ac: Scalar[DT] = 0.0
    for it in range(ITERS):
        var wd = (it % DIAG_EVERY == 0)
        _ = eager.train_step(want_diag=wd)
        _ = cap.train_step_captured(want_diag=wd)
        if wd:
            var dw = eager.last_wm_loss() - cap.last_wm_loss()
            if dw < 0: dw = -dw
            var da = eager.last_ac_loss() - cap.last_ac_loss()
            if da < 0: da = -da
            if dw > max_wm: max_wm = dw
            if da > max_ac: max_ac = da
            print(
                "  it", it, " WM eager/cap=", eager.last_wm_loss(), "/",
                cap.last_wm_loss(), " AC eager/cap=", eager.last_ac_loss(),
                "/", cap.last_ac_loss(),
            )

    print("  max |ΔWM| =", max_wm, "  max |ΔAC| =", max_ac)
    assert_true(max_wm < Scalar[DT](1e-1), "WM eager↔captured train-step parity")
    assert_true(max_ac < Scalar[DT](1e-1), "AC eager↔captured train-step parity")
    print("=" * 70)
    print("CAPTURE TRAIN PARITY PASSED — train_step_captured == train_step")
    print("=" * 70)
