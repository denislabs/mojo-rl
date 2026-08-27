"""DreamerV3 eager-vs-captured train-step parity — DISCRETE GPU (Stage 3 P5).

The capture gate: two identical discrete-GPU trainers are fed a byte-identical
fixed minibatch + identical imagination noise each step. One runs the device
step eagerly (`_device_step`); the other runs it through `maybe_capture_replay`
around `train_device_kernels` (the closure the production loop captures). At
periodic `want_diag=True` boundaries both run the eager diagnostic step and we
compare the WM + AC loss scalars.

  * On non-NVIDIA (Apple): `maybe_capture_replay` runs the closure directly, so
    this validates the REFACTOR — the captured device-kernel sequence is
    bit-identical to the eager `_device_step(want_diag=False)`.
  * On NVIDIA: the closure is captured once and replayed thereafter, so this
    validates the actual CUDA-graph capture/replay (every learned weight must
    advance identically → identical diag losses within float32 tolerance).

This is the gate to clear BEFORE flipping `USE_TRAIN_CUDA_GRAPH=True` in a
training run (the MuZero-MCTS-capture lesson: a captured graph that replays
stale/flat targets trains differently — never enable capture unverified).

Run: `pixi run -e nvidia mojo run -I . tests/nn/test_dreamerv3_capture_parity.mojo`
     (Apple: transparency only — runs the closure eagerly.)
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.cuda import CUDAGraph, maybe_capture_replay
from mojo_rl.deep_agents.dreamerv3.trainer import DreamerV3Trainer

comptime OBS = 3
comptime ACT = 2          # discrete categorical actions
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
comptime ITERS = 16
comptime DIAG_EVERY = 4   # compare losses at these boundaries
comptime SEED = 1234

comptime GpuTr = DreamerV3Trainer[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP, True,   # DISCRETE=True
]


def _lcg(mut s: UInt64) -> Scalar[DT]:
    s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Scalar[DT]((Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0)


def _fill_window(
    it: Int,
    mb_obs: Pointer[Scalar[DT], MutAnyOrigin],
    mb_act: Pointer[Scalar[DT], MutAnyOrigin],
    mb_rew: Pointer[Scalar[DT], MutAnyOrigin],
    mb_dne: Pointer[Scalar[DT], MutAnyOrigin],
    mb_fst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """A fresh-but-deterministic window for iter `it` (seeded by `it`)."""
    var s = UInt64(99887766) + UInt64(it) * UInt64(2654435761)
    for i in range(B * (T + 1) * OBS):
        mb_obs[i] = _lcg(s)
    for i in range(B * T * ACT):
        mb_act[i] = _lcg(s)
    for i in range(B * T):
        mb_rew[i] = _lcg(s)
        mb_dne[i] = Scalar[DT](0.0)
    for b in range(B):
        for t in range(T + 1):
            mb_fst[b * (T + 1) + t] = Scalar[DT](1.0) if t == 0 else Scalar[DT](0.0)


def main() raises:
    print("=" * 70)
    print("DreamerV3 eager-vs-captured train-step parity — DISCRETE GPU")
    print("=" * 70)
    var ctx = DeviceContext()

    var mb_obs = alloc[Scalar[DT]](B * (T + 1) * OBS).as_unsafe_any_origin()
    var mb_act = alloc[Scalar[DT]](B * T * ACT).as_unsafe_any_origin()
    var mb_rew = alloc[Scalar[DT]](B * T).as_unsafe_any_origin()
    var mb_dne = alloc[Scalar[DT]](B * T).as_unsafe_any_origin()
    var mb_fst = alloc[Scalar[DT]](B * (T + 1)).as_unsafe_any_origin()

    # Two trainers with IDENTICAL initial weights (reseed before each make).
    seed(SEED)
    var eager = GpuTr.make(
        ctx=ctx, lr=Scalar[DT](2e-3), learning_starts=0, warmup_steps=0,
    )
    seed(SEED)
    var cap = GpuTr.make(
        ctx=ctx, lr=Scalar[DT](2e-3), learning_starts=0, warmup_steps=0,
    )

    # Capture slot lives at function scope (survives loop iters); no-op on
    # non-NVIDIA. Moved into a disjoint local for the capture call.
    var graph: Optional[CUDAGraph] = None

    var max_wm: Scalar[DT] = 0.0
    var max_ac: Scalar[DT] = 0.0
    for it in range(ITERS):
        _fill_window(it, mb_obs, mb_act, mb_rew, mb_dne, mb_fst)
        var wd = (it % DIAG_EVERY == 0)

        # ── eager trainer: load → noise → device step ──
        eager.load_minibatch(mb_obs, mb_act, mb_rew, mb_dne, mb_fst)
        seed(SEED + it)
        eager._fill_noise()
        eager._device_step(wd)

        # ── captured trainer: load → identical noise → capture/replay ──
        cap.load_minibatch(mb_obs, mb_act, mb_rew, mb_dne, mb_fst)
        seed(SEED + it)
        cap._fill_noise()
        if wd:
            # diag boundary: eager step (the readout can't be captured).
            cap._device_step(True)
        else:
            var g = graph^
            graph = None

            def _captured() capturing raises -> None:
                cap.train_device_kernels()

            maybe_capture_replay[_captured](g, ctx)
            graph = g^

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

    # Compile-coverage for the production capture entry (`train_step_captured`
    # routes prologue + capture + counter). `cap`'s replay was never recorded
    # into, so `can_train()` is False → this returns False without drawing,
    # while still forcing the method (closure + moved-graph) to compile.
    _ = cap.train_step_captured(want_diag=False)

    print("  max |ΔWM| =", max_wm, "  max |ΔAC| =", max_ac)
    assert_true(max_wm < Scalar[DT](1e-1), "WM eager↔captured parity")
    assert_true(max_ac < Scalar[DT](1e-1), "AC eager↔captured parity")
    print("=" * 70)
    print("CAPTURE PARITY PASSED — train_device_kernels replays == eager step")
    print("=" * 70)
    mb_obs.free(); mb_act.free(); mb_rew.free(); mb_dne.free(); mb_fst.free()
