"""DreamerV3 CPU↔GPU train-step parity (Metal) — CONTINUOUS (Gaussian) actor.

Builds a CPU trainer and a GPU trainer with IDENTICAL config, feeds both a
BYTE-IDENTICAL fixed minibatch (via `load_minibatch`) plus identical
pre-seeded imagination noise, runs N `_run_minibatch`s on each target and
compares per-step WM + AC losses.

This is the validation gate for the `_ac_gpu`→`_ac_cpu` parity port (NS=T·B
imagination starts, mean-normalized cotangents, repval value-loss). Both paths
read the SAME pre-filled `state.noise[(t*NS+b)*ACT+a]`, so the only difference
is CPU vs GPU kernel arithmetic → expect float32-level agreement.

NOTE: the minibatch is loaded directly (not drawn from the replay). As of
Stage 3 P2b the backends use genuinely different samplers (host `SequenceReplay`
vs device-Philox `GPUSequenceReplay`), so they no longer draw the same window
from the same host seed; the replays are gated separately by
`tests/nn/dreamerv3/test_gpu_sequence_replay.mojo`. This test isolates the
WM/AC MATH parity by feeding identical inputs.

Run: `pixi run -e apple mojo run -I . tests/nn/test_dreamerv3_ac_parity.mojo`
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamerv3.trainer import DreamerV3Trainer

comptime OBS = 3
comptime ACT = 1
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
comptime ITERS = 12
comptime SEED = 1234

comptime CpuTr = DreamerV3Trainer[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP,
]
comptime GpuTr = DreamerV3Trainer[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP,
]


def _lcg(mut s: UInt64) -> Scalar[DT]:
    """Deterministic [-1, 1) sample from a 64-bit LCG (host-only) so the
    synthetic minibatch is reproducible."""
    s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Scalar[DT]((Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0)


def main() raises:
    print("=" * 70)
    print("DreamerV3 CPU↔GPU train-step parity — CONTINUOUS")
    print("=" * 70)
    var ctx = DeviceContext()

    # ── one fixed synthetic minibatch, shared by both backends ──
    var mb_obs = alloc[Scalar[DT]](B * (T + 1) * OBS)
    var mb_act = alloc[Scalar[DT]](B * T * ACT)
    var mb_rew = alloc[Scalar[DT]](B * T)
    var mb_dne = alloc[Scalar[DT]](B * T)
    var mb_fst = alloc[Scalar[DT]](B * (T + 1))
    var s = UInt64(99887766)
    for i in range(B * (T + 1) * OBS):
        mb_obs[i] = _lcg(s)
    for i in range(B * T * ACT):
        mb_act[i] = _lcg(s)
    for i in range(B * T):
        mb_rew[i] = _lcg(s)
        mb_dne[i] = Scalar[DT](0.0)   # episodes don't terminate in-window
    for b in range(B):
        for t in range(T + 1):
            mb_fst[b * (T + 1) + t] = Scalar[DT](1.0) if t == 0 else Scalar[DT](0.0)

    var wm_cpu = alloc[Scalar[DT]](ITERS)
    var ac_cpu = alloc[Scalar[DT]](ITERS)
    var wm_gpu = alloc[Scalar[DT]](ITERS)
    var ac_gpu = alloc[Scalar[DT]](ITERS)

    # reseed before EACH make so both backends draw IDENTICAL initial weights.
    seed(SEED)
    var cpu = CpuTr.make(lr=Scalar[DT](2e-3), learning_starts=0, warmup_steps=0)
    seed(SEED)
    var gpu = GpuTr.make(
        ctx=ctx, lr=Scalar[DT](2e-3), learning_starts=0, warmup_steps=0
    )

    for it in range(ITERS):
        cpu.load_minibatch(mb_obs, mb_act, mb_rew, mb_dne, mb_fst)
        gpu.load_minibatch(mb_obs, mb_act, mb_rew, mb_dne, mb_fst)
        seed(SEED + it)
        cpu._run_minibatch(True)
        seed(SEED + it)
        gpu._run_minibatch(True)
        wm_cpu[it] = cpu.last_wm_loss()
        ac_cpu[it] = cpu.last_ac_loss()
        wm_gpu[it] = gpu.last_wm_loss()
        ac_gpu[it] = gpu.last_ac_loss()

    # ── compare ──
    var max_wm: Scalar[DT] = 0.0
    var max_ac: Scalar[DT] = 0.0
    for it in range(ITERS):
        var dw = wm_cpu[it] - wm_gpu[it]
        if dw < 0: dw = -dw
        var da = ac_cpu[it] - ac_gpu[it]
        if da < 0: da = -da
        if dw > max_wm: max_wm = dw
        if da > max_ac: max_ac = da
        print(
            "  it", it, " WM cpu/gpu=", wm_cpu[it], "/", wm_gpu[it],
            " AC cpu/gpu=", ac_cpu[it], "/", ac_gpu[it],
        )
    print("  max |ΔWM| =", max_wm, "  max |ΔAC| =", max_ac)
    assert_true(max_wm < Scalar[DT](1e-1), "WM CPU↔GPU parity")
    assert_true(max_ac < Scalar[DT](1e-1), "AC CPU↔GPU parity")
    print("=" * 70)
    print("PARITY PASSED — _ac_gpu matches _ac_cpu")
    print("=" * 70)
    mb_obs.free(); mb_act.free(); mb_rew.free(); mb_dne.free(); mb_fst.free()
    wm_cpu.free(); ac_cpu.free(); wm_gpu.free(); ac_gpu.free()
