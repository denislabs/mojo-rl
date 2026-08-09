"""DreamerV3 CPU↔GPU train-step parity — DISCRETE (categorical) actor.

Discrete counterpart of `test_dreamerv3_ac_parity.mojo`. Both trainers are fed
a BYTE-IDENTICAL fixed minibatch (via `load_minibatch`) and identical
pre-seeded imagination noise, then run `_run_minibatch` per iter; we compare
per-step WM + AC losses. `DISCRETE=True` so the unimix-categorical actor path
is exercised. This is the validation gate for the device-resident discrete
`_ac_gpu` rewrite (λ-return / imag-loss / sampling on-device).

NOTE: the minibatch is loaded directly (not drawn from the replay). As of
Stage 3 P2b the two backends use genuinely different samplers — the CPU host
`SequenceReplay` vs the GPU device-Philox `GPUSequenceReplay` — so they no
longer draw the same window from the same host seed. The replay backends are
gated separately by `tests/nn/dreamerv3/test_gpu_sequence_replay.mojo`; this
test isolates the WM/AC MATH parity by feeding identical inputs.

Run: `pixi run -e apple mojo run -I . tests/nn/test_dreamerv3_ac_parity_discrete.mojo`
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamerv3.trainer import DreamerV3Trainer

comptime OBS = 3
comptime ACT = 2          # discrete: number of categorical actions (C)
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
    PU, BINS, B, T, T_IMAG, CAP, True,   # DISCRETE=True
]
comptime GpuTr = DreamerV3Trainer[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP, True,   # DISCRETE=True
]


def _lcg(mut s: UInt64) -> Scalar[DT]:
    """Deterministic [-1, 1) sample from a 64-bit LCG (host-only, no global
    RNG dependence) so the synthetic minibatch is reproducible."""
    s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Scalar[DT]((Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0)


def main() raises:
    print("=" * 70)
    print("DreamerV3 CPU↔GPU train-step parity — DISCRETE")
    print("=" * 70)
    var ctx = DeviceContext()

    # ── one fixed synthetic minibatch, shared by both backends ──
    var mb_obs = alloc[Scalar[DT]](B * (T + 1) * OBS).as_unsafe_any_origin()
    var mb_act = alloc[Scalar[DT]](B * T * ACT).as_unsafe_any_origin()
    var mb_rew = alloc[Scalar[DT]](B * T).as_unsafe_any_origin()
    var mb_dne = alloc[Scalar[DT]](B * T).as_unsafe_any_origin()
    var mb_fst = alloc[Scalar[DT]](B * (T + 1)).as_unsafe_any_origin()
    var s = UInt64(99887766)
    for i in range(B * (T + 1) * OBS):
        mb_obs[i] = _lcg(s)
    for i in range(B * T * ACT):
        mb_act[i] = _lcg(s)
    for i in range(B * T):
        mb_rew[i] = _lcg(s)
        mb_dne[i] = Scalar[DT](0.0)   # episodes don't terminate in-window
    # each window starts a fresh episode (first obs frame is_first=1).
    for b in range(B):
        for t in range(T + 1):
            mb_fst[b * (T + 1) + t] = Scalar[DT](1.0) if t == 0 else Scalar[DT](0.0)

    var wm_cpu = alloc[Scalar[DT]](ITERS).as_unsafe_any_origin()
    var ac_cpu = alloc[Scalar[DT]](ITERS).as_unsafe_any_origin()
    var wm_gpu = alloc[Scalar[DT]](ITERS).as_unsafe_any_origin()
    var ac_gpu = alloc[Scalar[DT]](ITERS).as_unsafe_any_origin()

    # reseed before EACH make so both backends draw IDENTICAL initial weights.
    seed(SEED)
    var cpu = CpuTr.make(lr=Scalar[DT](2e-3), learning_starts=0, warmup_steps=0)
    seed(SEED)
    # device_noise=False → host-seeded noise (uploaded) so the GPU reads the
    # SAME noise as the CPU (production uses on-device Philox; that path can't be
    # bit-matched against host RNG and is gated by the capture-parity test).
    var gpu = GpuTr.make(
        ctx=ctx, lr=Scalar[DT](2e-3), learning_starts=0, warmup_steps=0,
        device_noise=False,
    )

    for it in range(ITERS):
        cpu.load_minibatch(mb_obs, mb_act, mb_rew, mb_dne, mb_fst)
        gpu.load_minibatch(mb_obs, mb_act, mb_rew, mb_dne, mb_fst)
        # identical imagination noise both sides (reseed per iter → the noise
        # stream is the same for CPU and GPU on this step).
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
    assert_true(max_wm < Scalar[DT](1e-1), "WM CPU↔GPU parity (discrete)")
    assert_true(max_ac < Scalar[DT](1e-1), "AC CPU↔GPU parity (discrete)")
    print("=" * 70)
    print("PARITY PASSED — discrete _ac_gpu matches _ac_cpu")
    print("=" * 70)
    mb_obs.free(); mb_act.free(); mb_rew.free(); mb_dne.free(); mb_fst.free()
    wm_cpu.free(); ac_cpu.free(); wm_gpu.free(); ac_gpu.free()
