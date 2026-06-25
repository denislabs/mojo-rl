"""DreamerV3 CPU↔GPU train-step parity — DISCRETE (categorical) actor.

Discrete counterpart of `test_dreamerv3_ac_parity.mojo`. Same harness (two
trainers, identical synthetic replay, identical pre-seeded imagination noise)
but `DISCRETE=True` so the unimix-categorical actor path is exercised. This is
the validation gate for the device-resident discrete `_ac_gpu` rewrite (the
λ-return / imag-loss / sampling moved on-device): before the rewrite the
host-marshalling GPU path matches CPU; after, the device-resident path must
still match CPU within float32 tolerance.

Run: `pixi run -e apple mojo run -I . tests/nn/test_dreamerv3_ac_parity_discrete.mojo`
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

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
comptime NREC = 120
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


def _fill_replay_cpu(mut tr: CpuTr):
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


def _fill_replay_gpu(mut tr: GpuTr):
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
    print("DreamerV3 CPU↔GPU train-step parity — DISCRETE")
    print("=" * 70)
    var ctx = DeviceContext()

    var wm_cpu = alloc[Scalar[DT]](ITERS)
    var ac_cpu = alloc[Scalar[DT]](ITERS)
    var wm_gpu = alloc[Scalar[DT]](ITERS)
    var ac_gpu = alloc[Scalar[DT]](ITERS)

    # ── CPU run ──
    seed(SEED)
    var cpu = CpuTr.make(lr=Scalar[DT](2e-3), learning_starts=0, warmup_steps=0)
    _fill_replay_cpu(cpu)
    seed(SEED)            # imagination-noise RNG, identical to GPU below
    for it in range(ITERS):
        _ = cpu.train_step()
        wm_cpu[it] = cpu.last_wm_loss()
        ac_cpu[it] = cpu.last_ac_loss()

    # ── GPU run ──
    seed(SEED)
    var gpu = GpuTr.make(ctx=ctx, lr=Scalar[DT](2e-3), learning_starts=0, warmup_steps=0)
    _fill_replay_gpu(gpu)
    seed(SEED)
    for it in range(ITERS):
        _ = gpu.train_step()
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
    wm_cpu.free(); ac_cpu.free(); wm_gpu.free(); ac_gpu.free()
