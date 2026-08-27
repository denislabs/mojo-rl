"""CPU smoke for DreamerV3 on PIXEL observations (P2 of the pixel arc).

Builds a tiny DreamerV3Trainer/Agent whose encoder/decoder are the CNN nets
(DreamerEncoderCNN / DreamerDecoderCNN) instead of the MLP defaults, with
OBS = C*H*W (a flat image). Records synthetic image frames, runs a few WM+AC
train steps, and asserts the losses are finite + nonzero — i.e. the conv
encoder + transposed-conv decoder train end-to-end through the world model and
the reconstruction loss flows. Then select_action on an image obs.

This is the P2 gate: image obs threaded through the trainer/agent via the
ENC/DEC Module-type params (MLP path stays the default, covered by
test_dreamerv3_trainer_cpu_smoke).

Run: pixi run mojo run -I . tests/nn/test_dreamerv3_pixel_smoke.mojo
"""

from std.math import isfinite
from std.random import random_float64
from std.memory import alloc
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.dreamerv3.trainer import DreamerV3Trainer
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.deep_agents.dreamerv3.nets_cnn import (
    DreamerEncoderCNN,
    DreamerDecoderCNN,
)

# Small image: 3 channels, 16x16 (→ minres 1 over the 4-layer stride-2 stack).
comptime C = 3
comptime IMG = 16
comptime BASE = 4
comptime OBS = C * IMG * IMG          # 768 (flat image obs)
comptime ACT = 1
comptime DETER = 8
comptime H = 8
comptime STOCH = 3
comptime CLASSES = 4
comptime BLOCKS = 2
comptime TOKEN = 8
comptime DEC_U = 8                    # unused by the CNN decoder (BASE drives it)
comptime HU = 8
comptime VU = 8
comptime PU = 8
comptime BINS = 7
comptime B = 4
comptime T = 3
comptime T_IMAG = 3
comptime CAP = 2000

comptime FEATIN = STOCH * CLASSES + DETER
comptime ENC = DreamerEncoderCNN[C, IMG, IMG, BASE, TOKEN, SwishOp]
comptime DEC = DreamerDecoderCNN[FEATIN, C, IMG, IMG, BASE, SwishOp]

comptime TrainerT = DreamerV3Trainer[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU, PU,
    BINS, B, T, T_IMAG, CAP, False, ENC, DEC,
]
comptime AgentT = DreamerV3Agent[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU, PU,
    BINS, B, T, T_IMAG, CAP, False, ENC, DEC,
]
comptime TrainerGpuT = DreamerV3Trainer[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU, PU,
    BINS, B, T, T_IMAG, CAP, False, ENC, DEC,
]


def test_pixel_trainer_train_step() raises:
    var tr = TrainerT.make(learning_starts=8)
    var obs = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
    var act = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()
    for s in range(64):
        for i in range(OBS):
            obs[i] = Scalar[DT](random_float64())  # pixels in [0,1]
        for j in range(ACT):
            act[j] = Scalar[DT](random_float64() * 2.0 - 1.0)
        var rew = Scalar[DT](random_float64() - 0.5)
        var done = Scalar[DT](1.0) if (s % 17 == 16) else Scalar[DT](0.0)
        tr.record(
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](obs),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](act),
            rew, done,
        )
    if not tr.can_train():
        raise Error("pixel trainer cannot train after 64 transitions")

    var saw_wm = False
    var saw_ac = False
    for _ in range(5):
        var did = tr.train_step()
        if not did:
            raise Error("train_step returned False after can_train()")
        var wl = tr.last_wm_loss()
        var al = tr.last_ac_loss()
        if not isfinite(Float64(wl)):
            raise Error("pixel wm_loss not finite")
        if not isfinite(Float64(al)):
            raise Error("pixel ac_loss not finite")
        if wl != Scalar[DT](0.0):
            saw_wm = True
        if al != Scalar[DT](0.0):
            saw_ac = True
    if not saw_wm:
        raise Error("pixel wm_loss stayed exactly zero across 5 steps")
    if not saw_ac:
        raise Error("pixel ac_loss stayed exactly zero across 5 steps")
    obs.free()
    act.free()
    print("test_pixel_trainer_train_step: OK (conv WM trains; recon flows)")


def test_pixel_agent_select_action() raises:
    var ag = AgentT.make(learning_starts=8)
    var obs = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
    var out_action = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()
    for i in range(OBS):
        obs[i] = Scalar[DT](random_float64())
    for j in range(ACT):
        out_action[j] = Scalar[DT](0.0)
    ag.select_action(
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](obs),
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](out_action),
        explore=True,
    )
    for j in range(ACT):
        if not isfinite(Float64(out_action[j])):
            raise Error("pixel select_action produced a non-finite action")
    obs.free()
    out_action.free()
    print("test_pixel_agent_select_action: OK (conv encoder acts)")


def test_pixel_trainer_gpu(ctx: DeviceContext) raises:
    var tr = TrainerGpuT.make(ctx=ctx, lr=Scalar[DT](3e-3), learning_starts=0)
    var ob = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
    var ac = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()
    var s = UInt64(98765)
    for _t in range(80):
        for k in range(OBS):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            ob[k] = Scalar[DT](Float64((s >> 33)) / Float64(UInt64(1) << 31))
        for k in range(ACT):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            ac[k] = Scalar[DT](Float64((s >> 33)) / Float64(UInt64(1) << 31) - 1.0)
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT](Float64((s >> 33)) / Float64(UInt64(1) << 31) - 1.0)
        tr.record(ob.as_unsafe_any_origin(), ac, r, Scalar[DT](0.0))
    ob.free()
    ac.free()
    if not tr.can_train():
        raise Error("pixel GPU trainer cannot train")
    var first_wm: Scalar[DT] = 0.0
    var last_wm: Scalar[DT] = 0.0
    comptime ITERS = 20
    for it in range(ITERS):
        var ok = tr.train_step()
        if not ok:
            raise Error("pixel GPU train_step returned False")
        var wm = tr.last_wm_loss()
        if not isfinite(Float64(wm)):
            raise Error("pixel GPU wm_loss not finite")
        if it == 0:
            first_wm = wm
        last_wm = wm
    print("  pixel GPU WM:", first_wm, "->", last_wm)
    if not (last_wm < first_wm):
        raise Error("pixel GPU WM loss did not decrease")
    print("test_pixel_trainer_gpu: OK (conv WM trains on Metal, WM↓)")


def main() raises:
    print("DreamerV3 PIXEL (CNN enc/dec) smoke — OBS =", OBS, "(", C, "x",
          IMG, "x", IMG, ")")
    test_pixel_trainer_train_step()
    test_pixel_agent_select_action()
    print("CPU PIXEL GATES PASSED")
    try:
        var ctx = DeviceContext()
        test_pixel_trainer_gpu(ctx)
        print("GPU PIXEL GATE PASSED")
    except e:
        print("GPU pixel smoke SKIPPED (no device):", e)
    print("ALL DREAMERV3 PIXEL SMOKE TESTS PASSED")
