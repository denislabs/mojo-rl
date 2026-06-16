"""MuZero pixel representation (Nature-CNN `MZRepNetCNN`) — build + CPU
forward/backward shape & finiteness, no GPU.

Phase 0 of `docs/MUZERO_PIXEL_PONG_PLAN.md`: verifies the new CNN representation
torso and `MuZeroCNNConfig` are contract-identical to the MLP path once the
observation is encoded —

  * ``Cfg.OBS == FRAMES·84·84`` (derived; the driver reads ``Cfg.OBS``),
  * ``Rep.IN_DIMS[0] == OBS`` / ``Rep.OUT_DIM == LATENT`` (so `MZRepGPU` and the
    unroll wrap it verbatim, exactly like the MLP rep),
  * ``Dyn`` / ``Pred`` contracts unchanged (the learned model is identical in
    latent space — only ``Rep`` differs),
  * forward over a ``[B, FRAMES·84·84]`` flat pixel batch is finite and the
    latent is min-max scaled to [0,1] (the `MinMaxNorm` tail every rep ends in),
  * the conv→mish→linear→norm backward chain runs and produces finite grads.

Run:
    pixi run mojo run -I . tests/deep_agents/test_mz_rep_cnn_smoke.mojo
"""

from std.memory import alloc
from std.testing import assert_equal, assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.muzero.config import MuZeroCNNConfig


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def main() raises:
    comptime FRAMES = 4       # stacked grayscale frames (Pong pixel)
    comptime ACT = 3          # NOOP / UP / DOWN
    comptime LATENT = 64
    comptime HIDDEN = 128     # Nature-CNN projection width (small for the smoke)
    comptime BINS = 51
    comptime B = 2            # tiny batch: CPU conv on 84×84 is the cost

    comptime Cfg = MuZeroCNNConfig[FRAMES, ACT, LATENT, HIDDEN, BINS]
    comptime OBS = Cfg.OBS

    comptime Rep = Cfg.Rep
    comptime Dyn = Cfg.Dyn
    comptime Pred = Cfg.Pred

    # ── derived OBS + contracts vs the GPU model traits ──
    assert_equal(OBS, FRAMES * 84 * 84, "Cfg.OBS == FRAMES·84·84")
    assert_equal(Rep.IN_DIMS[0], OBS, "rep IN == OBS (flat pixel vector)")
    assert_equal(Rep.OUT_DIM, LATENT, "rep OUT == LATENT")
    assert_equal(Dyn.IN_DIMS[0], LATENT + ACT, "dyn IN (z+onehot a)")
    assert_equal(Dyn.OUT_DIM, LATENT + BINS, "dyn OUT (z'|reward)")
    assert_equal(Pred.IN_DIMS[0], LATENT, "pred IN")
    assert_equal(Pred.OUT_DIM, ACT + BINS, "pred OUT (policy|value)")
    print("contracts: OK  (OBS =", OBS, ")")

    var rep = Rep.make["cpu", INIT=Kaiming]()

    # ── h: flat pixel obs [B, OBS] → z [B, LATENT] ──
    var obs = _alloc(B * OBS)
    # k/255-like pixel values in [0,1] (a cheap deterministic ramp).
    for i in range(B * OBS):
        obs[i] = Scalar[DT](Float64(i % 256) / 255.0)
    var z0 = _alloc(B * LATENT)
    var obs_t = TileTensor(obs, row_major[B, OBS]())
    var z0_t = TileTensor(z0, row_major[B, LATENT]())
    rep.forward["cpu", B](obs_t, output=z0_t)

    var fin = True
    var in01 = True
    for i in range(B * LATENT):
        var v = Float64(z0[i])
        if not (v == v) or v > 1e30 or v < -1e30:
            fin = False
        if v < -1e-4 or v > 1.0 + 1e-4:
            in01 = False
    assert_true(fin, "rep-cnn non-finite")
    assert_true(in01, "rep-cnn latent not min-max scaled to [0,1]")
    print("h (rep-cnn) forward finite + latent in [0,1]: OK")

    # ── backward: the conv→mish→linear→norm chain produces finite grads ──
    var go = _alloc(B * LATENT)
    for i in range(B * LATENT):
        go[i] = Scalar[DT](0.01) * Scalar[DT]((i % 5) - 2)
    var gi = _alloc(B * OBS)
    var go_t = TileTensor(go, row_major[B, LATENT]())
    var gi_t = TileTensor(gi, row_major[B, OBS]())
    rep.zero_grad["cpu"]()
    rep.vjp["cpu", B](go_t, gi_t)
    var gfin = True
    var nonzero = 0
    for i in range(B * OBS):
        var v = Float64(gi[i])
        if not (v == v) or v > 1e30 or v < -1e30:
            gfin = False
        if v != 0.0:
            nonzero += 1
    assert_true(gfin, "rep-cnn grad_input non-finite")
    assert_true(nonzero > 0, "rep-cnn backward produced all-zero grad_input")
    print("h (rep-cnn) backward finite, nonzero gi lanes =", nonzero)

    obs.free(); z0.free(); go.free(); gi.free()
    print("MuZero pixel rep (MZRepNetCNN) smoke: OK")
