"""MuZero h/g/f nets — build + CPU forward shape/finiteness, no GPU.

Checks the three torsos expose the exact dims the planner's GPU model traits
require, produce finite forward output through the CPU path, and that the
latent-producing nets (rep, and the dynamics latent split) emit min-max-scaled
latents in [0,1] (the MinMaxNorm tail).

Run:
    pixi run mojo run -I . tests/deep_agents/test_mz_nets_smoke.mojo
"""

from std.memory import alloc
from std.testing import assert_equal, assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.muzero.nets import MZRepNet, MZDynNet, MZPredNet


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def main() raises:
    comptime OBS = 4       # CartPole
    comptime ACT = 2
    comptime LATENT = 16
    comptime BINS = 51
    comptime H = 32
    comptime B = 8

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    # ── Contracts vs RepresentationGPU / DynamicsGPU / PredictionGPU ──
    assert_equal(Rep.IN_DIMS[0], OBS, "rep IN")
    assert_equal(Rep.OUT_DIM, LATENT, "rep OUT")
    assert_equal(Dyn.IN_DIMS[0], LATENT + ACT, "dyn IN (z+onehot a)")
    assert_equal(Dyn.OUT_DIM, LATENT + BINS, "dyn OUT (z'|reward)")
    assert_equal(Pred.IN_DIMS[0], LATENT, "pred IN")
    assert_equal(Pred.OUT_DIM, ACT + BINS, "pred OUT (policy|value)")
    print("contracts: OK")

    var rep = Rep.make["cpu", INIT=Kaiming]()
    var dyn = Dyn.make["cpu", INIT=Kaiming]()
    var pred = Pred.make["cpu", INIT=Kaiming]()

    # ── h: obs → z ──
    var obs = _alloc(B * OBS)
    for i in range(B * OBS):
        obs[i] = Scalar[DT](0.1) * Scalar[DT](i % 7) - Scalar[DT](0.3)
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
    assert_true(fin, "rep non-finite")
    assert_true(in01, "rep latent not min-max scaled to [0,1]")
    print("h (rep) forward finite + latent in [0,1]: OK")

    # ── g: [z ⊕ onehot(a)] → [z' | reward_logits] ──
    var dyn_in = _alloc(B * (LATENT + ACT))
    for b in range(B):
        for i in range(LATENT):
            dyn_in[b * (LATENT + ACT) + i] = z0[b * LATENT + i]
        # one-hot action: alternate a=0 / a=1 per row.
        for a in range(ACT):
            dyn_in[b * (LATENT + ACT) + LATENT + a] = Scalar[DT](0.0)
        dyn_in[b * (LATENT + ACT) + LATENT + (b % ACT)] = Scalar[DT](1.0)
    var dyn_out = _alloc(B * (LATENT + BINS))
    var din_t = TileTensor(dyn_in, row_major[B, LATENT + ACT]())
    var dout_t = TileTensor(dyn_out, row_major[B, LATENT + BINS]())
    dyn.forward["cpu", B](din_t, output=dout_t)
    fin = True
    in01 = True
    for b in range(B):
        var base = b * (LATENT + BINS)
        for i in range(LATENT + BINS):
            var v = Float64(dyn_out[base + i])
            if not (v == v) or v > 1e30 or v < -1e30:
                fin = False
        # latent split (first LATENT) must be min-max scaled; reward logits raw.
        for i in range(LATENT):
            var lv = Float64(dyn_out[base + i])
            if lv < -1e-4 or lv > 1.0 + 1e-4:
                in01 = False
    assert_true(fin, "dyn non-finite")
    assert_true(in01, "dyn next-latent not min-max scaled to [0,1]")
    print("g (dyn) forward finite + next-latent in [0,1]: OK")

    # ── f: z → [policy_logits | value_logits] ──
    var pred_out = _alloc(B * (ACT + BINS))
    var pin_t = TileTensor(z0, row_major[B, LATENT]())
    var pout_t = TileTensor(pred_out, row_major[B, ACT + BINS]())
    pred.forward["cpu", B](pin_t, output=pout_t)
    fin = True
    for i in range(B * (ACT + BINS)):
        var v = Float64(pred_out[i])
        if not (v == v) or v > 1e30 or v < -1e30:
            fin = False
    assert_true(fin, "pred non-finite")
    print("f (pred) forward finite: OK")

    obs.free(); z0.free(); dyn_in.free(); dyn_out.free(); pred_out.free()
    print("MuZero h/g/f nets smoke: OK")
