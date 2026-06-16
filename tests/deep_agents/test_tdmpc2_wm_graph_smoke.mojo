"""Smoke test: TD-MPC2 world-model ComputeGraph forward compiles + runs (CPU).

Validates the wave-1 plumbing (losses + nets + wm_graph): build the graph at
small dims, set the 5 inputs, run forward, assert the [B, 7+LATENT] output is
finite. No BPTT yet (that's test_tdmpc2_wm_bptt).
"""

from std.memory import alloc
from std.math import isfinite
from std.random import random_float64, seed
from std.testing import assert_true, TestSuite
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.tdmpc2.nets import (
    TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet,
)
from mojo_rl.deep_agents.tdmpc2.wm_graph import TDMPC2WMGraph


comptime LATENT = 16
comptime ACT = 3
comptime MLP = 16
comptime BINS = 11
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime BATCH = 4


def _fill_rand(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int):
    for i in range(n):
        p[i] = Scalar[DT](random_float64() * 2.0 - 1.0)


def test_wm_graph_forward_cpu() raises:
    seed(0)
    comptime DynT = TDMPC2Dynamics[LATENT, ACT, MLP, SN]
    comptime RewT = TDMPC2Reward[LATENT, ACT, MLP, BINS]
    comptime QNetT = TDMPC2QNet[LATENT, ACT, MLP, BINS]
    comptime GraphT = TDMPC2WMGraph[LATENT, ACT, MLP, BINS, SN, VMIN, VMAX]
    var g = GraphT.make["cpu", INIT=Kaiming]()

    # Bind the external dynamics / reward / Q heads.
    var dyn = DynT.make["cpu", INIT=Kaiming]()
    var rew_net = RewT.make["cpu", INIT=Kaiming]()
    var q0 = QNetT.make["cpu", INIT=Kaiming]()
    var q1 = QNetT.make["cpu", INIT=Kaiming]()
    var q2 = QNetT.make["cpu", INIT=Kaiming]()
    var q3 = QNetT.make["cpu", INIT=Kaiming]()
    var q4 = QNetT.make["cpu", INIT=Kaiming]()
    g.set_external["znext", DynT](dyn)
    g.set_external["rlog", RewT](rew_net)
    g.set_external["q0", QNetT](q0)
    g.set_external["q1", QNetT](q1)
    g.set_external["q2", QNetT](q2)
    g.set_external["q3", QNetT](q3)
    g.set_external["q4", QNetT](q4)

    var z = _alloc_fill(BATCH * LATENT)
    var a = _alloc_fill(BATCH * ACT)
    var zen = _alloc_fill(BATCH * LATENT)
    var r = _alloc_fill(BATCH)
    var td = _alloc_fill(BATCH)
    var out = alloc[Scalar[DT]](BATCH * (7 + LATENT))

    g.set_input["z", BATCH](TileTensor(z, row_major[BATCH, LATENT]()))
    g.set_input["a", BATCH](TileTensor(a, row_major[BATCH, ACT]()))
    g.set_input["z_enc_next", BATCH](TileTensor(zen, row_major[BATCH, LATENT]()))
    g.set_input["r", BATCH](TileTensor(r, row_major[BATCH, 1]()))
    g.set_input["td", BATCH](TileTensor(td, row_major[BATCH, 1]()))

    var out_t = TileTensor(out, row_major[BATCH, 7 + LATENT]())
    g.forward["cpu", BATCH](out_t)

    var all_finite = True
    for i in range(BATCH * (7 + LATENT)):
        if not isfinite(out[i]):
            all_finite = False
    assert_true(all_finite, "WM graph forward output must be finite")

    # Loss columns (0..6) should be non-negative (MSE + two-hot CE).
    var losses_nonneg = True
    for b in range(BATCH):
        for c in range(7):
            if out[b * (7 + LATENT) + c] < Scalar[DT](-1e-5):
                losses_nonneg = False
    assert_true(losses_nonneg, "loss columns must be >= 0")

    z.free(); a.free(); zen.free(); r.free(); td.free(); out.free()


def _alloc_fill(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    var p = alloc[Scalar[DT]](n)
    _fill_rand(p, n)
    return p


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
