"""EZv2 CartPole — SimSiam representation-collapse diagnostic (CPU).

Runs the EZv2 CPU self-play loop TWICE with everything identical except the
consistency weight — once with the SimSiam objective ON (``cons 2.0``) and once
OFF (``cons 0.0``) — printing a per-batch collapse metric every ``DIAG`` steps:

  * ``latent_std``     — mean per-dim std of ``z = h(obs0)`` across the batch.
  * ``proj_norm_std``  — same on the L2-normalized projector output (the standard
                         SimSiam collapse metric; → 0 = projections collapse to
                         one direction).

If consistency is COLLAPSING the shared representation, the cons-ON run's
``latent_std`` / ``proj_norm_std`` should trend toward 0 (and far below the
cons-OFF run's) as training proceeds — that would be a representation-collapse
bug. If both stay healthy yet only cons-ON plateaus on return, the consistency
objective is merely mis-suited to this state-based env (no collapse, no bug).

Short run (8k iters each) — collapse, if present, shows within a few thousand
steps. Returns nothing; read the printed ``[collapse]`` lines.

Run (no GPU):
    pixi run mojo run -I . examples/cartpole/ezv2_cartpole_collapse_diag_cpu.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.deep_agents.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, MZPredNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_cpu import (
    run_ezv2_selfplay_cpu,
)
from mojo_rl.envs.cartpole import CartPoleEnv


def _run[CONS: Float64](label: String) raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 64
    comptime BINS = 51
    comptime H = 128
    comptime PROJ = 128
    comptime PROJ_HID = 128
    comptime BOTTLENECK = 64
    comptime NUM_SIMS = 25
    comptime MAX_NODES = 128
    comptime CAP = 50000
    comptime B = 64
    comptime K = 5
    comptime N = 10

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
    comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]

    var env = CartPoleEnv[DType.float64]()
    var rep = Rep.make["cpu", Kaiming]()
    var dyn = Dyn.make["cpu", Kaiming]()
    var pred = Pred.make["cpu", Kaiming]()
    var proj = Proj.make["cpu", Kaiming]()
    var predh = Predh.make["cpu", Kaiming]()
    var orep = Adam(lr=Scalar[DT](3e-4))
    var odyn = Adam(lr=Scalar[DT](3e-4))
    var opred = Adam(lr=Scalar[DT](3e-4))
    var oproj = Adam(lr=Scalar[DT](3e-4))
    var opredh = Adam(lr=Scalar[DT](3e-4))

    print("==== EZv2 collapse diag:", label, "(cons", CONS, ") ====")
    _ = run_ezv2_selfplay_cpu[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred, Proj, Predh,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
    ](
        env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        iterations=8000,
        learning_starts=500,
        train_per_iter=1,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](0.5),
        consistency_coef=Scalar[DT](CONS),
        temperature_decay_steps=8000,
        reanalyze_every=1,
        eval_every=4000,
        eval_episodes=5,
        diag_every=1000,
        seed=42,
        verbose=True,
    )


def main() raises:
    _run[2.0]("CONS-ON")
    print()
    _run[0.0]("CONS-OFF")
