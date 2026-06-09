"""O.2.b.5 — REDQOFETrainer one-file v2 checkpoint round-trip (CPU).

Gates:
  (1) Trainer constructs end-to-end, runs a few env steps + train
      updates (so the params drift from their Xavier init).
  (2) `save_state(path)` writes a non-empty file with the v2 header.
  (3) Build a FRESH trainer (Xavier init — params are different).
  (4) `fresh.load_state(path)` restores actor + N critics + SB + AB
      + PRED + their Adams + alpha_opt.
  (5) `select_greedy_action` on a probe obs matches the original
      trainer's output within format tolerance (text v2 writes ~7
      sig figs of float precision; SAC/REDQ use 1e-4 for actions).
  (6) Re-saving the loaded trainer produces an envelope BYTE-
      IDENTICAL to the original — strongest gate on serialization
      determinism that doesn't require parsing the file.

(R.5+/checkpoint convention from `test_redq_checkpoint_cpu.mojo`.)"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU

from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.training.blocks import UniformSampleCpuStep
from mojo_rl.deep_agents2.redq.kernels import REDQ_TARGET_MIN
from mojo_rl.deep_agents2.redq_ofe import (
    OFEStateBranch6, OFEActionBranch6, OFEPredictorHead,
    REDQOFETrainer,
    state_branch_out_dim, action_branch_out_dim,
)


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 32
comptime BATCH = 32
comptime CAP = 1_024
comptime PER_UNIT = 4
comptime N_BLOCKS = 6
comptime N = 2
comptime N_MIN = 2
comptime UTD = 1
comptime POLICY_DELAY = 1

comptime PHI_S_DIM = state_branch_out_dim(OBS, N_BLOCKS, PER_UNIT)
comptime PHI_SA_DIM = action_branch_out_dim(OBS, ACT, N_BLOCKS, PER_UNIT)

comptime SB = OFEStateBranch6[OBS, PER_UNIT]
comptime AB = OFEActionBranch6[PHI_S_DIM + ACT, PER_UNIT]
comptime PRED = OFEPredictorHead[PHI_SA_DIM, OBS]
comptime ACTOR = StochasticActor[
    PHI_S_DIM, ACT,
    Linear[PHI_S_DIM, HIDDEN], ReLU[HIDDEN],
]
comptime CRITIC = Sequential[
    Linear[PHI_SA_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime SAMPLE = UniformSampleCpuStep[OBS, ACT, BATCH, CAP]
comptime Trainer = REDQOFETrainer[
    "cpu", SAMPLE, ACTOR, CRITIC, SB, AB, PRED,
    N, N_MIN, UTD, POLICY_DELAY, REDQ_TARGET_MIN,
]


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def test_redq_ofe_checkpoint_round_trip() raises:
    print("=" * 70)
    print("O.2.b.5 — REDQOFETrainer one-file v2 checkpoint round-trip (CPU)")
    print("=" * 70)
    seed(42)

    var trainer = Trainer.make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        ofe_lr=Scalar[DT](3e-4),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](2.0),
        learning_starts=80,
        window_size=4,
        initial_episode_fill=Scalar[DT](0.0),
    )

    # (1) Drive a few env steps + train updates so params drift from
    # Xavier init. Synthetic transitions — we don't need a real env
    # for the checkpoint test, just enough buffer entries to clear
    # warmup + give the trainer real gradient signal.
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    for step in range(150):
        for d in range(OBS):
            obs[d] = Scalar[DT](0.2 * Float64(d) + 0.005 * Float64(step))
        trainer.select_action(obs, act, step)
        for d in range(OBS):
            nxt[d] = Scalar[DT](
                0.2 * Float64(d) + 0.005 * Float64(step + 1)
            )
        var rew = Scalar[DT](-0.3 + 0.2 * Float64(act[0]))
        var done = Scalar[DT](0.0) if step % 25 != 24 else Scalar[DT](1.0)
        trainer.record(obs, act, rew, nxt, done)
        if done == Scalar[DT](1.0):
            trainer.end_episode()
        _ = trainer.train_step(step)

    print("  total_train_steps after 150 env steps =",
          trainer.total_train_steps())

    # (2) Save.
    var path = String("/tmp/redq_ofe_ckpt.txt")
    trainer.save_state(path)
    var content_a: String
    with open(path, "r") as f:
        content_a = f.read()
    assert_true(
        content_a.byte_length() > 200,
        "checkpoint file must be non-trivial size",
    )
    assert_true(
        content_a.startswith("nn2-ckpt v2\n"),
        "checkpoint must start with the v2 header",
    )
    print("  saved checkpoint:", content_a.byte_length(), "bytes")

    # (3) Capture probe-obs greedy action from ORIGINAL trainer.
    var probe = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    probe[0] = Scalar[DT](0.7)
    probe[1] = Scalar[DT](-0.3)
    probe[2] = Scalar[DT](1.5)
    var act_orig = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    trainer.select_greedy_action(probe, act_orig)
    print("  greedy[orig]   =", act_orig[0])

    # (4) Build FRESH trainer (different Xavier seed → different
    # params). Then load and verify the greedy action matches.
    seed(7)
    var fresh = Trainer.make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        ofe_lr=Scalar[DT](3e-4),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](2.0),
        learning_starts=80,
        window_size=4,
        initial_episode_fill=Scalar[DT](0.0),
    )
    var act_fresh_pre = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    fresh.select_greedy_action(probe, act_fresh_pre)
    print("  greedy[fresh pre-load] =", act_fresh_pre[0])
    # Pre-load the fresh trainer should NOT match the original (sanity
    # check that the gate below isn't trivially true).
    assert_true(
        _abs(act_fresh_pre[0] - act_orig[0]) > Scalar[DT](1e-4),
        "fresh trainer (pre-load) should differ from original — gate sanity",
    )

    # Sanity: fresh trainer ran no train steps, so its cumulative counter
    # is 0 before load (gate that the round-trip assertion below is real).
    assert_true(
        fresh.total_train_steps() == 0,
        "fresh trainer should have 0 train steps before load",
    )

    fresh.load_state(path)
    # M4: cumulative _total_train_steps must survive save/resume (PER
    # β-anneal schedules key on it). The byte-identical gate (6) below
    # also covers this, but assert it directly for a clear failure.
    assert_true(
        fresh.total_train_steps() == trainer.total_train_steps(),
        "loaded trainer must restore the cumulative _total_train_steps",
    )
    var act_loaded = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    fresh.select_greedy_action(probe, act_loaded)
    print("  greedy[loaded] =", act_loaded[0])
    var dev = _abs(act_loaded[0] - act_orig[0])
    print("  |act_loaded - act_orig| =", dev)

    # (5) Tol-based: v2 text writes float as String(...) which keeps
    # ~7 sig figs; SAC/REDQ continuous-action convention is 1e-4.
    assert_true(
        dev < Scalar[DT](1e-4),
        "loaded trainer must reproduce greedy action within text-format tol",
    )

    # (6) Re-save the LOADED trainer → byte-identical envelope.
    var path_b = String("/tmp/redq_ofe_ckpt_b.txt")
    fresh.save_state(path_b)
    var content_b: String
    with open(path_b, "r") as f:
        content_b = f.read()
    assert_true(
        content_a.byte_length() == content_b.byte_length(),
        "re-saved checkpoint length must match the original",
    )
    assert_true(
        content_a == content_b,
        "re-saved checkpoint must be BYTE-IDENTICAL to the original",
    )
    print("PASS — checkpoint round-trip + re-save byte-identical.")


def main() raises:
    test_redq_ofe_checkpoint_round_trip()
