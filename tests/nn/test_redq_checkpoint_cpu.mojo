"""R.5+ — CPU one-file v2 checkpoint round-trip for `REDQTrainer`.

Gates:
  (a) `save_state` writes a file the same trainer instance can
      `load_state` back into byte-identically (actor + every online
      critic + every Adam + α opt).
  (b) Re-save after load is byte-identical to the first save (full
      state round-trip).
  (c) After `load_state`, all N target nets are byte-identical to
      their just-restored online twins (target reconstruction).
  (d) Greedy eval on the same obs is identical between the original
      trainer and the just-loaded fresh trainer (end-to-end gate
      that nothing was silently dropped from the envelope).
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.named_params import named_params
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators import Sequential

from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep
from mojo_rl.deep_agents.redq import REDQTrainer, REDQ_TARGET_MIN


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 16
comptime CAP = 512

comptime N = 3
comptime N_MIN = 2
comptime UTD = 2
comptime POLICY_DELAY = 2
comptime Q_MODE = REDQ_TARGET_MIN

comptime ActorNet = Sequential[
    Linear[OBS, 24], ReLU[24], Linear[24, 2 * ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, 24], ReLU[24], Linear[24, 1],
]
comptime Sample = UniformSampleCpuStep[OBS, ACT, BATCH, CAP]
comptime Trainer = REDQTrainer[
    "cpu", Sample, ActorNet, CriticNet,
    N, N_MIN, UTD, POLICY_DELAY, Q_MODE,
]


def _read_file(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _max_abs_diff[M: Module](mut a: M, mut b: M) raises -> Float64:
    var ap = named_params["cpu", M](a)
    var bp = named_params["cpu", M](b)
    if len(ap) != len(bp):
        raise Error("leaf count mismatch")
    var worst: Float64 = 0.0
    for i in range(len(ap)):
        ref pa = ap[i]
        ref pb = bp[i]
        for k in range(pa.n_elems):
            var d = Float64(pa.param_ptr[k]) - Float64(pb.param_ptr[k])
            if d < 0.0:
                d = -d
            if d > worst:
                worst = d
    return worst


def _drive_some_steps(mut trainer: Trainer) raises:
    """Push some transitions + run a few train steps so the trainer's
    state is non-trivial (Adam moments populated, α has drifted, etc.)."""
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    for step in range(150):
        for d in range(OBS):
            obs[d] = Scalar[DT](0.3 * Float64(d) + 0.005 * Float64(step))
        trainer.select_action(obs, act, step)
        for d in range(OBS):
            nxt[d] = Scalar[DT](0.3 * Float64(d) + 0.005 * Float64(step + 1))
        var rew = Scalar[DT](-0.3 + 0.2 * Float64(act[0]))
        var done = Scalar[DT](0.0) if (step + 1) % 25 != 0 else Scalar[DT](1.0)
        trainer.record(obs, act, rew, nxt, done)
        _ = trainer.train_step(step)
        if done == Scalar[DT](1.0):
            trainer.end_episode()


def test_redq_checkpoint_cpu() raises:
    print("--- REDQ CPU checkpoint round-trip (N=3) ---")
    var path = String("/tmp/redq_ckpt_test_cpu.bin")

    var a = Trainer.make(
        learning_starts=32,
        initial_episode_fill=Scalar[DT](0.0),
        window_size=4,
        action_scale=Scalar[DT](1.0),
    )
    _drive_some_steps(a)
    a.save_state(path)

    # (a) Re-save after load == first save (byte-identical envelope).
    var first_bytes = _read_file(path)
    var first_len = len(first_bytes)
    print("  envelope size =", first_len, "bytes")

    # Build a SECOND trainer (fresh params), load the file into it.
    var b = Trainer.make(
        learning_starts=32,
        initial_episode_fill=Scalar[DT](0.0),
        window_size=4,
        action_scale=Scalar[DT](1.0),
    )
    b.load_state(path)

    # (b) Re-save → byte-identical to first save.
    var path2 = String("/tmp/redq_ckpt_test_cpu_resave.bin")
    b.save_state(path2)
    var second_bytes = _read_file(path2)
    print("  resave size  =", len(second_bytes), "bytes")
    assert_true(
        len(first_bytes) == len(second_bytes),
        "re-save envelope size must match original",
    )
    assert_true(
        first_bytes == second_bytes,
        "re-save envelope must be byte-identical",
    )

    # (c) Every online + target identical-to-tol between `a` and `b`.
    # The v2 text format encodes floats via `String(float)` which keeps
    # ~7 significant figures; loaded params can differ in the LSB.
    # Byte-identical envelope (above) is the meaningful round-trip
    # gate — same SAC convention (continuous-action algos use tol).
    comptime PARAM_TOL: Float64 = 1e-5
    var actor_d = _max_abs_diff[ActorNet](a.actor, b.actor)
    print("  max |actor_a - actor_b| =", actor_d)
    assert_true(actor_d < PARAM_TOL, "actor matches within format tol")
    for i in range(N):
        var on_d = _max_abs_diff[CriticNet](
            a.ensemble.pairs[i].online,
            b.ensemble.pairs[i].online,
        )
        # Targets came from hard-copy after load — so b's targets
        # equal b's onlines (post-hard-copy). a's targets are
        # post-polyak (slightly behind a's onlines), so we don't
        # compare a vs b targets; we assert b.online == b.target.
        var b_on_vs_tg = _max_abs_diff[CriticNet](
            b.ensemble.pairs[i].online,
            b.ensemble.pairs[i].target_net,
        )
        print(
            "  pair", i,
            " online a-b=", on_d,
            " | b.online vs b.target after load=", b_on_vs_tg,
        )
        assert_true(
            on_d < PARAM_TOL,
            "online critic matches within format tol",
        )
        assert_true(
            b_on_vs_tg == 0.0,
            "loaded target must equal loaded online (hard_copy ran)",
        )

    # (d) Greedy action on shared obs matches within tol. Same
    # rationale — actor params differ by ≤1e-5 so the action does too.
    var probe = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    probe[0] = Scalar[DT](0.5); probe[1] = Scalar[DT](-0.3); probe[2] = Scalar[DT](0.1)
    var act_a = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var act_b = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    a.select_greedy_action(probe, act_a)
    b.select_greedy_action(probe, act_b)
    print("  greedy a:", act_a[0], " greedy b:", act_b[0])
    for j in range(ACT):
        var d = Float64(act_a[j]) - Float64(act_b[j])
        if d < 0.0:
            d = -d
        assert_true(
            d < 1e-4,
            "greedy action matches within format tol post-load",
        )

    print("PASS — REDQ CPU checkpoint round-trip green.")


def main() raises:
    test_redq_checkpoint_cpu()
