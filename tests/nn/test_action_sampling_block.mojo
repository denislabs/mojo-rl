"""ActionSamplingBlock — Block E-2.

Validates:
  1. Warmup branch (step_idx < learning_starts) emits uniform actions in
     [-action_scale, +action_scale]^ACT_DIM
  2. Post-warmup deterministic branch: action equals clamp(actor(obs))
     for `select_deterministic` (ACTOR.OUT_DIM == ACT_DIM)
  3. Stochastic branch: `select_stochastic` with a Linear[1, 2*ACT_DIM]
     actor + an RSample[ACT_DIM] sampler produces actions consistent
     with the underlying chain (clamped within ±action_scale)
  4. Deterministic + noise branch: action distribution shifts vs the
     no-noise call (smoke check — noise injection is wired)
"""

from std.math import abs as fabs
from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.deep_agents.primitives.rsample import RSample
from mojo_rl.deep_agents.training.action_sampling_block import ActionSamplingBlock
from mojo_rl.nn.initializer import Kaiming


def test_warmup_uniform_range() raises:
    comptime OBS_DIM = 3
    comptime ACT_DIM = 2
    # Actor isn't used in warmup, but the block type still parameterizes
    # on its concrete type. Use a Linear that satisfies the type.
    comptime ActorT = Linear[OBS_DIM, ACT_DIM]
    seed(7)
    var actor = ActorT.make["cpu", INIT=Kaiming]()
    var block = ActionSamplingBlock[ActorT, OBS_DIM, ACT_DIM, ACT_DIM].make["cpu"]()

    var obs_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OBS_DIM)
    var act_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ACT_DIM)
    for d in range(OBS_DIM):
        obs_p[d] = 0.0

    var scale = Scalar[DT](2.5)
    var n_outside = 0
    for trial in range(200):
        block.select_deterministic["cpu"](
            actor, obs_p, act_p,
            step_idx=trial, learning_starts=10_000, action_scale=scale,
        )
        for j in range(ACT_DIM):
            if act_p[j] > scale or act_p[j] < -scale:
                n_outside += 1
    assert_true(n_outside == 0, "warmup action out of range: " + String(n_outside))
    obs_p.free()
    act_p.free()
    print("  test_warmup_uniform_range PASSED")


def test_deterministic_matches_actor_clamped() raises:
    """Past warmup, `select_deterministic` writes clamp(actor(obs))."""
    comptime OBS_DIM = 4
    comptime ACT_DIM = 3
    comptime ActorT = Linear[OBS_DIM, ACT_DIM]
    seed(11)
    var actor = ActorT.make["cpu", INIT=Kaiming]()
    var block = ActionSamplingBlock[ActorT, OBS_DIM, ACT_DIM, ACT_DIM].make["cpu"]()

    var obs_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OBS_DIM)
    var act_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ACT_DIM)
    var ref_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ACT_DIM)
    for d in range(OBS_DIM):
        obs_p[d] = Scalar[DT](0.3 * Float64(d) - 0.2)

    var scale = Scalar[DT](1.5)
    block.select_deterministic["cpu"](
        actor, obs_p, act_p, step_idx=100, learning_starts=10, action_scale=scale,
    )

    # Direct: actor.forward into ref_p, then clamp manually.
    var obs_t = TileTensor(obs_p, row_major[1, OBS_DIM]())
    var ref_t = TileTensor(ref_p, row_major[1, ACT_DIM]())
    actor.forward["cpu", 1](obs_t, output=ref_t)

    var max_diff: Scalar[DT] = 0.0
    for j in range(ACT_DIM):
        var v = ref_p[j]
        if v > scale:
            v = scale
        elif v < -scale:
            v = -scale
        var d = fabs(act_p[j] - v)
        if d > max_diff:
            max_diff = d
    assert_true(max_diff == Scalar[DT](0.0),
                "deterministic select diverged from clamp(actor(obs)): " + String(max_diff))
    obs_p.free()
    act_p.free()
    ref_p.free()
    print("  test_deterministic_matches_actor_clamped PASSED")


def test_stochastic_within_scale() raises:
    """Stochastic path: action stays in ±scale; trials produce variation."""
    comptime OBS_DIM = 3
    comptime ACT_DIM = 1
    # Actor emits [mu | log_std], so OUT_DIM = 2*ACT_DIM.
    comptime ActorT = Linear[OBS_DIM, 2 * ACT_DIM]
    comptime SamplerT = RSample[ACT_DIM]
    seed(17)
    var actor = ActorT.make["cpu", INIT=Kaiming]()
    var sampler = SamplerT.make["cpu", INIT=Kaiming]()
    var block = ActionSamplingBlock[
        ActorT, OBS_DIM, ACT_DIM, ACT_DIM + 1
    ].make["cpu"]()

    var obs_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OBS_DIM)
    var act_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ACT_DIM)
    for d in range(OBS_DIM):
        obs_p[d] = 0.1

    var scale = Scalar[DT](1.0)
    var first: Scalar[DT] = 0.0
    var second: Scalar[DT] = 0.0
    block.select_stochastic["cpu", SAMPLER=SamplerT](
        actor, sampler, obs_p, act_p,
        step_idx=100, learning_starts=10, action_scale=scale,
    )
    first = act_p[0]
    block.select_stochastic["cpu", SAMPLER=SamplerT](
        actor, sampler, obs_p, act_p,
        step_idx=101, learning_starts=10, action_scale=scale,
    )
    second = act_p[0]

    assert_true(first <= scale and first >= -scale,
                "stochastic action out of scale on call 1: " + String(first))
    assert_true(second <= scale and second >= -scale,
                "stochastic action out of scale on call 2: " + String(second))
    assert_true(first != second,
                "stochastic samples identical across two RNG draws (RSample stuck?)")
    obs_p.free()
    act_p.free()
    print("  test_stochastic_within_scale PASSED")


def test_deterministic_with_noise_changes_distribution() raises:
    """Deterministic+noise produces non-zero shift across multiple draws
    relative to clamp(actor(obs))."""
    comptime OBS_DIM = 2
    comptime ACT_DIM = 2
    comptime ActorT = Linear[OBS_DIM, ACT_DIM]
    seed(23)
    var actor = ActorT.make["cpu", INIT=Kaiming]()
    var block = ActionSamplingBlock[ActorT, OBS_DIM, ACT_DIM, ACT_DIM].make["cpu"]()

    var obs_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OBS_DIM)
    var act_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ACT_DIM)
    var ref_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ACT_DIM)
    obs_p[0] = 0.1
    obs_p[1] = -0.2

    var scale = Scalar[DT](5.0)  # large scale so clamp rarely fires
    # Reference deterministic action (no noise).
    block.select_deterministic["cpu"](
        actor, obs_p, ref_p, step_idx=100, learning_starts=10, action_scale=scale,
    )

    var n_diff = 0
    var max_shift: Scalar[DT] = 0.0
    for trial in range(50):
        block.select_deterministic_with_noise["cpu"](
            actor, obs_p, act_p,
            step_idx=100 + trial, learning_starts=10, action_scale=scale,
            noise_scale=Scalar[DT](0.5),
        )
        var diff_any = False
        for j in range(ACT_DIM):
            var d = fabs(act_p[j] - ref_p[j])
            if d > Scalar[DT](1e-8):
                diff_any = True
            if d > max_shift:
                max_shift = d
        if diff_any:
            n_diff += 1
    assert_true(n_diff >= 45,
                "noise injection not effective: only " + String(n_diff) + "/50 differ")
    print("  noise max_shift=" + String(max_shift) + " over 50 trials, " + String(n_diff) + " differ from ref")
    obs_p.free()
    act_p.free()
    ref_p.free()
    print("  test_deterministic_with_noise_changes_distribution PASSED")


def main() raises:
    print("=" * 60)
    print("nn ActionSamplingBlock tests (Block E-2)")
    print("=" * 60)
    test_warmup_uniform_range()
    test_deterministic_matches_actor_clamped()
    test_stochastic_within_scale()
    test_deterministic_with_noise_changes_distribution()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
