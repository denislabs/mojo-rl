"""Relabelling gate: is a task's reward a function of `(qpos, qvel, action)`?

`docs/BFM_ZERO_SHOT_RL.md` §14 lists this as the first of four things asserted
from reading source rather than measured, and §11 ranks the failure it guards
against as the most expensive one to find late: a reward that is NOT
recomputable offline only surfaces at component 5, after 10 M transitions have
been collected in a format that cannot express it.

The measurement is the obvious one. Run a rollout, recording the state AFTER
each step together with the action that produced it and the reward the env
returned. Then, on a SEPARATE env, call `reward_at(qpos, qvel, action)` on each
recorded transition IN A SHUFFLED ORDER and diff. Three properties are gated at
once:

  * the reward is reproduced from generalized coordinates alone;
  * the observation is too (`obs_at`) — which is what lets the dataset store 18
    floats for walker instead of 24;
  * order does not matter. Visiting the transitions out of sequence is the
    discriminating part: a task carrying hidden state across calls (`prev_x`,
    an integrator warm start, a step counter) reproduces its own rollout
    perfectly IN ORDER and falls apart shuffled. Replaying in order would have
    passed a non-Markovian task and gated nothing.

⚠ This gates the tasks named below and nothing else. It is a per-task property:
the Gym-derived configs in this same package deliberately fail it — HalfCheetah's
forward reward is `(x - prev_x) / dt`, so `reward_at` on it returns a number
computed against `prev_x = 0`, silently. `test_gym_config_is_not_relabelable`
pins that as a fact rather than leaving it as a warning in a docstring.

Run with:
    pixi run mojo run -I . tests/dm_control/test_reward_relabel.mojo
"""

from std.math import abs
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig

from mojo_rl.envs.dm_control.point_mass import (
    DMPointMassModel,
    DMPointMassConfig,
)
from mojo_rl.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig
from mojo_rl.envs.dm_control.cheetah import DMCheetahModel, DMCheetahConfig
from mojo_rl.envs.half_cheetah import HalfCheetahModel, HalfCheetahConfig


comptime N_STEPS: Int = 60
comptime SEED: Int = 20260805

# fp64 everywhere: relabelling must be exact, and a float32 env would fold the
# question "is this reward Markovian" into "does it round the same way twice".
comptime TOL: Float64 = 1e-12


def _pseudo_action(step: Int, k: Int, scale: Float64) -> Float64:
    """Deterministic, non-degenerate, and NOT constant across actuators.

    A constant or all-equal action would leave `small_control`-style terms
    invariant and hide an action-indexing bug in the relabel path.

    Deliberately LOW frequency in `step` (a sawtooth over ~16 steps rather than
    sign-flipping every step): a fast-alternating control averages to zero and
    leaves the body near where it started, which is how `point_mass` ended up
    sweeping a 2e-4 band of its reward instead of the [0, 1] the task actually
    spans.
    """
    var t = Float64(step) * 0.017 + Float64(k) * 0.61
    return scale * ((t * 7.13) % 2.0 - 1.0)


def _sweep_action(step: Int, steps: Int, k: Int, scale: Float64) -> Float64:
    """A single monotone ramp from -scale to +scale, offset per actuator.

    For `point_mass` an oscillating control gets nowhere: the joints carry
    `damping="1"` against a 0.3 kg mass and the motors `gear=".1"`, so terminal
    velocity is 0.1·ctrl m/s with a 0.3 s time constant. Anything that changes
    sign every few steps never leaves the 1.5 cm target radius, and the reward
    stays pinned near 1 — which is how the first version of this gate swept a
    3e-3 band of a reward that spans [0, 1].
    """
    var u = 2.0 * Float64(step) / Float64(steps) - 1.0
    var v = u + 0.37 * Float64(k)
    if v > 1.0:
        v = v - 2.0
    return scale * v


struct Rollout(Movable & ImplicitlyDeletable):
    """Post-step states, the actions that produced them, and the online reward.
    """

    var qpos: List[List[Float64]]
    var qvel: List[List[Float64]]
    var act: List[List[Float64]]
    var rew: List[Float64]
    var obs: List[List[Float64]]

    def __init__(out self):
        self.qpos = List[List[Float64]]()
        self.qvel = List[List[Float64]]()
        self.act = List[List[Float64]]()
        self.rew = List[Float64]()
        self.obs = List[List[Float64]]()

    def __init__(out self, *, deinit move: Self):
        self.qpos = move.qpos^
        self.qvel = move.qvel^
        self.act = move.act^
        self.rew = move.rew^
        self.obs = move.obs^


def _collect[
    MODEL_DEF: ModelDefLike, CONFIG: Phyics3dEnvConfig
](
    steps: Int, act_scale: Float64, ref start_qpos: List[Float64], sweep: Bool
) raises -> Rollout:
    comptime Env = Phyics3dEnv[MODEL_DEF, CONFIG, DType.float64, False]
    comptime NQ = MODEL_DEF.NQ
    comptime NV = MODEL_DEF.NV
    comptime NACT = MODEL_DEF.ACTION_DIM

    var env = Env()
    _ = env.reset()
    if len(start_qpos) > 0:
        # Placing the start state is not cosmetic: `point_mass`'s reward is
        # ~1e-144 anywhere but within a few centimetres of the target, and a
        # gate run out there compares two underflowed zeros and passes on a
        # completely broken relabel path.
        var zv = List[Float64]()
        for _ in range(NV):
            zv.append(0.0)
        env.set_state(start_qpos, zv)
    var r = Rollout()

    for step in range(steps):
        var a = Env.ActionType()
        var alist = List[Float64]()
        for k in range(NACT):
            var v = (
                _sweep_action(step, steps, k, act_scale) if sweep
                else _pseudo_action(step, k, act_scale)
            )
            a.data[k] = v
            alist.append(v)

        var out = env.step(a)

        var q = List[Float64]()
        for i in range(NQ):
            q.append(Float64(env.d.qpos.data[i]))
        var v = List[Float64]()
        for i in range(NV):
            v.append(Float64(env.d.qvel.data[i]))
        var o = List[Float64]()
        for i in range(Env.OBS_DIM):
            o.append(Float64(out[0].data[i]))

        r.qpos.append(q^)
        r.qvel.append(v^)
        r.act.append(alist^)
        r.rew.append(Float64(out[1]))
        r.obs.append(o^)

    return r^


def _shuffled_order(n: Int) -> List[Int]:
    """A fixed derangement-ish permutation: stride by a coprime step so no
    element keeps its index and consecutive visits are far apart."""
    var order = List[Int]()
    var idx = 0
    for _ in range(n):
        idx = (idx + 23) % n
        order.append(idx)
    return order^


def _relabel_gate[
    MODEL_DEF: ModelDefLike, CONFIG: Phyics3dEnvConfig
](
    name: String,
    expect_markov: Bool = True,
    act_scale: Float64 = 0.9,
    start_qpos: List[Float64] = List[Float64](),
    sweep: Bool = False,
) raises -> Float64:
    """Returns the worst reward error seen. Asserts only when `expect_markov`.
    """
    comptime Env = Phyics3dEnv[MODEL_DEF, CONFIG, DType.float64, False]

    seed(SEED)
    var roll = _collect[MODEL_DEF, CONFIG](
        N_STEPS, act_scale, start_qpos, sweep
    )

    # A separate env: relabelling must not depend on the collecting env's
    # residual state, and `reward_at` is destructive.
    var scorer = Env()
    _ = scorer.reset()

    var max_rew_err = Float64(0)
    var max_obs_err = Float64(0)
    var order = _shuffled_order(N_STEPS)

    for oi in range(len(order)):
        var t = order[oi]
        var got = scorer.reward_at(roll.qpos[t], roll.qvel[t], roll.act[t])
        var e = abs(Float64(got[0]) - roll.rew[t])
        if e > max_rew_err:
            max_rew_err = e

        var o = scorer.obs_at(roll.qpos[t], roll.qvel[t])
        for i in range(Env.OBS_DIM):
            var eo = abs(Float64(o.data[i]) - roll.obs[t][i])
            if eo > max_obs_err:
                max_obs_err = eo

    # A reward that never moves would pass any tolerance. Report the range so
    # the number above is readable as a RELATIVE claim.
    var rmin = roll.rew[0]
    var rmax = roll.rew[0]
    for t in range(N_STEPS):
        if roll.rew[t] < rmin:
            rmin = roll.rew[t]
        if roll.rew[t] > rmax:
            rmax = roll.rew[t]

    print(
        "   ", name, "  reward err", max_rew_err, " obs err", max_obs_err,
        " (reward range [", rmin, ",", rmax, "])",
    )

    if expect_markov:
        assert_true(
            rmax - rmin > 1e-9,
            String(name) + ": the reward never moved over the rollout — this"
            " gate would pass on a broken relabel path. Pick a start state or"
            " an action sequence that exercises the reward.",
        )
        assert_true(
            max_rew_err < TOL,
            String(name) + ": relabelled reward differs by "
            + String(max_rew_err) + " (tol " + String(TOL) + "). The task is"
            " NOT a function of (qpos, qvel, action) and cannot feed a BFM"
            " dataset — see this file's docstring.",
        )
        assert_true(
            max_obs_err < TOL,
            String(name) + ": obs_at differs by " + String(max_obs_err)
            + " — the observation is not recoverable from generalized"
            " coordinates, so the dataset cannot drop it.",
        )
    return max_rew_err


def test_point_mass() raises:
    print("[1] point_mass easy ...")
    # Start ON the target with gentle controls, so the rollout sweeps the
    # reward from ~1 down as the mass drifts out. See `_collect`.
    var q0 = List[Float64](length=2, fill=0.0)
    _ = _relabel_gate[DMPointMassModel, DMPointMassConfig](
        "point_mass-easy", act_scale=1.0, start_qpos=q0, sweep=True
    )


def test_walker() raises:
    print("[2] walker stand / walk / run ...")
    # All three share a model and differ only in MOVE_SPEED, and that is worth
    # gating separately: `stand` reads xpos+xmat only, while `walk`/`run` also
    # read `xvel` through `subtree_linvel`. `xvel` is the field that a naive
    # `set_state` (qpos only, no `_fields_vel`) would leave stale, so `stand`
    # passing tells you nothing about `walk`.
    _ = _relabel_gate[DMWalkerModel, DMWalkerConfig[0.0]]("walker-stand")
    _ = _relabel_gate[DMWalkerModel, DMWalkerConfig[1.0]]("walker-walk")
    _ = _relabel_gate[DMWalkerModel, DMWalkerConfig[8.0]]("walker-run")


def test_cheetah() raises:
    print("[3] cheetah run ...")
    # dm_control's cheetah, NOT Gym's HalfCheetah below. Same body, opposite
    # answer on this gate — the pair is the point.
    _ = _relabel_gate[DMCheetahModel, DMCheetahConfig]("cheetah-run")


def test_gym_config_is_not_relabelable() raises:
    """The negative control.

    Gym's HalfCheetah rewards forward progress as `(x - prev_x) / dt`, so with
    `prev_x` defaulted to 0 the relabelled reward is wrong by roughly the whole
    forward term. If this ever starts matching, either the config changed or —
    far more likely — the gate above stopped measuring anything.
    """
    print("[4] negative control: Gym HalfCheetah must FAIL to relabel ...")
    var err = _relabel_gate[HalfCheetahModel, HalfCheetahConfig](
        "gym-half_cheetah", expect_markov=False
    )
    assert_true(
        err > 1e-6,
        "Gym HalfCheetah relabelled EXACTLY (err " + String(err) + "). Its"
        " reward reads prev_x, so a match means the relabel path is not"
        " actually recomputing the reward — the positive gates above are then"
        " vacuous.",
    )
    print("      failed as expected  OK")


def main() raises:
    test_point_mass()
    test_walker()
    test_cheetah()
    test_gym_config_is_not_relabelable()
    print("\n[PASS] reward relabelling gate")
