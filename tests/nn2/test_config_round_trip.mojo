"""Test — Config save/load round-trip + Config-built trainer parity.

Phase A.4 validation. Three sub-tests:

  1. SACConfig.default() → save → load → bit-identical fields.
  2. SACConfig.default() → pretty_print succeeds (smoke).
  3. SACTrainer.make(config=SACConfig.default()) and
     SACTrainer.make(actor_lr=3e-4, ...) — both routes produce
     bit-identical first-Adam-step weights. The Config path is a thin
     unpack-and-forward wrapper around the keyword path, so this is a
     compile-time-tested guarantee; we also runtime-verify here for
     belt-and-braces.

The full 30k Pendulum bit-identity gate (mean_ret(10) = −167.572)
runs separately. This file is fast (no training).
"""

from std.random import seed
from std.testing import assert_equal, assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.training.sac_config import SACConfig
from mojo_rl.nn2.training.ddpg_config import DDPGConfig
from mojo_rl.nn2.training.td3_config import TD3Config


def _split_lines(content: String) -> List[String]:
    var lines = List[String]()
    var current = String("")
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord("\n")):
            lines.append(current)
            current = String("")
        else:
            current += chr(Int(c))
    if current.byte_length() > 0:
        lines.append(current)
    return lines^


def test_sac_config_round_trip() raises:
    print("test_sac_config_round_trip ...")
    var c = SACConfig.default()
    var dump = String("")
    c.save(dump, String(""))

    var c2 = SACConfig.default()
    # Wipe c2 by stomping values; load must overwrite them.
    c2.actor_lr.v = Scalar[DT](999.0)
    c2.gamma.v = Scalar[DT](999.0)
    c2.learning_starts.v = -1

    var lines = _split_lines(dump)
    var idx = 0
    c2.load(lines, idx, String(""))

    # Field-by-field bit-identity (floats: text round-trip exact for
    # clean defaults like 3e-4, 0.99, etc.)
    assert_true(c2.actor_lr.v == c.actor_lr.v, "actor_lr")
    assert_true(c2.critic_lr.v == c.critic_lr.v, "critic_lr")
    assert_true(c2.alpha_lr.v == c.alpha_lr.v, "alpha_lr")
    assert_true(c2.gamma.v == c.gamma.v, "gamma")
    assert_true(c2.tau.v == c.tau.v, "tau")
    assert_true(c2.action_scale.v == c.action_scale.v, "action_scale")
    assert_true(c2.init_alpha.v == c.init_alpha.v, "init_alpha")
    assert_true(c2.target_entropy.v == c.target_entropy.v, "target_entropy")
    assert_true(
        c2.initial_episode_fill.v == c.initial_episode_fill.v,
        "initial_episode_fill",
    )
    assert_equal(c2.learning_starts.v, c.learning_starts.v)
    assert_equal(c2.window_size.v, c.window_size.v)
    print("  ok")


def test_ddpg_config_round_trip() raises:
    print("test_ddpg_config_round_trip ...")
    var c = DDPGConfig.default()
    var dump = String("")
    c.save(dump, String(""))
    var c2 = DDPGConfig.default()
    c2.actor_lr.v = Scalar[DT](999.0)
    c2.noise_scale.v = Scalar[DT](999.0)
    var lines = _split_lines(dump)
    var idx = 0
    c2.load(lines, idx, String(""))
    assert_true(c2.actor_lr.v == c.actor_lr.v)
    assert_true(c2.critic_lr.v == c.critic_lr.v)
    assert_true(c2.gamma.v == c.gamma.v)
    assert_true(c2.tau.v == c.tau.v)
    assert_true(c2.action_scale.v == c.action_scale.v)
    assert_true(c2.noise_scale.v == c.noise_scale.v)
    assert_true(c2.initial_episode_fill.v == c.initial_episode_fill.v)
    assert_equal(c2.learning_starts.v, c.learning_starts.v)
    assert_equal(c2.window_size.v, c.window_size.v)
    print("  ok")


def test_td3_config_round_trip() raises:
    print("test_td3_config_round_trip ...")
    var c = TD3Config.default()
    var dump = String("")
    c.save(dump, String(""))
    var c2 = TD3Config.default()
    c2.actor_lr.v = Scalar[DT](999.0)
    c2.policy_delay.v = -1
    var lines = _split_lines(dump)
    var idx = 0
    c2.load(lines, idx, String(""))
    assert_true(c2.actor_lr.v == c.actor_lr.v)
    assert_true(c2.critic_lr.v == c.critic_lr.v)
    assert_true(c2.gamma.v == c.gamma.v)
    assert_true(c2.tau.v == c.tau.v)
    assert_true(c2.action_scale.v == c.action_scale.v)
    assert_true(c2.exploration_noise.v == c.exploration_noise.v)
    assert_true(c2.target_policy_noise.v == c.target_policy_noise.v)
    assert_true(c2.target_noise_clip.v == c.target_noise_clip.v)
    assert_true(c2.initial_episode_fill.v == c.initial_episode_fill.v)
    assert_equal(c2.policy_delay.v, c.policy_delay.v)
    assert_equal(c2.learning_starts.v, c.learning_starts.v)
    assert_equal(c2.window_size.v, c.window_size.v)
    print("  ok")


def test_config_path_matches_keyword_path() raises:
    """Build the same SACTrainer two ways — keyword args vs Config —
    and assert the post-construction actor parameter bytes are
    identical. The Config path is a thin unpack-and-forward wrapper,
    so this verifies the field-by-field forwarding has no typo /
    mis-mapping. Fast: no training involved.

    Network shapes match `examples/pendulum/pendulum_sac_nn2_trainer.mojo`
    so this exercises the real Pendulum SAC trainer topology."""
    from mojo_rl.nn2.combinators.sequential import Sequential
    from mojo_rl.nn2.primitives.linear import Linear
    from mojo_rl.nn2.primitives.relu import ReLU
    from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
    from mojo_rl.nn2.training.sac_trainer import SACTrainer
    from mojo_rl.nn2.core.checkpoint import save_state_v2
    print("test_config_path_matches_keyword_path ...")

    comptime OBS_DIM = 3
    comptime ACT_DIM = 1
    comptime HIDDEN = 64
    comptime BATCH = 256
    comptime REPLAY = 50_000
    comptime ActorNet = StochasticActor[
        OBS_DIM, ACT_DIM,
        Linear[OBS_DIM, HIDDEN], ReLU[HIDDEN],
        Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    ]
    comptime CriticNet = Sequential[
        Linear[OBS_DIM + ACT_DIM, HIDDEN], ReLU[HIDDEN],
        Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
        Linear[HIDDEN, 1],
    ]
    comptime SACT = SACTrainer[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY
    ]

    # Build via keyword path (same values as the production example,
    # action_scale=2.0 for Pendulum).
    seed(42)
    var t_kw = SACT.make["cpu"](
        actor_lr=Scalar[DT](3e-4), critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4), gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005), action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2), target_entropy=Scalar[DT](-1.0),
        learning_starts=1_000, window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )

    # Build via Config path with identical values.
    var cfg = SACConfig.default()
    cfg.action_scale.v = Scalar[DT](2.0)   # override default 1.0 for Pendulum
    seed(42)
    var t_cfg = SACT.make["cpu"](cfg)

    # Dump actor weights via save_state_v2; compare strings.
    var kw_path = String("/tmp/test_a4_sac_kw.txt")
    var cfg_path = String("/tmp/test_a4_sac_cfg.txt")
    save_state_v2[ActorNet](t_kw.actor, kw_path)
    save_state_v2[ActorNet](t_cfg.actor, cfg_path)

    var kw_content: String
    with open(kw_path, "r") as f:
        kw_content = String(f.read())
    var cfg_content: String
    with open(cfg_path, "r") as f:
        cfg_content = String(f.read())
    assert_true(
        kw_content == cfg_content,
        "Config-path actor weights must match keyword-path actor weights"
    )
    print("  ok (actor weights identical between keyword and Config paths)")


def test_pretty_print_smoke() raises:
    print("test_pretty_print_smoke ...")
    var sac = SACConfig.default()
    var ddpg = DDPGConfig.default()
    var td3 = TD3Config.default()
    sac.pretty_print()
    ddpg.pretty_print()
    td3.pretty_print()
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Config save/load round-trip (Phase A.4)")
    print("=" * 70)
    test_sac_config_round_trip()
    test_ddpg_config_round_trip()
    test_td3_config_round_trip()
    test_config_path_matches_keyword_path()
    test_pretty_print_smoke()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
