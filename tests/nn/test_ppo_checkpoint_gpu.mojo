"""PPO GPU checkpoint round-trip + GPU→CPU interchange (Phase 2).

PPO is continuous/on-policy; this test exercises the GPU save/load
SERIALIZATION paths (actor + critic params, actor_opt + critic_opt Adam
moments) without a full rollout loop — the numeric Adam-moment round-trip
rigor is covered by `test_dqn_checkpoint_gpu.mojo` (shared
`save_optimizer_v2_body_gpu` / `load_optimizer_v2_body_gpu` helpers).

Invariants:
  1. GPU save → fresh-GPU load → re-save is byte-identical.
  2. GPU save → CPU load → re-save is byte-identical (the GPU checkpoint
     is a valid CPU checkpoint → train-on-GPU → eval-on-CPU).

Guards on DeviceContext (no-op on CPU-only CI).
Run: pixi run -e apple mojo run -I . tests/nn/test_ppo_checkpoint_gpu.mojo
"""

from std.gpu.host import DeviceContext
from std.testing import assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.tanh import Tanh
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.ppo.trainer import PPOTrainer


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime ROLLOUT = 128
comptime MINIBATCH = 64
comptime N_EPOCHS = 4
comptime ACTOR = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, 2 * ACT_DIM],
]
comptime CRITIC = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, 1],
]

comptime CKPT = String("/tmp/ppo_gpu_ckpt.ckpt")
comptime CKPT_B = String("/tmp/ppo_gpu_ckpt_b.ckpt")


def _read_file(path: String) raises -> String:
    with open(path, "r") as f:
        return String(f.read())


def test_ppo_gpu_checkpoint() raises:
    print("--- PPO GPU checkpoint round-trip ---")
    try:
        var _probe = DeviceContext()
    except:
        print("  no accelerator — skipping")
        return

    var trainer = PPOTrainer[
        "gpu", ACTOR, CRITIC, OBS_DIM, ACT_DIM, ROLLOUT, MINIBATCH, N_EPOCHS,
    ].make(ctx=DeviceContext())
    trainer.save_state(CKPT)
    print("  saved GPU PPO trainer ->", CKPT)

    # Invariant 1: GPU → fresh-GPU → re-save byte-identical.
    var loaded = PPOTrainer[
        "gpu", ACTOR, CRITIC, OBS_DIM, ACT_DIM, ROLLOUT, MINIBATCH, N_EPOCHS,
    ].make(ctx=DeviceContext())
    loaded.load_state(CKPT)
    loaded.save_state(CKPT_B)
    assert_equal(
        _read_file(CKPT), _read_file(CKPT_B),
        "PPO GPU checkpoint not byte-identical after GPU save→load→save",
    )
    print("  [1] GPU re-save byte-identity OK")

    # Invariant 2: GPU → CPU → re-save byte-identical (interchange).
    var cpu = PPOTrainer[
        "cpu", ACTOR, CRITIC, OBS_DIM, ACT_DIM, ROLLOUT, MINIBATCH, N_EPOCHS,
    ].make()
    cpu.load_state(CKPT)
    cpu.save_state(CKPT_B)
    assert_equal(
        _read_file(CKPT), _read_file(CKPT_B),
        "PPO GPU→CPU interchange: CPU re-save not byte-identical",
    )
    print("  [2] GPU→CPU interchange OK ( CPU re-save byte-identical )")
    print("PASS")


def main() raises:
    test_ppo_gpu_checkpoint()
