"""DQNTrainer.select_action_batched[N_ENVS>1] GPU smoke (M6).

Before M6 the GPU `select_action_batched` raised a `comptime assert
N_ENVS == 1` in both the warmup and policy branches, capping GPU-batched
DQN to a single env (while the sibling C51 trainer already supported
N_ENVS>1 via its lazily-sized batched-action scratch). M6 ports that same
`_ensure_batch_scratch` pattern to DQN.

This smoke drives N_ENVS=4 device obs through both branches:
  (1) warmup (step < learning_starts) → 4 uniform random action indices.
  (2) policy (step ≥ learning_starts, ε=0) → 4 greedy argmax indices.
Both readbacks must be valid action indices in [0, NUM_ACTIONS). The key
gate is simply that the multi-env GPU path runs at all (it could not be
constructed before M6) and writes one valid action per env lane.

Run: `pixi run -e apple mojo run -I . tests/nn/test_dqn_select_action_batched_gpu.mojo`
"""

from std.random import seed
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.dqn.trainer import DQNTrainer
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN = 32
comptime BATCH = 32
comptime CAP = 1_024
comptime N_ENVS = 4
comptime WARMUP = 100


def main() raises:
    print("=" * 70)
    print("DQN select_action_batched[N_ENVS=4] GPU smoke (M6)")
    print("=" * 70)
    var ctx = DeviceContext()
    seed(42)

    comptime QNet = Sequential[
        Linear[OBS_DIM, HIDDEN],
        ReLU[HIDDEN],
        Linear[HIDDEN, NUM_ACTIONS],
    ]
    var trainer = DQNTrainer[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
        QNet,
    ].make(
        ctx=ctx,
        lr=Scalar[DT](1e-3),
        epsilon=Scalar[DT](0.0),  # ε=0 → policy branch is pure greedy
        learning_starts=WARMUP,
    )

    # Device obs [N_ENVS, OBS] + device action [N_ENVS] buffers.
    var obs_dev = ctx.enqueue_create_buffer[DT](N_ENVS * OBS_DIM)
    var act_dev = ctx.enqueue_create_buffer[DT](N_ENVS)
    var obs_host = List[Scalar[DT]](
        length=N_ENVS * OBS_DIM, fill=Scalar[DT](0.0)
    )
    for i in range(N_ENVS * OBS_DIM):
        obs_host[i] = Scalar[DT](0.1 * Float64(i) - 0.5)
    ctx.enqueue_copy(obs_dev, obs_host.unsafe_ptr())
    ctx.synchronize()

    var obs_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        obs_dev.unsafe_ptr()
    )
    var act_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        act_dev.unsafe_ptr()
    )
    var act_host = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](-1.0))

    # (1) Warmup branch: step_idx < learning_starts → random actions.
    trainer.select_action_batched[N_ENVS](obs_p, act_p, 0)
    ctx.enqueue_copy(act_host.unsafe_ptr(), act_dev)
    ctx.synchronize()
    print("  warmup actions:", act_host[0], act_host[1], act_host[2], act_host[3])
    for i in range(N_ENVS):
        var a = Int(act_host[i])
        assert_true(
            a >= 0 and a < NUM_ACTIONS,
            "warmup action lane must be a valid index",
        )

    # (2) Policy branch: step_idx ≥ learning_starts, ε=0 → greedy argmax.
    for i in range(N_ENVS):
        act_host[i] = Scalar[DT](-1.0)
    ctx.enqueue_copy(act_dev, act_host.unsafe_ptr())
    ctx.synchronize()
    trainer.select_action_batched[N_ENVS](obs_p, act_p, WARMUP + 1)
    ctx.enqueue_copy(act_host.unsafe_ptr(), act_dev)
    ctx.synchronize()
    print("  greedy actions:", act_host[0], act_host[1], act_host[2], act_host[3])
    for i in range(N_ENVS):
        var a = Int(act_host[i])
        assert_true(
            a >= 0 and a < NUM_ACTIONS,
            "policy action lane must be a valid index",
        )

    print("=" * 70)
    print("PASS — DQN GPU select_action_batched runs at N_ENVS=4")
    print("=" * 70)
