"""GPU PushT smoke test.

Verifies that:
  * The pusht_v2 step/reset GPU kernels compile and run on the active backend
  * Batched simulation produces sane outputs (obs in expected ranges,
    rewards in [0, 1], dones in {0, 1})

Run with:
    pixi run -e apple mojo run -I . tests/test_pusht_gpu.mojo   (Metal)
    pixi run -e nvidia mojo run -I . tests/test_pusht_gpu.mojo  (CUDA)
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import Pointer
from mojo_rl.physics2d import dtype, SHAPE_MAX_SIZE
from mojo_rl.envs.pusht import (
    PushTV2,
    PConstants,
    PushTLayout,
    PushTShapeBuf,
)


comptime BATCH = 4
comptime STATE_SIZE = PushTLayout.STATE_SIZE
comptime OBS_DIM = PConstants.OBS_DIM
comptime ACTION_DIM = PConstants.ACTION_DIM
comptime SHAPES_SIZE = PushTShapeBuf.NUM_SHAPES * SHAPE_MAX_SIZE


def main() raises:
    var ctx = DeviceContext()

    var states = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
    var actions = ctx.enqueue_create_buffer[dtype](BATCH * ACTION_DIM)
    var rewards = ctx.enqueue_create_buffer[dtype](BATCH)
    var dones = ctx.enqueue_create_buffer[dtype](BATCH)
    var terminated = ctx.enqueue_create_buffer[dtype](BATCH)
    var obs = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)
    var workspace = ctx.enqueue_create_buffer[dtype](SHAPES_SIZE)

    # Init workspace (uploads shape buffer) + reset all envs
    PushTV2[dtype].init_step_workspace_gpu[BATCH](ctx, workspace)
    PushTV2[dtype].reset_kernel_gpu[BATCH, STATE_SIZE](
        ctx, states, rng_seed=7
    )
    PushTV2[dtype].extract_obs_kernel_gpu[BATCH, STATE_SIZE, OBS_DIM](
        ctx, states, obs
    )

    # Read back initial obs
    var obs_host = List[Scalar[dtype]](capacity=BATCH * OBS_DIM)
    for _ in range(BATCH * OBS_DIM):
        obs_host.append(Scalar[dtype](0.0))
    ctx.enqueue_copy(obs_host.unsafe_ptr(), obs)
    ctx.synchronize()

    print("== initial obs (each row = 18D) ==")
    for b in range(BATCH):
        # agent_pos is the last two slots
        var ax = obs_host[b * OBS_DIM + PConstants.KEYPOINTS_DIM]
        var ay = obs_host[b * OBS_DIM + PConstants.KEYPOINTS_DIM + 1]
        print("  env=", b, " agent=(", ax, ",", ay, ")")
        if ax < Scalar[dtype](
            PConstants.AGENT_RESET_LOW - 1.0
        ) or ax > Scalar[dtype](PConstants.AGENT_RESET_HIGH + 1.0):
            raise Error("agent x out of reset range")

    # Set actions to (100, 100) for all envs and run a few steps
    var act_host = List[Scalar[dtype]](capacity=BATCH * ACTION_DIM)
    for _ in range(BATCH * ACTION_DIM):
        act_host.append(Scalar[dtype](100.0))
    ctx.enqueue_copy(actions, act_host.unsafe_ptr())

    var rew_host = List[Scalar[dtype]](capacity=BATCH)
    for _ in range(BATCH):
        rew_host.append(Scalar[dtype](0.0))
    var done_host = List[Scalar[dtype]](capacity=BATCH)
    for _ in range(BATCH):
        done_host.append(Scalar[dtype](0.0))

    for step in range(5):
        PushTV2[dtype].step_kernel_gpu[
            BATCH, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            states,
            actions,
            rewards,
            dones,
            terminated,
            obs,
            rng_seed=UInt64(step),
            workspace_ptr=workspace.unsafe_ptr().as_unsafe_any_origin(),
        )
        ctx.enqueue_copy(rew_host.unsafe_ptr(), rewards)
        ctx.enqueue_copy(done_host.unsafe_ptr(), dones)
        ctx.enqueue_copy(obs_host.unsafe_ptr(), obs)
        ctx.synchronize()

        print("step ", step, ":")
        for b in range(BATCH):
            var ax = obs_host[b * OBS_DIM + PConstants.KEYPOINTS_DIM]
            var ay = obs_host[b * OBS_DIM + PConstants.KEYPOINTS_DIM + 1]
            print(
                "  env=",
                b,
                " agent=(",
                ax,
                ",",
                ay,
                ") reward=",
                rew_host[b],
                " done=",
                done_host[b],
            )
            if rew_host[b] < Scalar[dtype](0.0) or rew_host[b] > Scalar[
                dtype
            ](1.0):
                raise Error("reward out of [0,1]")
            if done_host[b] != Scalar[dtype](0.0) and done_host[b] != Scalar[
                dtype
            ](1.0):
                raise Error("done not in {0, 1}")

    print("GPU PushTV2 smoke test passed.")
