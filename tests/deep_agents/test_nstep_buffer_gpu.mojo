"""Test GPUNStepBuffer for correctness."""

from std.gpu.host import DeviceContext, HostBuffer
from mojo_rl.deep_agents.core.replay import GPUNStepBuffer
from mojo_rl.nn.constants import dtype


fn main() raises:
    print("=== GPUNStepBuffer Tests ===")

    with DeviceContext() as ctx:
        # 4 environments, 3-step, obs_dim=2, gamma=1.0 for easy verification
        var nstep = GPUNStepBuffer[3, 2, 4](ctx, gamma=1.0)

        # Create host buffers for input
        var h_obs = ctx.enqueue_create_host_buffer[dtype](4 * 2)   # [4, 2]
        var h_act = ctx.enqueue_create_host_buffer[dtype](4)       # [4]
        var h_rew = ctx.enqueue_create_host_buffer[dtype](4)       # [4]
        var h_nobs = ctx.enqueue_create_host_buffer[dtype](4 * 2)  # [4, 2]
        var h_done = ctx.enqueue_create_host_buffer[dtype](4)      # [4]

        # Device buffers
        var d_obs = ctx.enqueue_create_buffer[dtype](4 * 2)
        var d_act = ctx.enqueue_create_buffer[dtype](4)
        var d_rew = ctx.enqueue_create_buffer[dtype](4)
        var d_nobs = ctx.enqueue_create_buffer[dtype](4 * 2)
        var d_done = ctx.enqueue_create_buffer[dtype](4)

        # --- Step 0: all envs get reward=1.0, not done ---
        for e in range(4):
            h_obs[e * 2] = Scalar[dtype](Float64(e))
            h_obs[e * 2 + 1] = Scalar[dtype](0)
            h_act[e] = Scalar[dtype](0)
            h_rew[e] = Scalar[dtype](1.0)
            h_nobs[e * 2] = Scalar[dtype](0)
            h_nobs[e * 2 + 1] = Scalar[dtype](0)
            h_done[e] = Scalar[dtype](0.0)
        ctx.enqueue_copy(d_obs, h_obs)
        ctx.enqueue_copy(d_act, h_act)
        ctx.enqueue_copy(d_rew, h_rew)
        ctx.enqueue_copy(d_nobs, h_nobs)
        ctx.enqueue_copy(d_done, h_done)

        nstep.process(ctx, d_obs, d_act, d_rew, d_nobs, d_done)

        # Check: no valid outputs yet (only 1 step)
        var h_valid = ctx.enqueue_create_host_buffer[DType.int32](4)
        ctx.enqueue_copy(h_valid, nstep.out_valid)
        ctx.synchronize()
        var any_valid = False
        for e in range(4):
            if h_valid[e] != 0:
                any_valid = True
        print("Step 0: any_valid =", any_valid, "(expected False)")

        # --- Step 1: all envs get reward=2.0, env 2 done ---
        for e in range(4):
            h_rew[e] = Scalar[dtype](2.0)
            h_done[e] = Scalar[dtype](0.0)
        h_done[2] = Scalar[dtype](1.0)  # env 2 terminates
        ctx.enqueue_copy(d_rew, h_rew)
        ctx.enqueue_copy(d_done, h_done)

        nstep.process(ctx, d_obs, d_act, d_rew, d_nobs, d_done)

        ctx.enqueue_copy(h_valid, nstep.out_valid)
        ctx.synchronize()

        # Only env 2 should be valid (done → partial flush, 2-step R = 1+2 = 3)
        print("Step 1: valid = [", end="")
        for e in range(4):
            print(h_valid[e], end="")
            if e < 3:
                print(", ", end="")
        print("]  (expected [0, 0, 1, 0])")

        # Check env 2's reward: R_2 = 1.0 + 1.0*2.0 = 3.0 (gamma=1.0)
        var h_out_rew = ctx.enqueue_create_host_buffer[dtype](4)
        ctx.enqueue_copy(h_out_rew, nstep.out_rew)
        ctx.synchronize()
        print("Step 1: env 2 R =", h_out_rew[2], "(expected 3.0)")

        # --- Step 2: all envs get reward=3.0, none done ---
        for e in range(4):
            h_rew[e] = Scalar[dtype](3.0)
            h_done[e] = Scalar[dtype](0.0)
        ctx.enqueue_copy(d_rew, h_rew)
        ctx.enqueue_copy(d_done, h_done)

        nstep.process(ctx, d_obs, d_act, d_rew, d_nobs, d_done)

        ctx.enqueue_copy(h_valid, nstep.out_valid)
        ctx.synchronize()

        # Envs 0,1,3 should emit (3 steps accumulated: R=1+2+3=6)
        # Env 2 was reset after done, so only has 1 step now (not ready)
        print("Step 2: valid = [", end="")
        for e in range(4):
            print(h_valid[e], end="")
            if e < 3:
                print(", ", end="")
        print("]  (expected [1, 1, 0, 1])")

        ctx.enqueue_copy(h_out_rew, nstep.out_rew)
        ctx.synchronize()
        print(
            "Step 2: env 0 R =",
            h_out_rew[0],
            " env 1 R =",
            h_out_rew[1],
            " (expected 6.0)",
        )

    print("=== GPUNStepBuffer Tests Complete ===")
