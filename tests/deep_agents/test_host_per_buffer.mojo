"""Test HostPrioritizedReplayBuffer store→sample roundtrip."""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.core.replay import HostPrioritizedReplayBuffer


def main() raises:
    print("=" * 60)
    print("Test: HostPrioritizedReplayBuffer roundtrip")
    print("=" * 60)

    var ctx = DeviceContext()

    # Small buffer for testing: 100 capacity, 8 obs_dim, 1 action
    comptime CAP = 100
    comptime OBS = 8
    comptime BATCH = 4
    comptime N_ENVS = 2

    # Test 1: float32 storage (no quantization)
    print("\n--- Test 1: STORE_DTYPE = float32 ---")
    var buf_f32 = HostPrioritizedReplayBuffer[CAP, OBS, 1, BATCH, N_ENVS, dtype](
        ctx
    )
    _test_roundtrip[CAP, OBS, BATCH, N_ENVS](ctx, buf_f32, "float32")

    # Test 2: uint8 storage (quantization)
    print("\n--- Test 2: STORE_DTYPE = uint8 ---")
    var buf_u8 = HostPrioritizedReplayBuffer[
        CAP, OBS, 1, BATCH, N_ENVS, DType.uint8
    ](ctx)
    _test_roundtrip[CAP, OBS, BATCH, N_ENVS](ctx, buf_u8, "uint8")

    # Test 3: Large OBS_DIM (pixel-like)
    print("\n--- Test 3: Large OBS (1024), uint8 ---")
    comptime LARGE_OBS = 1024
    var buf_large = HostPrioritizedReplayBuffer[
        CAP, LARGE_OBS, 1, BATCH, N_ENVS, DType.uint8
    ](ctx)
    _test_large_roundtrip[CAP, LARGE_OBS, BATCH, N_ENVS](ctx, buf_large)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def _test_roundtrip[
    CAP: Int, OBS: Int, BATCH: Int, N_ENVS: Int, SD: DType = dtype
](
    ctx: DeviceContext,
    mut buf: HostPrioritizedReplayBuffer[CAP, OBS, 1, BATCH, N_ENVS, SD],
    name: String,
) raises:
    # Create known data on GPU
    var obs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    var nobs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    var act_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var rew_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var done_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)

    # Fill with known values
    for e in range(N_ENVS):
        for d in range(OBS):
            obs_host[e * OBS + d] = Scalar[dtype](
                Float64(e * OBS + d) / Float64(N_ENVS * OBS)
            )  # [0, 1) range
            nobs_host[e * OBS + d] = Scalar[dtype](
                Float64(e * OBS + d + 1) / Float64(N_ENVS * OBS + 1)
            )
        act_host[e] = Scalar[dtype](e)
        rew_host[e] = Scalar[dtype](Float64(e) * 0.5 - 0.25)
        done_host[e] = Scalar[dtype](0.0)

    # Copy to GPU
    var obs_gpu = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var nobs_gpu = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var act_gpu = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var rew_gpu = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var done_gpu = ctx.enqueue_create_buffer[dtype](N_ENVS)
    ctx.enqueue_copy(obs_gpu, obs_host)
    ctx.enqueue_copy(nobs_gpu, nobs_host)
    ctx.enqueue_copy(act_gpu, act_host)
    ctx.enqueue_copy(rew_gpu, rew_host)
    ctx.enqueue_copy(done_gpu, done_host)

    # Store multiple times to fill buffer
    for _ in range(10):
        buf.store[N_ENVS](ctx, obs_gpu, act_gpu, rew_gpu, nobs_gpu, done_gpu)

    print("  Buffer size:", buf.size, "(expected:", min(N_ENVS * 10, CAP), ")")

    # Sample
    var s_obs = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
    var s_act = ctx.enqueue_create_buffer[dtype](BATCH)
    var s_rew = ctx.enqueue_create_buffer[dtype](BATCH)
    var s_nobs = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
    var s_done = ctx.enqueue_create_buffer[dtype](BATCH)
    var s_idx = ctx.enqueue_create_buffer[DType.int32](BATCH)
    var s_weights = ctx.enqueue_create_buffer[dtype](BATCH)

    buf.sample[BATCH](ctx, s_obs, s_act, s_rew, s_nobs, s_done, s_idx, s_weights)

    # Copy results back to host
    var r_obs = ctx.enqueue_create_host_buffer[dtype](BATCH * OBS)
    var r_nobs = ctx.enqueue_create_host_buffer[dtype](BATCH * OBS)
    var r_act = ctx.enqueue_create_host_buffer[dtype](BATCH)
    var r_rew = ctx.enqueue_create_host_buffer[dtype](BATCH)
    var r_done = ctx.enqueue_create_host_buffer[dtype](BATCH)
    var r_weights = ctx.enqueue_create_host_buffer[dtype](BATCH)
    ctx.enqueue_copy(r_obs, s_obs)
    ctx.enqueue_copy(r_nobs, s_nobs)
    ctx.enqueue_copy(r_act, s_act)
    ctx.enqueue_copy(r_rew, s_rew)
    ctx.enqueue_copy(r_done, s_done)
    ctx.enqueue_copy(r_weights, s_weights)
    ctx.synchronize()

    # Verify: check sampled values are within expected range
    var obs_ok = True
    var nobs_ok = True
    var act_ok = True
    var rew_ok = True
    var weight_ok = True
    var max_obs_err: Float64 = 0.0

    for b in range(BATCH):
        # Check obs in [0, 1]
        for d in range(OBS):
            var v = Float64(r_obs[b * OBS + d])
            if v < -0.01 or v > 1.01:
                obs_ok = False
                print("  BAD obs[", b, ",", d, "] =", v)
            var nv = Float64(r_nobs[b * OBS + d])
            if nv < -0.01 or nv > 1.01:
                nobs_ok = False
                print("  BAD nobs[", b, ",", d, "] =", nv)

            # Check obs matches expected pattern (within quantization tolerance)
            # All stored obs come from the same pattern, so any sample should match
            var expected_v = Float64(r_obs[b * OBS + d])
            var err = v - expected_v
            if err < 0:
                err = -err
            if err > max_obs_err:
                max_obs_err = err

        # Check action is one of the stored values
        var a = Float64(r_act[b])
        if a < -0.5 or a > Float64(N_ENVS):
            act_ok = False
            print("  BAD action[", b, "] =", a)

        # Check reward matches stored pattern
        var r = Float64(r_rew[b])
        if r < -1.0 or r > 1.0:
            rew_ok = False
            print("  BAD reward[", b, "] =", r)

        # Check IS weight is positive
        var w = Float64(r_weights[b])
        if w <= 0.0 or w > 10.0:
            weight_ok = False
            print("  BAD weight[", b, "] =", w)

    print("  Obs values in range [0,1]:", obs_ok)
    print("  Next-obs values in range:", nobs_ok)
    print("  Actions valid:", act_ok)
    print("  Rewards valid:", rew_ok)
    print("  IS weights valid:", weight_ok)
    print("  Max obs error:", max_obs_err)

    # Print first sample for visual inspection
    print("  Sample 0 obs[:4]:", r_obs[0], r_obs[1], r_obs[2], r_obs[3])
    print("  Sample 0 act:", r_act[0], "rew:", r_rew[0], "done:", r_done[0])
    print("  Sample 0 weight:", r_weights[0])


def _test_large_roundtrip[
    CAP: Int, OBS: Int, BATCH: Int, N_ENVS: Int
](
    ctx: DeviceContext,
    mut buf: HostPrioritizedReplayBuffer[
        CAP, OBS, 1, BATCH, N_ENVS, DType.uint8
    ],
) raises:
    # Fill with gradient pattern [0, 1)
    var obs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    var nobs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    var act_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var rew_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var done_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)

    for e in range(N_ENVS):
        for d in range(OBS):
            obs_host[e * OBS + d] = Scalar[dtype](
                Float64(d) / Float64(OBS)
            )
            nobs_host[e * OBS + d] = Scalar[dtype](
                Float64(d + 1) / Float64(OBS + 1)
            )
        act_host[e] = Scalar[dtype](1.0)
        rew_host[e] = Scalar[dtype](0.5)
        done_host[e] = Scalar[dtype](0.0)

    var obs_gpu = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var nobs_gpu = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var act_gpu = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var rew_gpu = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var done_gpu = ctx.enqueue_create_buffer[dtype](N_ENVS)
    ctx.enqueue_copy(obs_gpu, obs_host)
    ctx.enqueue_copy(nobs_gpu, nobs_host)
    ctx.enqueue_copy(act_gpu, act_host)
    ctx.enqueue_copy(rew_gpu, rew_host)
    ctx.enqueue_copy(done_gpu, done_host)

    for _ in range(10):
        buf.store[N_ENVS](ctx, obs_gpu, act_gpu, rew_gpu, nobs_gpu, done_gpu)

    # Sample and verify
    var s_obs = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
    var s_act = ctx.enqueue_create_buffer[dtype](BATCH)
    var s_rew = ctx.enqueue_create_buffer[dtype](BATCH)
    var s_nobs = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
    var s_done = ctx.enqueue_create_buffer[dtype](BATCH)
    var s_idx = ctx.enqueue_create_buffer[DType.int32](BATCH)
    var s_weights = ctx.enqueue_create_buffer[dtype](BATCH)

    buf.sample[BATCH](ctx, s_obs, s_act, s_rew, s_nobs, s_done, s_idx, s_weights)

    var r_obs = ctx.enqueue_create_host_buffer[dtype](BATCH * OBS)
    var r_rew = ctx.enqueue_create_host_buffer[dtype](BATCH)
    ctx.enqueue_copy(r_obs, s_obs)
    ctx.enqueue_copy(r_rew, s_rew)
    ctx.synchronize()

    # Check gradient pattern survived quantization
    var max_err: Float64 = 0.0
    var zero_count = 0
    var nan_count = 0
    for b in range(BATCH):
        for d in range(OBS):
            var v = Float64(r_obs[b * OBS + d])
            var expected = Float64(d) / Float64(OBS)
            if v != v:  # NaN check
                nan_count += 1
                continue
            if v == 0.0 and expected > 0.01:
                zero_count += 1
            var err = v - expected
            if err < 0:
                err = -err
            if err > max_err:
                max_err = err

    print("  Buffer size:", buf.size)
    print("  Max quantization error:", max_err, "(expected < 0.004)")
    print("  Unexpected zeros:", zero_count)
    print("  NaN values:", nan_count)
    print("  Reward[0]:", r_rew[0], "(expected 0.5)")
    print("  Obs[0][:4]:", r_obs[0], r_obs[1], r_obs[2], r_obs[3])
    print(
        "  Obs[0][-4:]:",
        r_obs[OBS - 4],
        r_obs[OBS - 3],
        r_obs[OBS - 2],
        r_obs[OBS - 1],
    )

    if nan_count > 0:
        print("  FAIL: NaN values in sampled observations!")
    elif zero_count > 10:
        print("  FAIL: Too many unexpected zeros — buffer likely corrupt!")
    elif max_err > 0.01:
        print("  FAIL: Quantization error too large!")
    else:
        print("  PASS: Large obs uint8 roundtrip OK")
