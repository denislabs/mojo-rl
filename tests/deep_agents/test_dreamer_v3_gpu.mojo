"""Test DreamerV3 GPU state allocation, upload/download, and training step."""

from std.random import random_float64
from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.dreamer_v3 import DreamerV3Agent


fn main() raises:
    print("=" * 60)
    print("DreamerV3 GPU Tests")
    print("=" * 60)

    # Small config matching CPU test
    comptime OBS = 6
    comptime ACT = 2
    comptime B = 4
    comptime BL = 8
    comptime HORIZON = 5

    var agent = DreamerV3Agent[
        obs_dim=OBS,
        action_dim=ACT,
        deter_dim=32,
        hidden=16,
        stoch_dim=4,
        classes=4,
        units=16,
        num_bins=31,
        blocks=2,
        batch_size=B,
        batch_length=BL,
        imagine_horizon=HORIZON,
        buffer_capacity=1000,
    ]()

    var ctx = DeviceContext()

    # Test 1: GPU state allocation
    print("Test make_gpu_state...")
    var gpu_state = agent.make_gpu_state(ctx)
    ctx.synchronize()
    print("  PASS: GPU state allocated")

    # Test 2: Upload to GPU
    print("Test upload_to_gpu...")
    agent.upload_to_gpu(gpu_state, ctx)
    ctx.synchronize()
    print("  PASS: Weights uploaded")

    # Test 3: Download from GPU
    print("Test download_from_gpu...")
    agent.download_from_gpu(gpu_state, ctx)
    ctx.synchronize()
    print("  PASS: Weights downloaded")

    # Test 4: Fill buffer and run GPU train step
    print("Test do_gpu_train_step...")
    for ep in range(10):
        for step in range(20):
            var o = List[Scalar[dtype]](capacity=OBS)
            for i in range(OBS):
                o.append(Scalar[dtype](random_float64(-1.0, 1.0)))
            var a = List[Scalar[dtype]](capacity=ACT)
            for i in range(ACT):
                a.append(Scalar[dtype](random_float64(-1.0, 1.0)))
            var r = random_float64(-1.0, 1.0)
            var done = step == 19
            agent.observe(o, a, r, done)

    print("  Buffer size: " + String(agent.state.buffer.len()))
    print("  Buffer ready: " + String(agent.state.is_ready()))

    # Sample batch on CPU (pre-fill with zeros)
    var batch_obs = List[Scalar[DType.float32]](capacity=B * (BL + 1) * OBS)
    var batch_actions = List[Scalar[DType.float32]](capacity=B * BL * ACT)
    var batch_rewards = List[Scalar[DType.float32]](capacity=B * BL)
    var batch_dones = List[Scalar[DType.float32]](capacity=B * BL)
    for _ in range(B * (BL + 1) * OBS):
        batch_obs.append(Scalar[DType.float32](0))
    for _ in range(B * BL * ACT):
        batch_actions.append(Scalar[DType.float32](0))
    for _ in range(B * BL):
        batch_rewards.append(Scalar[DType.float32](0))
        batch_dones.append(Scalar[DType.float32](0))

    agent.state.buffer.sample_sequences[B, BL](
        batch_obs, batch_actions, batch_rewards, batch_dones,
    )

    print("  Batch sampled: obs=" + String(len(batch_obs))
        + " act=" + String(len(batch_actions))
        + " rew=" + String(len(batch_rewards)))

    # Run GPU train step
    agent.do_gpu_train_step(
        ctx, gpu_state,
        batch_obs, batch_actions, batch_rewards, batch_dones,
    )
    print("  Train step count: " + String(agent.train_step_count))
    print("  PASS: GPU train step completed")

    # Test 5: Second GPU train step
    print("Test second GPU train step...")
    agent.state.buffer.sample_sequences[B, BL](
        batch_obs, batch_actions, batch_rewards, batch_dones,
    )
    agent.do_gpu_train_step(
        ctx, gpu_state,
        batch_obs, batch_actions, batch_rewards, batch_dones,
    )
    print("  Train step count: " + String(agent.train_step_count))
    print("  PASS")

    # Final sync
    agent.download_from_gpu(gpu_state, ctx)
    ctx.synchronize()

    print("=" * 60)
    print("All GPU tests passed.")
    print("=" * 60)
