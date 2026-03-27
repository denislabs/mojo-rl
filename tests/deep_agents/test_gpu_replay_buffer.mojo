"""Minimal test to isolate GPU replay buffer crash."""

from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.core.replay import GPUReplayBuffer
from mojo_rl.nn.training import GPUNetworkPair
from mojo_rl.nn.model import Linear
from mojo_rl.nn.optimizer import Adam


def main() raises:
    print("Creating DeviceContext...")
    with DeviceContext() as ctx:
        print("DeviceContext created")

        print("Allocating small DeviceBuffer...")
        var buf1 = ctx.enqueue_create_buffer[dtype](1024)
        print("buf1 allocated")

        print("Allocating second buffer...")
        var buf2 = ctx.enqueue_create_buffer[dtype](1024)
        print("buf2 allocated")

        ctx.synchronize()
        print("Sync done")

        print("Creating GPUReplayBuffer...")

        var replay = GPUReplayBuffer[1000, 17, 6](ctx)
        print("GPUReplayBuffer created")

        ctx.synchronize()
        print("Sync done after replay buffer")

        print("Creating GPUNetworkPair...")

        var pair = GPUNetworkPair[Linear[4, 2], Adam[0.001]](ctx)
        print("GPUNetworkPair created")

        ctx.synchronize()
        print("All GPU allocations successful")

    print("DeviceContext destroyed cleanly")
