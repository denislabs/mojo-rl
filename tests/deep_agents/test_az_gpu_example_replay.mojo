"""Gate for GpuMCTSExampleReplay — store → flush(z in-kernel) → device sample.

N_ENVS=1 game of 2 moves, then a win-flush, then sample B=4 with replacement:
  step0: obs=[1,1], pol=[0.3,0.7]   step1: obs=[2,2], pol=[0.4,0.6]
  win, L=2  ⇒  z_k = +1 if (L-1-k) even else -1  ⇒  z_0=-1 (k=0→L-1-k=1 odd),
                                                     z_1=+1 (k=1→0 even)
  ring[0] = (obs[1,1], [0.3,0.7,-1]),  ring[1] = (obs[2,2], [0.4,0.6,+1])

Asserts every sampled example is one of the two stored rows AND obs↔target stay
paired (the same-seed gather picks the same slot for obs and tgt) with the right
z — i.e. store + flush + device gather are all correct.

Run (Apple Metal): pixi run -e apple mojo run -I . \
    tests/deep_agents/test_az_gpu_example_replay.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.zero.gpu_example_replay import GpuMCTSExampleReplay


def _fill(ctx: DeviceContext, mut buf: DeviceBuffer[DT], vals: List[Float64]) raises:
    var h = ctx.enqueue_create_host_buffer[DT](len(vals))
    ctx.synchronize()
    for i in range(len(vals)):
        h[i] = Scalar[DT](vals[i])
    ctx.enqueue_copy(buf, h)
    ctx.synchronize()


def main() raises:
    comptime OBS = 2
    comptime ACT = 2
    comptime CAP = 16
    comptime N_ENVS = 1
    comptime MAX_TRAJ = 4
    comptime B = 4
    var ctx = DeviceContext()

    var replay = GpuMCTSExampleReplay[OBS, ACT, CAP, N_ENVS, MAX_TRAJ](ctx)

    var obs_dev = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var pol_dev = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var done_dev = ctx.enqueue_create_buffer[DT](N_ENVS)
    var rew_dev = ctx.enqueue_create_buffer[DT](N_ENVS)

    # move 0
    _fill(ctx, obs_dev, [1.0, 1.0])
    _fill(ctx, pol_dev, [0.3, 0.7])
    replay.record_step_gpu(obs_dev, pol_dev)
    # move 1
    _fill(ctx, obs_dev, [2.0, 2.0])
    _fill(ctx, pol_dev, [0.4, 0.6])
    replay.record_step_gpu(obs_dev, pol_dev)
    # finish: env 0 wins
    _fill(ctx, done_dev, [1.0])
    _fill(ctx, rew_dev, [1.0])
    replay.flush_finished_gpu(done_dev, rew_dev)

    assert_true(len(replay) == 2, "ring size should be 2, got " + String(len(replay)))

    var obs_out = Tensor.alloc_gpu(ctx, B * OBS)
    var tgt_out = Tensor.alloc_gpu(ctx, B * (ACT + 1))
    replay.sample_batch_gpu[B](obs_out, tgt_out)
    obs_out.download(ctx)
    tgt_out.download(ctx)

    var ok = True
    var saw1 = False
    var saw2 = False
    for b in range(B):
        var o0 = Float64(obs_out.data[b * OBS + 0])
        var p0 = Float64(tgt_out.data[b * (ACT + 1) + 0])
        var p1 = Float64(tgt_out.data[b * (ACT + 1) + 1])
        var z = Float64(tgt_out.data[b * (ACT + 1) + 2])
        if o0 > 1.5:  # example 1: obs=[2,2], [0.4,0.6,+1]
            saw2 = True
            if not (abs(p0 - 0.4) < 1e-4 and abs(p1 - 0.6) < 1e-4 and abs(z - 1.0) < 1e-4):
                ok = False
        else:  # example 0: obs=[1,1], [0.3,0.7,-1]
            saw1 = True
            if not (abs(p0 - 0.3) < 1e-4 and abs(p1 - 0.7) < 1e-4 and abs(z + 1.0) < 1e-4):
                ok = False
        print("b", b, " obs0", o0, " pol", p0, p1, " z", z)

    assert_true(ok, "a sampled example had mismatched obs/target/z pairing")
    assert_true(saw1 or saw2, "no examples sampled")
    print("AZ GPU example replay: OK")
