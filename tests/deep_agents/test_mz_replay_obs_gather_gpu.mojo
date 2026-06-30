"""Does the GPU obs-gather assemble the TRAINING obs batch correctly?

The MuZero training step trains on obs gathered from the replay ring by a GPU
kernel (3 non-owning DeviceBuffer LayoutTensor operands, inferred origin — the
documented GPU-miscompile footgun). If that gather corrupts/misaligns the obs on
NVIDIA, the net trains on garbage and never learns (loss flat, no promotions),
even though the reward TARGETS are correct — matching the regression exactly.

Store ONE episode with POSITION-ENCODED obs (obs[s][j] = f(s,j)), run the real
sample+gather, download the gathered obs, and check each row equals the obs
stored at the slot the sampler picked (h_slots).

  worst err ~0   → obs gather is correct; look elsewhere
  worst err big  → GPU obs gather is corrupting/misaligning training obs = the bug
Run on NVIDIA AND apple; apple is the baseline.
"""

from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.zero.prioritized_sequence_replay_mcts import (
    PrioritizedMCTSSequenceReplay,
)


def _enc(s: Int, j: Int) -> Scalar[DT]:
    return Scalar[DT](0.1) * Scalar[DT](((s * 7 + j * 3) % 13) - 6)


def main() raises:
    comptime OBS = 8
    comptime ACT = 7
    comptime CAP = 1000
    comptime L = 20
    comptime B = 16
    comptime K = 5
    comptime N = 8
    comptime M = (K + 1) * B
    var ctx = DeviceContext()
    var rb = PrioritizedMCTSSequenceReplay[OBS, ACT, CAP](ctx)

    var ep_obs = List[Scalar[DT]](length=L * OBS, fill=0)
    for s in range(L):
        for j in range(OBS):
            ep_obs[s * OBS + j] = _enc(s, j)
    var ep_act = List[Scalar[DT]](length=L, fill=0)
    var ep_rew = List[Scalar[DT]](length=L, fill=0)
    var ep_pol = List[Scalar[DT]](length=L * ACT, fill=Scalar[DT](1.0 / ACT))
    var ep_val = List[Scalar[DT]](length=L, fill=0)
    var ep_tp = List[Scalar[DT]](length=L, fill=0)
    var ep_legal = List[Scalar[DT]](length=L * ACT, fill=1)
    rb.store_episode(ep_obs, ep_act, ep_rew, ep_pol, ep_val, ep_tp, ep_legal, L)

    var out_obs = ctx.enqueue_create_buffer[DT](M * OBS)
    var d_slots = ctx.enqueue_create_buffer[DType.int32](M)
    var h_slots = List[Int32](length=M, fill=0)
    var t_act = List[Scalar[DT]](length=K * B, fill=0)
    var t_pol = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    var t_val = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    var t_rew = List[Scalar[DT]](length=K * B, fill=0)
    var t_isw = List[Scalar[DT]](length=B, fill=0)
    var t_slots = List[Int](length=B, fill=0)

    rb.sample_training_batch_seq_per_gpu[B, K, N](
        ctx, Scalar[DT](1.0), out_obs, d_slots, h_slots,
        t_act, t_pol, t_val, t_rew, t_isw, t_slots,
    )

    # download the gathered obs slab
    var got = List[Scalar[DT]](length=M * OBS, fill=0)
    ctx.enqueue_copy(got.unsafe_ptr(), out_obs)
    ctx.synchronize()

    var worst = Scalar[DT](0)
    var nbad = 0
    for r in range(M):
        var slot = Int(h_slots[r])   # one episode @ ep_start 0 -> slot == position
        for j in range(OBS):
            var exp = _enc(slot, j)
            var err = abs(got[r * OBS + j] - exp)
            if err > Scalar[DT](1e-5):
                nbad += 1
                if nbad <= 8:
                    print("  MISMATCH row", r, "slot", slot, "j", j,
                          "got", got[r * OBS + j], "exp", exp)
            if err > worst:
                worst = err
    print("worst obs-gather err =", worst, " bad entries:", nbad, "/", M * OBS)
    if nbad > 0:
        print(">>> GPU OBS GATHER CORRUPTS/MISALIGNS TRAINING OBS = the bug <<<")
    else:
        print(">>> obs gather correct on this backend <<<")
