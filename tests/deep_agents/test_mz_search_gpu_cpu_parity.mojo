"""CPU ↔ GPU MuZero search parity — the vanilla device-search diagnostic.

Background: the fully-on-device vanilla driver (`run_muzero_selfplay_gpu_device`)
showed a pathological eval curve on the 60k CartPole run — greedy (NoNoise
planner) stuck at ~110-130 while the *noisy* training policy reached ~250.
Greedy < noisy is impossible for a healthy search, so either the GPU NoNoise
search path is defective or the GPU search differs from the (converged) CPU
search in some systematic way.

This test pins the two searches against each other under conditions where they
should agree almost exactly:

  * identical nets — GPU nets downloaded into CPU mirrors via the byte-exact
    `mz_sync_gpu_to_cpu` checkpoint round-trip;
  * no root noise — CPU `add_noise=False`, GPU `NoNoise` planner;
  * serial simulation — `BATCH_SIMS=1`, `VIRTUAL_LOSS=0` on both, so the tree
    grows deterministically with no batched-leaf ordering differences.

Both searches then expand the SAME deterministic tree (modulo f32-vs-f64 tree
arithmetic), so visit policies and root values must match closely. A systematic
kernel defect (reward sign, discount, min-max normalization, h⁻¹ decode, prior
softmax) shows up as a large divergence here. Also sanity-checks the GPU
Dirichlet planner against the GPU NoNoise planner (finite, sums to 1).

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_mz_search_gpu_cpu_parity.mojo
"""

from std.gpu.host import DeviceContext
from std.math import abs
from std.memory import alloc
from std.testing import assert_true
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import mptr
from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.deep_agents.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents.muzero.selfplay_gpu import mz_sync_gpu_to_cpu
from mojo_rl.deep_agents.zero.mcts_adapters_mz import (
    MZRepGPU, MZDynGPU, MZPredGPU,
)
from mojo_rl.deep_agents.zero.mcts_adapters_mz_cpu import (
    MZRepCPU, MZDynCPU, MZPredCPU,
)
from mojo_rl.planners.tree_search import (
    GenericCPUMCTS,
    GenericGPUMCTS,
    MuZeroPUCT,
    DirichletNoise,
    NoNoise,
    SinglePlayer,
)


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 32
    comptime BINS = 51
    comptime H = 64
    comptime NUM_SIMS = 32
    comptime MAX_NODES = 64
    comptime N_OBS = 4

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    var v_min = Scalar[DT](-20.0)
    var v_max = Scalar[DT](20.0)
    var gamma = 0.997

    var ctx = DeviceContext()

    # ── GPU nets + byte-exact CPU mirrors ──
    var rep_g = Rep.make["gpu", Kaiming](Optional(ctx))
    var dyn_g = Dyn.make["gpu", Kaiming](Optional(ctx))
    var pred_g = Pred.make["gpu", Kaiming](Optional(ctx))
    var rep_c = Rep.make["cpu", Kaiming]()
    var dyn_c = Dyn.make["cpu", Kaiming]()
    var pred_c = Pred.make["cpu", Kaiming]()
    mz_sync_gpu_to_cpu(rep_g, rep_c, ctx)
    mz_sync_gpu_to_cpu(dyn_g, dyn_c, ctx)
    mz_sync_gpu_to_cpu(pred_g, pred_c, ctx)

    # ── CPU search (serial, no noise at runtime) ──
    var cpu_mcts = GenericCPUMCTS[
        ACT, LATENT, NUM_SIMS, MAX_NODES,
        MuZeroPUCT[19652.0, 1.25], DirichletNoise[0.25, 0.25], SinglePlayer,
        1, 0,
    ](gamma=gamma)
    var rep_ac = MZRepCPU[OBS, LATENT, Rep](net=UnsafePointer(to=rep_c))
    var dyn_ac = MZDynCPU[LATENT, ACT, BINS, Dyn](
        net=UnsafePointer(to=dyn_c), v_min=v_min, v_max=v_max
    )
    var pred_ac = MZPredCPU[LATENT, ACT, BINS, Pred](
        net=UnsafePointer(to=pred_c), v_min=v_min, v_max=v_max
    )

    # ── GPU searches (serial): NoNoise + Dirichlet sanity instance ──
    var gpu_nonoise = GenericGPUMCTS[
        1, ACT, LATENT, BINS, MAX_NODES, NUM_SIMS, 1,
        MuZeroPUCT[19652.0, 1.25], NoNoise, SinglePlayer, 0, 0,
    ](ctx, gamma=gamma, v_min=Float64(v_min), v_max=Float64(v_max))
    var gpu_noisy = GenericGPUMCTS[
        1, ACT, LATENT, BINS, MAX_NODES, NUM_SIMS, 1,
        MuZeroPUCT[19652.0, 1.25], DirichletNoise[0.25, 0.25], SinglePlayer,
        0, 0,
    ](ctx, gamma=gamma, v_min=Float64(v_min), v_max=Float64(v_max))
    var rep_ag = MZRepGPU[OBS, LATENT, Rep].make(rep_g)
    var dyn_ag = MZDynGPU[LATENT, ACT, BINS, Dyn].make(dyn_g)
    var pred_ag = MZPredGPU[LATENT, ACT, BINS, Pred].make(pred_g)

    # ── batched instances (the 60k run's settings: BATCH_SIMS=8, VLOSS=3) ──
    # Serial parity exonerates the core kernels; this pair isolates the
    # batched-leaf/virtual-loss path, which is what BOTH collection and eval
    # used in the anomalous run.
    var cpu_b = GenericCPUMCTS[
        ACT, LATENT, NUM_SIMS, MAX_NODES,
        MuZeroPUCT[19652.0, 1.25], DirichletNoise[0.25, 0.25], SinglePlayer,
        8, 3,
    ](gamma=gamma)
    var gpu_b = GenericGPUMCTS[
        1, ACT, LATENT, BINS, MAX_NODES, NUM_SIMS, 8,
        MuZeroPUCT[19652.0, 1.25], NoNoise, SinglePlayer, 0, 3,
        UNSAFE_BATCHED=True,    # diagnostic: measuring the known bias
    ](ctx, gamma=gamma, v_min=Float64(v_min), v_max=Float64(v_max))

    var d_obs = ctx.enqueue_create_buffer[DT](OBS)
    var h_obs = mptr(alloc[Scalar[DT]](OBS))
    var h_pol = mptr(alloc[Scalar[DT]](ACT))
    var h_val = mptr(alloc[Scalar[DT]](1))

    # A handful of CartPole-shaped root observations.
    var test_obs = List[List[Float64]]()
    test_obs.append([0.0, 0.0, 0.0, 0.0])
    test_obs.append([0.05, 0.3, 0.04, -0.2])
    test_obs.append([-0.1, -0.5, -0.06, 0.4])
    test_obs.append([1.2, 1.0, 0.1, 0.8])

    var argmax_match = 0
    for i in range(N_OBS):
        var ob = test_obs[i].copy()

        # CPU reference (noise off)
        var cpol = cpu_mcts.search[
            type_of(rep_ac), type_of(dyn_ac), type_of(pred_ac)
        ](rep_ac, dyn_ac, pred_ac, ob, add_noise=False)
        var cval = cpu_mcts.root_value()

        # GPU NoNoise
        for j in range(OBS):
            h_obs[j] = Scalar[DT](ob[j])
        ctx.enqueue_copy(d_obs, h_obs)
        var obs_t = LayoutTensor[DT, Layout.row_major(1, OBS), MutAnyOrigin](
            mptr(d_obs.unsafe_ptr())
        )
        gpu_nonoise.search_gpu[
            type_of(rep_ag), type_of(dyn_ag), type_of(pred_ag)
        ](ctx, rep_ag, dyn_ag, pred_ag, obs_t, rng_seed=UInt32(7))
        ctx.enqueue_copy(h_pol, gpu_nonoise.policies_out)
        ctx.enqueue_copy(h_val, gpu_nonoise.root_value_out)
        ctx.synchronize()
        var gpol0 = Float64(h_pol[0])
        var gpol1 = Float64(h_pol[1])
        var gval = Float64(h_val[0])

        # GPU Dirichlet (sanity only)
        ctx.enqueue_copy(d_obs, h_obs)
        gpu_noisy.search_gpu[
            type_of(rep_ag), type_of(dyn_ag), type_of(pred_ag)
        ](ctx, rep_ag, dyn_ag, pred_ag, obs_t, rng_seed=UInt32(7))
        ctx.enqueue_copy(h_pol, gpu_noisy.policies_out)
        ctx.synchronize()
        var npol0 = Float64(h_pol[0])
        var npol1 = Float64(h_pol[1])

        # Batched pair (CPU noise-off vs GPU NoNoise, BATCH_SIMS=8/VLOSS=3)
        var cpol_b = cpu_b.search[
            type_of(rep_ac), type_of(dyn_ac), type_of(pred_ac)
        ](rep_ac, dyn_ac, pred_ac, ob, add_noise=False)
        var cval_b = cpu_b.root_value()
        ctx.enqueue_copy(d_obs, h_obs)
        gpu_b.search_gpu[
            type_of(rep_ag), type_of(dyn_ag), type_of(pred_ag)
        ](ctx, rep_ag, dyn_ag, pred_ag, obs_t, rng_seed=UInt32(7))
        ctx.enqueue_copy(h_pol, gpu_b.policies_out)
        ctx.enqueue_copy(h_val, gpu_b.root_value_out)
        ctx.synchronize()
        var bpol0 = Float64(h_pol[0])
        var bpol1 = Float64(h_pol[1])
        var bval = Float64(h_val[0])

        var c_arg = 0 if cpol[0] >= cpol[1] else 1
        var g_arg = 0 if gpol0 >= gpol1 else 1
        if c_arg == g_arg:
            argmax_match += 1

        print("obs", i)
        print("  cpu  pol", cpol[0], cpol[1], "root_v", cval)
        print("  gpu  pol", gpol0, gpol1, "root_v", gval, "(NoNoise)")
        print("  gpu  pol", npol0, npol1, "(Dirichlet)")
        print("  |dv|", abs(cval - gval), "|dp0|", abs(cpol[0] - gpol0))
        print("  cpuB pol", cpol_b[0], cpol_b[1], "root_v", cval_b,
              "(batch8/vl3)")
        print("  gpuB pol", bpol0, bpol1, "root_v", bval, "(batch8/vl3)")
        print("  |dvB|", abs(cval_b - bval), "|dp0B|", abs(cpol_b[0] - bpol0))

        # Hard checks: finite, normalized.
        assert_true(gval == gval, "GPU NoNoise root value NaN")
        assert_true(
            gpol0 + gpol1 > 0.99 and gpol0 + gpol1 < 1.01,
            "GPU NoNoise policy not normalized",
        )
        assert_true(
            npol0 + npol1 > 0.99 and npol0 + npol1 < 1.01,
            "GPU Dirichlet policy not normalized",
        )
        # Parity checks (deterministic serial trees, f32-vs-f64 drift only).
        assert_true(
            abs(cval - gval) < 0.5,
            "CPU vs GPU root value diverges beyond fp drift",
        )
        assert_true(
            abs(cpol[0] - gpol0) < 0.2,
            "CPU vs GPU visit policy diverges beyond fp drift",
        )

    assert_true(
        argmax_match >= N_OBS - 1,
        "CPU vs GPU argmax disagrees on multiple roots",
    )

    h_obs.free(); h_pol.free(); h_val.free()
    _ = rep_c^; _ = dyn_c^; _ = pred_c^
    _ = rep_g^; _ = dyn_g^; _ = pred_g^
    print("MuZero CPU↔GPU search parity: OK")
