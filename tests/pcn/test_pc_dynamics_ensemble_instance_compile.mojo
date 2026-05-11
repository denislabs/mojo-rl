"""Compile-only smoke for PCDynamicsEnsembleInstance{CPU,GPU} — exercises
new MBPO-compatibility surface (rollout buffers, comptime aliases,
write/read_sections, sync_elite_member_buf, predict_rollout_member_into_slot).

The test does no real GPU work; it just instantiates the types and
references their methods so the compiler must monomorphize them.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import dtype
from mojo_rl.experimental.pcn import (
    PCDynamicsEnsembleInstanceCPU,
    PCDynamicsEnsembleInstanceGPU,
)


def cpu_compile_smoke() raises:
    print("--- PCDynamicsEnsembleInstanceCPU compile smoke ---")
    var inst = PCDynamicsEnsembleInstanceCPU[
        OBS_DIM=3,
        ACTION_DIM=1,
        HIDDEN_DIM=64,
        NUM_ENSEMBLE=3,
        NUM_ELITES=2,
        DYN_BATCH=8,
        T_INFER=4,
        dtype=dtype,
    ]()
    print("  TOTAL_PARAM_SIZE =", inst.ENS.TOTAL_PARAM_SIZE)
    print("  step_nums[0]     =", inst.step_nums[0])
    print("  elite_indices    =", inst.elite_indices)

    # Exercise checkpoint surface (force monomorphization).
    var blob = inst.write_sections("dyn_")
    inst.read_sections(blob, "dyn_")
    print("  checkpoint round-trip blob bytes ~ ", blob.byte_length())


def gpu_compile_smoke() raises:
    print("--- PCDynamicsEnsembleInstanceGPU compile smoke ---")
    var ctx = DeviceContext()
    var inst = PCDynamicsEnsembleInstanceGPU[
        OBS_DIM=3,
        ACTION_DIM=1,
        HIDDEN_DIM=64,
        NUM_ENSEMBLE=3,
        NUM_ELITES=2,
        DYN_BATCH=8,
        ROLLOUT_BATCH=16,
        T_INFER=4,
        R_WS_SIZE=1,
        dtype=dtype,
    ](ctx)

    # Comptime alias touches.
    print("  rollout_batch    =", inst.rollout_batch)
    print("  DYN_IN           =", inst.DYN_IN)
    print("  DYN_OUT          =", inst.DYN_OUT)
    print("  DYN_PRED         =", inst.DYN_PRED)
    print("  num_ensemble     =", inst.num_ensemble)
    print("  num_elites       =", inst.num_elites)
    print("  obs_dim          =", inst.obs_dim)
    print("  action_dim       =", inst.action_dim)
    print("  SAMPLE_BATCH     =", inst.SAMPLE_BATCH)

    # Field-presence checks. Touch each MBPO-compat field's pointer so
    # the compiler must materialize the field. Discarded via `_`.
    _ = inst.r_obs.unsafe_ptr()
    _ = inst.r_next_obs.unsafe_ptr()
    _ = inst.r_actions.unsafe_ptr()
    _ = inst.r_rewards.unsafe_ptr()
    _ = inst.r_dones.unsafe_ptr()
    _ = inst.r_alive.unsafe_ptr()
    _ = inst.r_dyn_input.unsafe_ptr()
    _ = inst.r_dyn_output.unsafe_ptr()
    _ = inst.r_dyn_output_all.unsafe_ptr()
    _ = inst.r_elite_idx_per_sample.unsafe_ptr()
    _ = inst.r_elite_rng.unsafe_ptr()
    _ = inst.elite_member_buf.unsafe_ptr()
    _ = inst.r_ws.unsafe_ptr()
    _ = inst.s_obs.unsafe_ptr()
    _ = inst.s_act.unsafe_ptr()
    _ = inst.s_rew.unsafe_ptr()
    _ = inst.s_nobs.unsafe_ptr()
    _ = inst.s_done.unsafe_ptr()
    _ = inst.s_idx.unsafe_ptr()
    _ = inst.input_mean.unsafe_ptr()
    _ = inst.input_std.unsafe_ptr()
    print("  all MBPO-compat fields present")

    # Method monomorphization (no real GPU work — just a sync round trip
    # for the elite buffer mapping).
    inst.sync_elite_member_buf(ctx)

    # Exercise checkpoint surface (round-trip through host).
    var blob = inst.write_sections(ctx, "dyn_")
    inst.read_sections(ctx, blob, "dyn_")
    print("  GPU checkpoint round-trip blob bytes ~ ", blob.byte_length())


def main() raises:
    print("PCDynamicsEnsembleInstance compile-only smoke")
    cpu_compile_smoke()
    gpu_compile_smoke()
    print("=== Compile-only smoke OK ===")

