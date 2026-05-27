"""BatchedEnv trait + BatchedCpuEnv adapter — Tier-2 prototype.

The Tier-1 unified drivers had to remain two functions because the env
interaction APIs differ structurally between CPU envs
(`BoxContinuousActionEnv.step_continuous_vec` returns host Lists) and
GPU envs (`GPUContinuousEnv.step_kernel_gpu` writes into device
buffers). Tier-2 lifts a uniform `BatchedEnv` trait over both:

  comptime ENV_TARGET: StaticString   # "cpu" or "gpu"
  comptime OBS_DIM: Int
  comptime ACT_DIM: Int

  def reset_batch[N_ENVS](
      mut self, ctx, obs_ptr, rng_seed,
  ) raises
  def step_batch[N_ENVS](
      mut self, ctx, action_ptr, obs_ptr, reward_ptr, done_ptr,
      rng_seed,
  ) raises
  def selective_reset_batch[N_ENVS](
      mut self, ctx, obs_ptr, dones_ptr, rng_seed,
  ) raises

Pointer semantics: target-side scalars, all sized N_ENVS × {dim}.
For ENV_TARGET="cpu" the caller passes host pointers; for "gpu",
device pointers. `ctx: Optional[DeviceContext]` is `None` for CPU
envs and required for GPU envs.

`BatchedCpuEnv[E: BoxContinuousActionEnv & Copyable & Movable, N_ENVS]`
adapter: wraps any CPU env conforming to the existing trait via an
`InlineArray[E, N_ENVS]` of independent env instances (verified by the
Tier-2 spike: each instance holds independent state). Implements
`BatchedEnv` by looping N_ENVS times inside each method and converting
E.dtype ↔ DT at the boundary. This is the "every env gets batched-CPU
for free via an adapter" piece — no per-env refactor required.

GPU adapter is intentionally NOT in this file. The GPU env kernels
(E.step_kernel_gpu, E.reset_kernel_gpu, etc.) expect
`mut DeviceBuffer[dtype]` parameters, not raw pointers — wrapping a
raw pointer in a synthetic DeviceBuffer isn't safely expressible in
Mojo nightly. The existing `run_offpolicy_train_unified_gpu_env`
driver (Phase 3.5) continues to handle the GPU env case; a BatchedEnv-
conforming GPU adapter is Tier-3 work.
"""

from std.gpu.host import DeviceContext

from ..constants import DT
from mojo_rl.core.env_traits import BoxContinuousActionEnv


# ──────────────────────────────────────────────────────────────────────
# BatchedEnv trait — uniform env-interaction surface.
# ──────────────────────────────────────────────────────────────────────


trait BatchedEnv(Movable & ImplicitlyDestructible):
    """Uniform N_ENVS-batched env interaction surface. Pointer args
    are always sized N_ENVS × {dim} and live on ENV_TARGET.

    `rng_seed` is consumed deterministically per call. CPU envs may
    ignore it (they typically rely on the std lib's global PRNG state
    via `random_float64`) or fold it in for testability; GPU envs use
    it to seed Philox kernels."""

    comptime ENV_TARGET: StaticString
    comptime OBS_DIM: Int
    comptime ACT_DIM: Int

    def reset_batch[
        BATCH: Int
    ](
        mut self,
        ctx: Optional[DeviceContext],
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rng_seed: UInt64,
    ) raises:
        """Initial reset for all N envs. Writes N_ENVS * OBS_DIM
        scalars into `obs_ptr`."""
        ...

    def step_batch[
        BATCH: Int
    ](
        mut self,
        ctx: Optional[DeviceContext],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rng_seed: UInt64,
    ) raises:
        """Step all N envs in parallel. `action_ptr` is read; `obs_ptr`
        is overwritten with next_obs in place; `reward_ptr` and
        `done_ptr` are written as outputs (each N_ENVS scalars)."""
        ...

    def selective_reset_batch[
        BATCH: Int
    ](
        mut self,
        ctx: Optional[DeviceContext],
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        dones_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rng_seed: UInt64,
    ) raises:
        """Reset only envs where `dones_ptr[i] > 0.5`. Refreshes
        `obs_ptr` for reset envs; leaves other slots untouched."""
        ...


# ──────────────────────────────────────────────────────────────────────
# BatchedCpuEnv[E, N_ENVS] — CPU env adapter.
# ──────────────────────────────────────────────────────────────────────


struct BatchedCpuEnv[
    E: BoxContinuousActionEnv & Copyable & Movable,
    N_ENVS: Int,
    OBS_DIM_: Int,
    ACT_DIM_: Int,
](BatchedEnv):
    """Adapter that holds N independent instances of a CPU env `E`
    and dispatches the BatchedEnv surface by looping internally.

    Per-env state independence is verified by the Tier-2 viability
    spike (envs stepped different numbers of times produce distinct
    observations).

    Construction:
        var template = PendulumEnv[DT]()  # OBS=3, ACT=1
        var batched = BatchedCpuEnv[PendulumEnv[DT], 4, 3, 1](template)

    `OBS_DIM_` / `ACT_DIM_` are explicit comptime params because
    `BoxContinuousActionEnv` exposes obs/action dimensions only as
    runtime methods (`obs_dim()` / `action_dim()`), not comptime
    members — we need them comptime to lay out the pointer slabs.

    The template-clone pattern (caller passes one configured env, the
    wrapper clones it `N_ENVS` times via `InlineArray(fill=template)`)
    avoids requiring E to conform to `Defaultable` — works with any
    `BoxContinuousActionEnv & Copyable & Movable` env.
    """

    comptime ENV_TARGET: StaticString = "cpu"
    comptime OBS_DIM: Int = Self.OBS_DIM_
    comptime ACT_DIM: Int = Self.ACT_DIM_

    var envs: InlineArray[Self.E, Self.N_ENVS]

    # Pre-allocated host scratch for the per-env action List we feed
    # into `step_continuous_vec` (avoids allocating a new List every
    # step × env). Sized ACT_DIM (single env's action), not N×ACT.
    var _action_scratch: List[Scalar[Self.E.dtype]]

    def __init__(out self, template: Self.E):
        self.envs = InlineArray[Self.E, Self.N_ENVS](fill=template)
        self._action_scratch = List[Scalar[Self.E.dtype]](
            capacity=Self.ACT_DIM
        )
        for _ in range(Self.ACT_DIM):
            self._action_scratch.append(Scalar[Self.E.dtype](0.0))

    def reset_batch[
        BATCH: Int
    ](
        mut self,
        ctx: Optional[DeviceContext],
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rng_seed: UInt64,
    ) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "BatchedCpuEnv: reset_batch BATCH must match struct param"
        )
        _ = ctx
        _ = rng_seed
        for env_idx in range(Self.N_ENVS):
            var obs_list = self.envs[env_idx].reset_obs_list()
            for d in range(Self.OBS_DIM):
                obs_ptr[env_idx * Self.OBS_DIM + d] = Scalar[DT](obs_list[d])

    def step_batch[
        BATCH: Int
    ](
        mut self,
        ctx: Optional[DeviceContext],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rng_seed: UInt64,
    ) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "BatchedCpuEnv: step_batch BATCH must match struct param"
        )
        _ = ctx
        _ = rng_seed
        for env_idx in range(Self.N_ENVS):
            # Stage this env's slice of actions into the per-env List
            # (E.dtype may differ from DT).
            for j in range(Self.ACT_DIM):
                self._action_scratch[j] = Scalar[Self.E.dtype](
                    action_ptr[env_idx * Self.ACT_DIM + j]
                )
            var step_res = self.envs[env_idx].step_continuous_vec[
                Self.E.dtype
            ](self._action_scratch)
            var nxt = step_res[0].copy()
            var reward = step_res[1]
            var done = step_res[2]
            for d in range(Self.OBS_DIM):
                obs_ptr[env_idx * Self.OBS_DIM + d] = Scalar[DT](nxt[d])
            reward_ptr[env_idx] = Scalar[DT](reward)
            done_ptr[env_idx] = Scalar[DT](1.0) if done else Scalar[DT](0.0)

    def selective_reset_batch[
        BATCH: Int
    ](
        mut self,
        ctx: Optional[DeviceContext],
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        dones_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rng_seed: UInt64,
    ) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "BatchedCpuEnv: selective_reset_batch BATCH must match struct param"
        )
        _ = ctx
        _ = rng_seed
        for env_idx in range(Self.N_ENVS):
            if dones_ptr[env_idx] > Scalar[DT](0.5):
                var obs_list = self.envs[env_idx].reset_obs_list()
                for d in range(Self.OBS_DIM):
                    obs_ptr[env_idx * Self.OBS_DIM + d] = Scalar[DT](
                        obs_list[d]
                    )
