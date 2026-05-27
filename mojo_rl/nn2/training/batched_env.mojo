"""BatchedEnv trait + BatchedCpuEnv / BatchedGpuEnv adapters — Tier-3.

The Tier-1 drivers had to remain two functions because env
APIs differ structurally — CPU envs return host Lists, GPU envs take
DeviceBuffer kernel args. Tier-3 lifts a uniform `BatchedEnv` trait
over both via the **env-owns-buffers** pattern:

  trait BatchedEnv:
      comptime ENV_TARGET: StaticString
      comptime OBS_DIM: Int
      comptime ACT_DIM: Int

      def reset_batch[BATCH](mut self, ctx, rng_seed) raises
      def step_batch[BATCH](mut self, ctx, rng_seed) raises
      def selective_reset_batch[BATCH](mut self, ctx, rng_seed) raises

      def obs_ptr(self)    -> UnsafePointer[Scalar[DT], MutAnyOrigin]
      def action_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]
      def reward_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]
      def done_ptr(self)   -> UnsafePointer[Scalar[DT], MutAnyOrigin]

The env adapter owns the obs/action/reward/done buffers (host List for
CPU, DeviceBuffer for GPU). The driver writes actions into
`env.action_ptr()`, calls `env.step_batch()`, then reads
obs/reward/done via the accessor pointers.

Two adapters:

  - `BatchedCpuEnv[E, N_ENVS, OBS_DIM, ACT_DIM]` wraps any
    `BoxContinuousActionEnv & Copyable & Movable` env. Holds
    `InlineArray[E, N_ENVS]` of independent env instances.
  - `BatchedGpuEnv[E, N_ENVS, OBS_DIM, ACT_DIM, STATE_SIZE]` wraps any
    `GPUContinuousEnv`. Holds internal `DeviceBuffer` fields for
    state/obs/action/reward/done/terminated; dispatches the env's
    static `*_kernel_gpu` methods.

For consumers needing the underlying `DeviceBuffer` (e.g. to call
trainer.record_batch_gpu which takes DeviceBuffer args), the driver
reconstructs non-owning views via
`DeviceBuffer[DT](ctx, env.obs_ptr(), N*OBS, owning=False)`.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ..constants import DT
from mojo_rl.core.env_traits import BoxContinuousActionEnv, GPUContinuousEnv


# ──────────────────────────────────────────────────────────────────────
# BatchedEnv trait — uniform env-interaction surface.
# ──────────────────────────────────────────────────────────────────────


trait BatchedEnv(Movable & ImplicitlyDestructible):
    """Uniform N_ENVS-batched env surface. Env owns its
    obs/action/reward/done buffers internally; driver reads/writes via
    pointer accessors. Method comptime is `BATCH` (not `N_ENVS`) so
    impls whose struct already has an N_ENVS comptime can conform
    without name shadowing."""

    comptime ENV_TARGET: StaticString
    comptime OBS_DIM: Int
    comptime ACT_DIM: Int

    def reset_batch[BATCH: Int](
        mut self,
        ctx: Optional[DeviceContext],
        rng_seed: UInt64,
    ) raises:
        """Initial reset for all N envs. Writes the env's internal
        obs buffer; subsequent `obs_ptr()` reads see the new obs."""
        ...

    def step_batch[BATCH: Int](
        mut self,
        ctx: Optional[DeviceContext],
        rng_seed: UInt64,
    ) raises:
        """Step all N envs. Reads the env's `action_ptr()`; overwrites
        `obs_ptr()`, `reward_ptr()`, `done_ptr()` in place."""
        ...

    def selective_reset_batch[BATCH: Int](
        mut self,
        ctx: Optional[DeviceContext],
        rng_seed: UInt64,
    ) raises:
        """Reset only envs where the last `step_batch` wrote
        `done > 0.5`; obs for reset envs gets refreshed. Reads
        `done_ptr()`."""
        ...

    # ─── Pointer accessors — same return type for CPU and GPU ────────

    def obs_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to the [N_ENVS, OBS_DIM] obs slab on ENV_TARGET."""
        ...

    def action_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to the [N_ENVS, ACT_DIM] action slab. Driver
        writes action into this slab before calling step_batch."""
        ...

    def reward_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to the [N_ENVS] reward slab."""
        ...

    def done_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to the [N_ENVS] done slab (1.0 if done else 0.0)."""
        ...


# ──────────────────────────────────────────────────────────────────────
# BatchedCpuEnv[E, N_ENVS, OBS, ACT] — CPU env adapter.
# ──────────────────────────────────────────────────────────────────────


struct BatchedCpuEnv[
    E: BoxContinuousActionEnv & Copyable & Movable,
    N_ENVS: Int,
    OBS_DIM_: Int,
    ACT_DIM_: Int,
](BatchedEnv):
    """Wraps N independent instances of a CPU env via
    `InlineArray[E, N_ENVS]` and owns the obs/action/reward/done host
    buffers. Per-env state independence is verified by the Tier-2
    viability spike.

    Construction:
        var template = PendulumEnv[DT]()  # OBS=3, ACT=1
        var batched = BatchedCpuEnv[PendulumEnv[DT], 4, 3, 1](template)

    `OBS_DIM_` / `ACT_DIM_` are explicit comptime params because
    `BoxContinuousActionEnv` exposes obs/action dimensions only as
    runtime methods, not comptime members — we need them comptime to
    size the internal buffers.
    """

    comptime ENV_TARGET: StaticString = "cpu"
    comptime OBS_DIM: Int = Self.OBS_DIM_
    comptime ACT_DIM: Int = Self.ACT_DIM_

    var envs: InlineArray[Self.E, Self.N_ENVS]

    # Internally-owned host buffers — driver reads/writes via accessors.
    var _obs: List[Scalar[DT]]
    var _action: List[Scalar[DT]]
    var _reward: List[Scalar[DT]]
    var _done: List[Scalar[DT]]

    # Pre-allocated host scratch for the per-env action List we feed
    # into `step_continuous_vec` (avoids allocating a new List every
    # step × env). E.dtype may differ from DT.
    var _action_scratch: List[Scalar[Self.E.dtype]]

    def __init__(out self, template: Self.E):
        self.envs = InlineArray[Self.E, Self.N_ENVS](fill=template)
        self._obs = List[Scalar[DT]](
            length=Self.N_ENVS * Self.OBS_DIM, fill=Scalar[DT](0.0),
        )
        self._action = List[Scalar[DT]](
            length=Self.N_ENVS * Self.ACT_DIM, fill=Scalar[DT](0.0),
        )
        self._reward = List[Scalar[DT]](
            length=Self.N_ENVS, fill=Scalar[DT](0.0),
        )
        self._done = List[Scalar[DT]](
            length=Self.N_ENVS, fill=Scalar[DT](0.0),
        )
        self._action_scratch = List[Scalar[Self.E.dtype]](
            capacity=Self.ACT_DIM,
        )
        for _ in range(Self.ACT_DIM):
            self._action_scratch.append(Scalar[Self.E.dtype](0.0))

    def reset_batch[BATCH: Int](
        mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,
    ) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "BatchedCpuEnv: reset_batch BATCH must match struct param"
        )
        _ = ctx
        _ = rng_seed
        for env_idx in range(Self.N_ENVS):
            var obs_list = self.envs[env_idx].reset_obs_list()
            for d in range(Self.OBS_DIM):
                self._obs[env_idx * Self.OBS_DIM + d] = Scalar[DT](
                    obs_list[d]
                )

    def step_batch[BATCH: Int](
        mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,
    ) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "BatchedCpuEnv: step_batch BATCH must match struct param"
        )
        _ = ctx
        _ = rng_seed
        for env_idx in range(Self.N_ENVS):
            # Stage this env's action lane (E.dtype) from the host buf.
            for j in range(Self.ACT_DIM):
                self._action_scratch[j] = Scalar[Self.E.dtype](
                    self._action[env_idx * Self.ACT_DIM + j]
                )
            var step_res = self.envs[env_idx].step_continuous_vec[
                Self.E.dtype
            ](self._action_scratch)
            var nxt = step_res[0].copy()
            var reward = step_res[1]
            var done = step_res[2]
            for d in range(Self.OBS_DIM):
                self._obs[env_idx * Self.OBS_DIM + d] = Scalar[DT](nxt[d])
            self._reward[env_idx] = Scalar[DT](reward)
            self._done[env_idx] = (
                Scalar[DT](1.0) if done else Scalar[DT](0.0)
            )

    def selective_reset_batch[BATCH: Int](
        mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,
    ) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "BatchedCpuEnv: selective_reset_batch BATCH must match struct param"
        )
        _ = ctx
        _ = rng_seed
        for env_idx in range(Self.N_ENVS):
            if self._done[env_idx] > Scalar[DT](0.5):
                var obs_list = self.envs[env_idx].reset_obs_list()
                for d in range(Self.OBS_DIM):
                    self._obs[env_idx * Self.OBS_DIM + d] = Scalar[DT](
                        obs_list[d]
                    )

    def obs_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._obs.unsafe_ptr()
        )

    def action_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._action.unsafe_ptr()
        )

    def reward_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._reward.unsafe_ptr()
        )

    def done_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._done.unsafe_ptr()
        )


# ──────────────────────────────────────────────────────────────────────
# BatchedGpuEnv[E, N_ENVS, OBS, ACT, STATE_SIZE] — GPU env adapter.
# ──────────────────────────────────────────────────────────────────────


struct BatchedGpuEnv[
    E: GPUContinuousEnv,
    N_ENVS: Int,
    OBS_DIM_: Int,
    ACT_DIM_: Int,
](BatchedEnv):
    """Wraps a `GPUContinuousEnv` and owns the per-step device buffers
    (state, obs, action, reward, done, terminated). Dispatches the
    env's static `*_kernel_gpu` methods inside the BatchedEnv methods.

    Construction:
        var ctx = DeviceContext()
        var env = BatchedGpuEnv[PendulumV2[DT], 4, 3, 1](ctx)

    `STATE_SIZE` is derived from `E.STATE_SIZE` — the trait surface
    exposes it directly, so taking it as a separate comptime would
    invite the caller-passes-wrong-value bug (which we hit during
    Tier-3 bring-up when STATE_SIZE=3 was passed for PendulumV2
    whose actual STATE_SIZE=6, sizing the state buffer too small
    and making the kernel read garbage step-counters).

    The adapter is intentionally `E`-agnostic for its buffers — only
    the kernel dispatches use `E`. Driver bounds on `BatchedEnv`
    (uniform trait); the driver reconstructs `DeviceBuffer` views
    from `obs_ptr()` etc. via `(ctx, ptr, size, owning=False)` to
    pass to trainer.record_batch_gpu.
    """

    comptime ENV_TARGET: StaticString = "gpu"
    comptime OBS_DIM: Int = Self.OBS_DIM_
    comptime ACT_DIM: Int = Self.ACT_DIM_
    comptime STATE_SIZE: Int = Self.E.STATE_SIZE

    var _states: DeviceBuffer[DT]
    var _obs: DeviceBuffer[DT]
    var _action: DeviceBuffer[DT]
    var _reward: DeviceBuffer[DT]
    var _done: DeviceBuffer[DT]
    var _terminated: DeviceBuffer[DT]

    def __init__(out self, ctx: DeviceContext) raises:
        self._states = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.STATE_SIZE
        )
        self._obs = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.OBS_DIM
        )
        self._action = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.ACT_DIM
        )
        self._reward = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._done = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._terminated = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        ctx.enqueue_memset(self._action, 0)
        ctx.enqueue_memset(self._reward, 0)
        ctx.enqueue_memset(self._done, 0)
        ctx.enqueue_memset(self._terminated, 0)

    def reset_batch[BATCH: Int](
        mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,
    ) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "BatchedGpuEnv: reset_batch BATCH must match struct param"
        )
        if not ctx:
            raise Error("BatchedGpuEnv.reset_batch: ctx required")
        var c = ctx.value()
        Self.E.reset_kernel_gpu[Self.N_ENVS, Self.STATE_SIZE](
            c, self._states, rng_seed=rng_seed,
        )
        Self.E.extract_obs_kernel_gpu[
            Self.N_ENVS, Self.STATE_SIZE, Self.OBS_DIM
        ](c, self._states, self._obs)

    def step_batch[BATCH: Int](
        mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,
    ) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "BatchedGpuEnv: step_batch BATCH must match struct param"
        )
        if not ctx:
            raise Error("BatchedGpuEnv.step_batch: ctx required")
        var c = ctx.value()
        Self.E.step_kernel_gpu[
            Self.N_ENVS, Self.STATE_SIZE, Self.OBS_DIM, Self.ACT_DIM
        ](
            c,
            self._states,
            self._action,
            self._reward,
            self._done,
            self._terminated,
            self._obs,
            rng_seed=rng_seed,
        )

    def selective_reset_batch[BATCH: Int](
        mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,
    ) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "BatchedGpuEnv: selective_reset_batch BATCH must match struct param"
        )
        if not ctx:
            raise Error("BatchedGpuEnv.selective_reset_batch: ctx required")
        var c = ctx.value()
        Self.E.selective_reset_kernel_gpu[
            Self.N_ENVS, Self.STATE_SIZE
        ](c, self._states, self._done, rng_seed=rng_seed)
        Self.E.extract_obs_kernel_gpu[
            Self.N_ENVS, Self.STATE_SIZE, Self.OBS_DIM
        ](c, self._states, self._obs)

    def obs_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._obs.unsafe_ptr()
        )

    def action_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._action.unsafe_ptr()
        )

    def reward_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._reward.unsafe_ptr()
        )

    def done_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._done.unsafe_ptr()
        )
