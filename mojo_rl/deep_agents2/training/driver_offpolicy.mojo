"""Off-policy training + eval drivers — Tier-1 + Tier-3.

Two training driver functions covering all useful (env_target,
train_target, N_ENVS) combinations:

  env_target | train_target | N_ENVS | driver
  -----------|--------------|--------|----------------------------------
  cpu        | cpu          | >=1    | run_offpolicy_train_batched
  gpu        | gpu          | >=1    | run_offpolicy_train_batched
  cpu        | gpu          | 1      | run_offpolicy_train

Plus one eval driver `run_offpolicy_eval` that replaces the
legacy `run_offpolicy_eval_cpu` / `run_offpolicy_eval_gpu` split — the
trainer dispatches CPU vs GPU internally inside
`select_greedy_action`.

The (env=gpu, train=cpu) combination is omitted as degenerate
(D2H every obs back to CPU for training — never useful in practice).
Batched cross-target (cpu env, gpu train, N>1) is reachable in
principle by extending `run_offpolicy_train_batched` with H2D/D2H
boundary plumbing; deferred until a consumer needs it.

Trait surface
  - `OffPolicyAgent` — minimal: select_action_batched[N_ENVS],
    record, train_step, episode tracker accessors, batched
    CPU record + add_complete_return.
  - `OffPolicyAgentGpu(OffPolicyAgent)` — adds
    record_batch_gpu / record_batch_gpu_nstep for the gpu-env path.

Storage: all driver-owned buffers live in `DriverScratch[NAME, N, DIM]`
which unifies host `List` and device `DeviceBuffer` backing behind one
type. Env adapters (`BatchedCpuEnv`, `BatchedGpuEnv` in `batched_env.mojo`)
own their obs/action/reward/done buffers and expose pointer accessors
through the `BatchedEnv` trait.
"""

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from ..data.n_step_replay import GPUNStepBuffer
from mojo_rl.core.env_traits import BoxContinuousActionEnv
from .batched_env import BatchedEnv
from .driver_scratch import DriverScratch


# ──────────────────────────────────────────────────────────────────────
# OffPolicyAgent — trait for the off-policy drivers.
# ──────────────────────────────────────────────────────────────────────


trait OffPolicyAgent(Movable, ImplicitlyDestructible):
    """Single-trait surface for the off-policy drivers.
    Exposes `AGENT_TRAIN_TARGET` (so the driver can comptime-gate
    H2D/D2H around the env step) and routes all action selection
    through one `select_action_batched[N_ENVS]` entry instead of the
    historic three `select_action[/_gpu/_gpu_batched]` variants.

    `record` keeps a host-`List` signature for single-env use (env step
    returns Lists). Batched record paths live on the `OffPolicyAgentGpu`
    sub-trait (or, for the CPU env batched path, in `record_batch_cpu`
    here). SAC / MBPO / DDPG / TD3 all conform."""

    # Trait-visible alias of the trainer's struct-comptime `train_target`.
    # SACTrainer exposes this via `AGENT_TRAIN_TARGET = Self.train_target`.
    # Conceptually distinct from the env's `ENV_TARGET` — see the
    # module docstring for the dual-target model.
    comptime AGENT_TRAIN_TARGET: StaticString
    # `AGENT_` prefix avoids clashing with the struct's own
    # OBS_DIM/ACT_DIM comptime params on conforming trainers.
    comptime AGENT_OBS_DIM: Int
    comptime AGENT_ACT_DIM: Int

    def select_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ao_scratch_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alp_scratch_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        ...

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Deterministic, exploration-free action selection for eval.
        Host-list signature; trainers dispatch internally on
        `AGENT_TRAIN_TARGET` (CPU trainers run native, GPU trainers
        H2D the obs and D2H the action under the hood). Used by
        `run_offpolicy_eval`."""
        ...

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        ...

    def end_episode(mut self):
        ...

    def train_step(mut self, step_idx: Int) raises -> Bool:
        ...

    def mean_return(self) -> Scalar[DT]:
        ...

    def ep_count(self) -> Int:
        ...

    # ─── Tier-2 additions — batched CPU env support ──────────────────
    #
    # Drivers that step many envs per iteration push complete-episode
    # returns through `add_complete_return` (instead of the single-env
    # `end_episode`/`record` flow which auto-updates the tracker's
    # `current_return`). `record_batch_cpu` pushes N transitions into
    # the trainer's replay without touching the tracker so the driver
    # can manage per-env return accumulators on the host.
    #
    # `add_complete_return` was previously declared on
    # `OffPolicyAgentGpu`; lifted to the parent so both the
    # GPU-env driver (Phase 3.5) and the new CPU-env batched driver
    # (Tier-2) share one source of truth — and so the Gpu sub-trait
    # doesn't re-declare it (which would create diamond ambiguity).

    def add_complete_return(mut self, ret: Scalar[DT]):
        ...

    def record_batch_cpu[
        N_ENVS: Int
    ](
        mut self,
        prev_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Push N transitions from host pointer slabs into the
        trainer's replay buffer. Does NOT update the trainer's
        episode tracker — caller manages per-env returns via
        `add_complete_return`."""
        ...


# ──────────────────────────────────────────────────────────────────────
# OffPolicyAgentGpu — adds GPU-batched methods on top.
# ──────────────────────────────────────────────────────────────────────


trait OffPolicyAgentGpu(OffPolicyAgent):
    """Extends `OffPolicyAgent` with the GPU-batched record
    surfaces needed by the GPU-env driver. `add_complete_return`
    is inherited from the parent — single source of truth.

    CPU-only trainers (e.g. MBPOTrainer) conform with raising stubs for
    `record_batch_gpu` / `record_batch_gpu_nstep`; the Tier-3 driver
    comptime-elides the GPU branch when env_target == "cpu" so the
    stubs are never invoked."""

    def record_batch_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        ...

    def record_batch_gpu_nstep[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS, Self.AGENT_OBS_DIM, Self.AGENT_ACT_DIM, N_ENVS,
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        ...


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_train — single-env, env_target="cpu".
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_train[
    A: OffPolicyAgent,
    E: BoxContinuousActionEnv,
](
    mut trainer: A,
    mut env: E,
    total_timesteps: Int,
    *,
    ctx: Optional[DeviceContext] = None,
    print_every: Int = 1_000,
    verbose: Bool = True,
) raises -> List[Scalar[DT]]:
    """Single-env off-policy training driver bound on the CPU env trait
    (`BoxContinuousActionEnv`). Covers (env_target=cpu, train_target=cpu)
    and (env_target=cpu, train_target=gpu).

    `ctx` is required for `train_target=gpu` (used for H2D obs / D2H
    action staging) and ignored for `train_target=cpu`. Must be the
    SAME `DeviceContext` the trainer was built with — Apple Metal's
    queue pool exhausts if a new context is constructed per call.

    Loop semantics: one env step + one `train_step` per
    iteration. `select_action_batched` consumes RNG in the same order
    on CPU as the legacy single-env CPU path. The GPU branch differs
    in warmup RNG (Philox kernel vs host `random_float64`).
    """
    # ENV_TARGET is implicit "cpu" since E is bound on BoxContinuousActionEnv.
    # Made explicit here so the dual-axis model is visible at the dispatch site.
    comptime env_target: StaticString = "cpu"
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM
    comptime ACT = A.AGENT_ACT_DIM

    comptime assert (
        train_target == "cpu" or train_target == "gpu"
    ), "run_offpolicy_train: train_target must be 'cpu' or 'gpu'"
    comptime if train_target == "gpu":
        if not ctx:
            raise Error(
                "run_offpolicy_train[train_target='gpu']:"
                " ctx required for env→trainer H2D/D2H staging"
            )

    # Driver-owned scratches. Allocated on train_target so the trainer's
    # select_action_batched consumes them natively. When env_target !=
    # train_target (cpu env + gpu trainer), obs + action also need host
    # mirrors for the per-step H2D/D2H around the env step.
    comptime needs_boundary_copy: Bool = env_target != train_target
    var obs_scratch = DriverScratch["obs", 1, OBS].make[train_target](
        ctx=ctx, with_host_mirror=needs_boundary_copy,
    )
    var action_scratch = DriverScratch["action", 1, ACT].make[train_target](
        ctx=ctx, with_host_mirror=needs_boundary_copy,
    )
    var ao = DriverScratch["ao", 1, 2 * ACT].make[train_target](ctx=ctx)
    var alp = DriverScratch["alp", 1, ACT + 1].make[train_target](ctx=ctx)

    # Host-side Lists for `trainer.record` + env stepping. (`record`'s
    # signature is the legacy host-List form — see trait docstring.)
    var obs_list = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var next_obs_list = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action_list = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))

    var env_obs = env.reset_obs_list()
    var env_action = List[Scalar[E.dtype]](capacity=ACT)
    for _ in range(ACT):
        env_action.append(Scalar[E.dtype](0.0))

    var ep_returns = List[Scalar[DT]]()
    var current_ep_count = trainer.ep_count()

    var t_start = perf_counter_ns()
    var step: Int = 0
    while step < total_timesteps:
        # Copy env obs (E.dtype) into obs_list (DT) for record + into
        # the driver scratch (DT) for the select_action_batched call.
        # When train_target=="cpu", obs_scratch.host_ptr() IS the
        # scratch's only storage; when "gpu" it's the host mirror that
        # will be H2D'd below.
        var obs_scratch_h = obs_scratch.host_ptr()
        for d in range(OBS):
            var v = Scalar[DT](env_obs[d])
            obs_list[d] = v
            obs_scratch_h[d] = v

        # Boundary copy: env_target != train_target requires H2D obs.
        # Elided when env_target == train_target.
        comptime if needs_boundary_copy:
            var c = ctx.value()
            c.enqueue_copy(obs_scratch.dev.value(), obs_scratch_h)

        trainer.select_action_batched[1](
            obs_scratch.target_ptr[train_target](),
            action_scratch.target_ptr[train_target](),
            ao.target_ptr[train_target](),
            alp.target_ptr[train_target](),
            step,
        )

        # Boundary copy: env_target != train_target requires D2H action.
        comptime if needs_boundary_copy:
            var c = ctx.value()
            c.enqueue_copy(
                action_scratch.host_ptr(), action_scratch.dev.value()
            )
            c.synchronize()

        var action_h = action_scratch.host_ptr()
        for j in range(ACT):
            var a = action_h[j]
            action_list[j] = a
            env_action[j] = Scalar[E.dtype](a)

        # Env step (CPU env).
        var step_res = env.step_continuous_vec[E.dtype](env_action)
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(OBS):
            next_obs_list[d] = Scalar[DT](nxt[d])

        trainer.record(
            obs_list,
            action_list,
            Scalar[DT](reward),
            next_obs_list,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )

        if done:
            trainer.end_episode()
            env_obs = env.reset_obs_list()
            var new_ep_count = trainer.ep_count()
            if new_ep_count > current_ep_count:
                ep_returns.append(trainer.mean_return())
                current_ep_count = new_ep_count
        else:
            env_obs = nxt^

        step += 1
        _ = trainer.train_step(step)

        if verbose and print_every > 0 and step % print_every == 0:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ",
                step,
                "] mean_ret(10)=",
                trainer.mean_return(),
                " ep=",
                trainer.ep_count(),
                " elapsed=",
                elapsed,
                "s",
            )

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_train_batched — Tier-3: ONE driver for all
#   (env_target, train_target, N_ENVS) combos via BatchedEnv.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_train_batched[
    A: OffPolicyAgentGpu,
    E: BatchedEnv,
    N_ENVS: Int = 1,
    NS: Int = 1,
](
    ctx: Optional[DeviceContext],
    mut trainer: A,
    mut env: E,
    total_env_steps: Int,
    *,
    rng_seed: UInt64 = UInt64(42),
    updates_per_step: Int = 1,
    print_every: Int = 5_000,
    verbose: Bool = True,
    nstep_gamma: Scalar[DT] = Scalar[DT](0.99),
) raises -> List[Scalar[DT]]:
    """Tier-3 off-policy driver covering same-target (env_target ==
    train_target) combinations through the `BatchedEnv` trait:

      env_target | train_target | N_ENVS | covered
      -----------|--------------|--------|--------
      cpu        | cpu          | >=1    | yes  (via BatchedCpuEnv)
      gpu        | gpu          | >=1    | yes  (via BatchedGpuEnv)

    Cross-target combinations are NOT covered here:
      - (cpu env, gpu train) reachable via `run_offpolicy_train`
        (Tier-1 Phase 3) at N_ENVS=1. Batched cross-target requires
        H2D-ing prev_obs/action/reward/obs/done before record_batch_gpu;
        the boundary plumbing is straightforward but the use case is
        rare (people with a GPU usually also have a GPU env), so it's
        deferred until a consumer needs it.
      - (gpu env, cpu train) rejected as degenerate (D2H every obs).

    Bounded on `OffPolicyAgentGpu` because the gpu-env branch
    needs `record_batch_gpu`; the cpu-env branch uses `record_batch_cpu`
    inherited from the parent. The driver comptime-branches on
    `(env_target, N_ENVS)` so each combination compiles only the
    kernels it actually needs.

    `ctx` is required for `env_target == "gpu"`; pass `None` for the
    pure CPU case.

    Loop:
      1. snapshot env.obs_ptr()           → prev_obs (driver-owned)
      2. trainer.select_action_batched[N_ENVS] → env.action_ptr() directly
      3. env.step_batch[N_ENVS]           → env.obs / .reward / .done
      4. trainer.record_batch_cpu OR record_batch_gpu (env-side ptrs)
      5. per-env return accumulation + add_complete_return on done
      6. env.selective_reset_batch[N_ENVS]
      7. updates_per_step × trainer.train_step
    """
    comptime env_target: StaticString = E.ENV_TARGET
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM
    comptime ACT = A.AGENT_ACT_DIM

    comptime assert (
        env_target == "cpu" or env_target == "gpu"
    ), "env_target must be 'cpu' or 'gpu'"
    comptime assert (
        train_target == "cpu" or train_target == "gpu"
    ), "train_target must be 'cpu' or 'gpu'"
    comptime assert env_target == train_target, (
        "run_offpolicy_train_batched: env_target must equal train_target."
        " Cross-target combinations: (cpu env, gpu train) → use"
        " run_offpolicy_train (Tier-1, single-env); (gpu env,"
        " cpu train) → rejected as degenerate."
    )
    comptime assert N_ENVS > 0, "N_ENVS must be > 0"
    comptime assert NS > 0, "NS must be > 0"
    comptime assert (
        E.OBS_DIM == OBS and E.ACT_DIM == ACT
    ), "BatchedEnv dimensions must match trainer dimensions"
    comptime if NS > 1:
        comptime assert env_target == "gpu", (
            "run_offpolicy_train_batched[NS>1]: n-step only supported"
            " on GPU env path (GPUNStepBuffer is GPU-only)"
        )
    comptime if env_target == "gpu":
        if not ctx:
            raise Error(
                "run_offpolicy_train_batched: ctx required when"
                " env_target is 'gpu'"
            )

    # n-step buffer (NS > 1 only). Declared as Optional at function
    # level because Mojo nightly's `comptime if` does not bleed
    # bindings to sibling blocks — Optional+`if ctx:` is the bridge.
    # For CPU env (ctx=None), stays None and is never touched
    # (`comptime assert NS == 1` for the CPU path is enforced above).
    var nstep_buf: Optional[
        GPUNStepBuffer[NS, A.AGENT_OBS_DIM, A.AGENT_ACT_DIM, N_ENVS]
    ] = None
    if ctx:
        nstep_buf = Optional(
            GPUNStepBuffer[
                NS, A.AGENT_OBS_DIM, A.AGENT_ACT_DIM, N_ENVS,
            ].new(ctx.value(), gamma=nstep_gamma)
        )

    # All scratches on the single target (env_target == train_target).
    var ao = DriverScratch["ao", N_ENVS, 2 * ACT].make[train_target](ctx=ctx)
    var alp = DriverScratch["alp", N_ENVS, ACT + 1].make[train_target](
        ctx=ctx
    )
    var prev_obs = DriverScratch["prev_obs", N_ENVS, OBS].make[
        env_target
    ](ctx=ctx)

    var per_env_returns = List[Scalar[DT]](
        length=N_ENVS, fill=Scalar[DT](0.0),
    )

    env.reset_batch[N_ENVS](ctx=ctx, rng_seed=rng_seed)

    var ep_returns = List[Scalar[DT]]()
    var t_start = perf_counter_ns()
    var step_idx: Int = 0
    var iter_idx: Int = 0
    var next_print: Int = print_every

    while step_idx < total_env_steps:
        # ── 1. Snapshot prev_obs from env.obs_ptr().
        comptime if env_target == "cpu":
            var po_p = prev_obs.host_ptr()
            var ob_p = env.obs_ptr()
            for k in range(N_ENVS * OBS):
                po_p[k] = ob_p[k]
        else:
            # GPU env: D→D enqueue_copy. Reconstruct DeviceBuffer view
            # over env.obs_ptr() (owning=False — env still owns memory).
            var c = ctx.value()
            var env_obs_view = DeviceBuffer[DT](
                c, env.obs_ptr(), N_ENVS * OBS, owning=False,
            )
            c.enqueue_copy(prev_obs.dev.value(), env_obs_view)

        # ── 2. Trainer writes action directly into env.action_ptr().
        # env_target == train_target so the pointer is on the right side.
        trainer.select_action_batched[N_ENVS](
            env.obs_ptr(),
            env.action_ptr(),
            ao.target_ptr[train_target](),
            alp.target_ptr[train_target](),
            step_idx,
        )

        # ── 3. Env step (writes env-internal obs/reward/done).
        env.step_batch[N_ENVS](
            ctx=ctx,
            rng_seed=rng_seed + UInt64(iter_idx + 1),
        )

        # ── 4. Replay push (env-target-specific).
        comptime if env_target == "cpu":
            trainer.record_batch_cpu[N_ENVS](
                prev_obs.host_ptr(),
                env.action_ptr(),
                env.reward_ptr(),
                env.obs_ptr(),
                env.done_ptr(),
            )
        else:
            # GPU env. Reconstruct non-owning DeviceBuffer views over
            # the env's pointers to pass to record_batch_gpu /
            # record_batch_gpu_nstep.
            var c = ctx.value()
            var action_buf = DeviceBuffer[DT](
                c, env.action_ptr(), N_ENVS * ACT, owning=False,
            )
            var reward_buf = DeviceBuffer[DT](
                c, env.reward_ptr(), N_ENVS, owning=False,
            )
            var obs_buf = DeviceBuffer[DT](
                c, env.obs_ptr(), N_ENVS * OBS, owning=False,
            )
            var done_buf = DeviceBuffer[DT](
                c, env.done_ptr(), N_ENVS, owning=False,
            )
            comptime if NS > 1:
                trainer.record_batch_gpu_nstep[N_ENVS, NS](
                    c,
                    nstep_buf.value(),
                    prev_obs.dev.value(),
                    action_buf,
                    reward_buf,
                    obs_buf,
                    done_buf,
                )
            else:
                trainer.record_batch_gpu[N_ENVS](
                    c,
                    prev_obs.dev.value(),
                    action_buf,
                    reward_buf,
                    obs_buf,
                    done_buf,
                )

        # ── 5. Per-env episode tracking. Needs host-side reward+done.
        comptime if env_target == "cpu":
            var rewards_h = env.reward_ptr()
            var dones_h = env.done_ptr()
            for e in range(N_ENVS):
                per_env_returns[e] = per_env_returns[e] + rewards_h[e]
                if dones_h[e] > Scalar[DT](0.5):
                    trainer.add_complete_return(per_env_returns[e])
                    per_env_returns[e] = Scalar[DT](0.0)
                    ep_returns.append(trainer.mean_return())
        else:
            # GPU env: D2H of reward + done (small, N_ENVS*2 scalars).
            var c = ctx.value()
            var host_rewards = c.enqueue_create_host_buffer[DT](N_ENVS)
            var host_dones = c.enqueue_create_host_buffer[DT](N_ENVS)
            var reward_view = DeviceBuffer[DT](
                c, env.reward_ptr(), N_ENVS, owning=False,
            )
            var done_view = DeviceBuffer[DT](
                c, env.done_ptr(), N_ENVS, owning=False,
            )
            c.enqueue_copy(host_rewards, reward_view)
            c.enqueue_copy(host_dones, done_view)
            c.synchronize()
            var rewards_h = host_rewards.unsafe_ptr()
            var dones_h = host_dones.unsafe_ptr()
            for e in range(N_ENVS):
                per_env_returns[e] = per_env_returns[e] + rewards_h[e]
                if dones_h[e] > Scalar[DT](0.5):
                    trainer.add_complete_return(per_env_returns[e])
                    per_env_returns[e] = Scalar[DT](0.0)
                    ep_returns.append(trainer.mean_return())

        # ── 8. Selective env reset.
        env.selective_reset_batch[N_ENVS](
            ctx=ctx,
            rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7),
        )

        step_idx += N_ENVS
        iter_idx += 1

        # ── 9. Trainer updates.
        for _ in range(updates_per_step):
            _ = trainer.train_step(step_idx)

        if verbose and print_every > 0 and step_idx >= next_print:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ",
                step_idx,
                "] mean_ret(10)=",
                trainer.mean_return(),
                " ep=",
                trainer.ep_count(),
                " elapsed=",
                elapsed,
                "s",
            )
            next_print += print_every

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_eval — single-env greedy eval, target-agnostic.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_eval[
    A: OffPolicyAgent,
    E: BoxContinuousActionEnv,
](
    mut trainer: A,
    mut env: E,
    num_episodes: Int,
    *,
    max_steps_per_episode: Int = 1_000,
    verbose: Bool = False,
) raises -> Scalar[DT]:
    """Non-mutating greedy eval driver — replaces both
    `run_offpolicy_eval_cpu` and `run_offpolicy_eval_gpu`.

    Trainer contract: `OffPolicyAgent.select_greedy_action`
    handles target dispatch internally (CPU trainers run native; GPU
    trainers H2D the obs and D2H the action under the hood). Only that
    method is invoked here — `record` / `train_step` /
    `end_episode` / `add_complete_return` are intentionally skipped so
    eval doesn't touch the trainer's replay buffer, optimizers, or
    episode tracker. `obs_dim` / `act_dim` are read from
    `A.AGENT_OBS_DIM` / `A.AGENT_ACT_DIM`.
    """
    comptime OBS = A.AGENT_OBS_DIM
    comptime ACT = A.AGENT_ACT_DIM

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))

    var action_list = List[Scalar[E.dtype]](capacity=ACT)
    for _ in range(ACT):
        action_list.append(Scalar[E.dtype](0.0))

    var total_return = Scalar[DT](0.0)
    var t_start = perf_counter_ns()

    for ep in range(num_episodes):
        var obs_list = env.reset_obs_list()
        var ep_return = Scalar[DT](0.0)
        var ep_steps: Int = 0

        for _ in range(max_steps_per_episode):
            for d in range(OBS):
                obs[d] = Scalar[DT](obs_list[d])
            trainer.select_greedy_action(obs, action)
            for j in range(ACT):
                action_list[j] = Scalar[E.dtype](action[j])
            var step_res = env.step_continuous_vec[E.dtype](action_list)
            var nxt = step_res[0].copy()
            var reward = step_res[1]
            var done = step_res[2]
            ep_return += Scalar[DT](reward)
            ep_steps += 1
            if done:
                break
            obs_list = nxt^

        total_return += ep_return
        if verbose:
            print(
                "  [eval ep ",
                ep + 1,
                "/",
                num_episodes,
                "] return=",
                ep_return,
                " steps=",
                ep_steps,
            )

    var mean = total_return / Scalar[DT](num_episodes)
    if verbose:
        var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
        print(
            "eval: mean_return=",
            mean,
            " (",
            num_episodes,
            " episodes, ",
            elapsed,
            " s)",
        )
    return mean
