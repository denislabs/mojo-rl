"""Off-policy training + eval drivers — Tier-1 + Tier-3.

Two training driver functions covering all useful (env_target,
train_target, N_ENVS) combinations:

  env_target | train_target | N_ENVS | driver
  -----------|--------------|--------|----------------------------------
  cpu        | cpu          | >=1    | run_offpolicy_train_batched
  gpu        | gpu          | >=1    | run_offpolicy_train_batched
  cpu        | gpu          | 1      | run_offpolicy_train
  cpu        | gpu          | >=1    | run_offpolicy_train_cpu_env_gpu_agent

Plus one eval driver `run_offpolicy_eval` that replaces the
legacy `run_offpolicy_eval_cpu` / `run_offpolicy_eval_gpu` split — the
trainer dispatches CPU vs GPU internally inside
`select_greedy_action`.

The (env=gpu, train=cpu) combination is omitted as degenerate
(D2H every obs back to CPU for training — never useful in practice).
Batched cross-target (cpu env, gpu train, N>=1) — the GPU-agent /
CPU-env hybrid, ported from legacy `gpu_agent_cpu_env_train.mojo`
(Phase 6.1) — is covered by `run_offpolicy_train_cpu_env_gpu_agent`,
which wraps the same `BatchedCpuEnv` adapter and inserts per-step H2D
obs / D2H action / H2D transition-slab boundary copies around the GPU
trainer's `select_action_batched` + `record_batch_gpu`. Lets a GPU
SAC/TD3/DDPG agent train against any CPU-stepped env (e.g. a Python
Gymnasium MuJoCo env) so training failures can be attributed to the
env vs the algorithm.

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
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.cuda import CUDAGraph, maybe_capture_replay
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

    def total_train_steps(self) -> Int:
        """Cumulative gradient-update count (UTD-aware), monotonic, not
        reset by flushes. The single-env driver keys the `diag_every`
        cadence on THIS rather than env-steps, so one diag point spans
        `diag_every` updates regardless of UTD. (At UTD=40 an env-step
        cadence makes each point span 40×diag_every updates — the
        Q/critic curves collapse to a handful of points.) Default 0 means
        a conformer doesn't track it; every real trainer overrides."""
        return 0

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

    # ─── Optional cadence hooks (default no-op) ──────────────────────
    #
    # These let the driver call into the trainer at the user's
    # `diag_every` / `checkpoint_every` cadences without splitting the
    # env-loop into chunks. Each has a `pass` default so existing
    # trainers (DDPG/TD3/PPO/MBPO/DQN) don't need to do anything until
    # their own metric/checkpoint paths are wired up. SACTrainer
    # overrides both with real bodies that call its `flush_metrics` and
    # a one-file v2 `save_state` writer.
    #
    # `flush_metrics_through_logger[L]` discards the typed bundle so
    # the trait can be uniform across algorithms whose `flush_metrics`
    # return different `*Metrics` structs.

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        pass

    def save_state(mut self, path: String) raises:
        pass


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

    # ─── CUDA-graph capture surface (Slice 7) ─────────────────────────
    #
    # Default bodies so agents that don't support train-step capture still
    # conform. Only invoked on the `USE_TRAIN_CUDA_GRAPH=True` driver path
    # (SAC GPU today), so the defaults are never reached for other agents.

    def train_device_kernels(mut self) raises:
        """Pure device-kernel train step (no host work) — the body captured
        into the CUDA graph. Default raises: capture unsupported."""
        raise Error(
            "train_device_kernels: CUDA-graph capture not supported by"
            " this agent (USE_TRAIN_CUDA_GRAPH must stay False)"
        )

    def note_train_update(mut self):
        """Advance one logical update's host bookkeeping (counters/metric
        accumulators). Default no-op."""
        pass

    def learning_starts_count(self) -> Int:
        """Cumulative env-step threshold after which training begins — the
        driver gates the capture path on this. Default 0."""
        return 0


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_train — single-env, env_target="cpu".
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_train[
    A: OffPolicyAgent,
    E: BoxContinuousActionEnv,
    L: Logger = NoOpLogger,
](
    mut trainer: A,
    mut env: E,
    total_timesteps: Int,
    *,
    ctx: Optional[DeviceContext] = None,
    print_every: Int = 1_000,
    verbose: Bool = True,
    logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
    diag_every: Int = 0,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    base_step: Int = 0,
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
    # Diag cadence is keyed on TRAIN steps (gradient updates) via
    # `trainer.total_train_steps()`, not env-steps — see the trait method.
    # `last_diag_bucket` tracks the last emitted `train_steps // diag_every`.
    var last_diag_bucket: Int = 0
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

        # `base_step + step` is the cumulative env-step counter — required
        # so the trainer's `learning_starts` warmup gate stays correct
        # across chunked calls (see `SACAgent.train_single` which loops
        # this driver in cadence-sized blocks). Equivalent to `step` when
        # `base_step=0` (single-call usage).
        trainer.select_action_batched[1](
            obs_scratch.target_ptr[train_target](),
            action_scratch.target_ptr[train_target](),
            ao.target_ptr[train_target](),
            alp.target_ptr[train_target](),
            base_step + step,
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
        # `done` (terminated OR truncated) drives reset/episode tracking; the
        # replay buffer stores `terminated` ONLY (natural termination) so the
        # SAC TD bootstrap is kept on truncation but dropped on termination.
        var terminated = env.was_terminated()
        for d in range(OBS):
            next_obs_list[d] = Scalar[DT](nxt[d])

        trainer.record(
            obs_list,
            action_list,
            Scalar[DT](reward),
            next_obs_list,
            Scalar[DT](1.0) if terminated else Scalar[DT](0.0),
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
        _ = trainer.train_step(base_step + step)

        var abs_step = base_step + step

        if verbose and print_every > 0 and abs_step % print_every == 0:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ",
                abs_step,
                "] mean_ret(10)=",
                trainer.mean_return(),
                " ep=",
                trainer.ep_count(),
                " elapsed=",
                elapsed,
                "s",
            )

        # Logger emit at the print cadence. Comptime-elided when
        # L=NoOpLogger (default).
        #
        # NOTE: no forced `.flush()` here — `log_scalar` auto-flushes
        # when the logger's buffer fills (CsvLogger / RemoteLogger
        # check `len(entries) >= buffer_size`). The user controls flush
        # cadence via `buffer_size`. Final residual entries are sent
        # by `logger.close()` at end of training. Synchronous flushes
        # are expensive (~50-200 ms per HTTP POST on a remote
        # endpoint); forcing one every print_every was the dominant
        # logger overhead in profiling.
        comptime if L.ENABLED:
            if (
                print_every > 0
                and abs_step % print_every == 0
                and Bool(logger)
            ):
                logger.value()[].log_scalar(
                    "avg_reward",
                    Float64(trainer.mean_return()),
                    abs_step,
                )
                logger.value()[].log_scalar(
                    "episodes",
                    Float64(trainer.ep_count()),
                    abs_step,
                )

        # `diag_every` — drain the trainer's metric bundle through the
        # logger every `diag_every` TRAIN steps (gradient updates), not
        # env-steps. At UTD>1 the env-step cadence makes each diag point
        # span UTD×diag_every updates (e.g. 200k) → the Q/critic/dyn curves
        # collapse to a few points. Keying on `total_train_steps()` keeps
        # one point per `diag_every` updates at any UTD. Default trait impl
        # of `flush_metrics_through_logger` is a no-op for trainers that
        # haven't wired metrics up yet.
        var diag_due = False
        if diag_every > 0:
            var bucket = trainer.total_train_steps() // diag_every
            if bucket > last_diag_bucket:
                last_diag_bucket = bucket
                diag_due = True

        comptime if L.ENABLED:
            if diag_due and Bool(logger):
                trainer.flush_metrics_through_logger[L](logger, abs_step)

        # Live flush: push whatever was just buffered to the monitoring
        # server at the print/diag cadence so the dashboard updates DURING
        # training (otherwise points only auto-flush once the buffer hits
        # `buffer_size`, leaving long runs blind until `logger.close()`).
        # `flush()` early-returns when the buffer is empty, so this is a
        # no-op on the vast majority of steps.
        comptime if L.ENABLED:
            if (
                Bool(logger)
                and (
                    (print_every > 0 and abs_step % print_every == 0)
                    or diag_due
                )
            ):
                logger.value()[].flush()

        # `checkpoint_every` — overwrite `checkpoint_path` with the
        # trainer's one-file v2 envelope. Default trait impl is no-op.
        if (
            checkpoint_every > 0
            and abs_step % checkpoint_every == 0
            and checkpoint_path.byte_length() > 0
        ):
            trainer.save_state(checkpoint_path)

    # Always overwrite the final checkpoint at end so resume gets the
    # freshest weights regardless of cadence alignment.
    if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
        trainer.save_state(checkpoint_path)

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
    L: Logger = NoOpLogger,
    USE_TRAIN_CUDA_GRAPH: Bool = False,
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
    logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
    base_step: Int = 0,
    diag_every: Int = 0,
    episode_sync_every: Int = 1,
) raises -> List[Scalar[DT]]:
    """Tier-3 off-policy driver covering same-target combinations.

    `episode_sync_every` (GPU-env path only): batch the per-iteration
    reward/done D2H readback used for episode-return bookkeeping over this many
    iterations, draining (one `synchronize`) only when the ring fills OR a
    print/diag boundary is reached OR the loop ends. The default `1` reproduces
    the original per-iteration sync exactly; higher values trade logging
    granularity for far fewer host↔device stalls — important once the train
    step is a captured CUDA graph (the sync would otherwise serialize every
    iteration and negate the capture win). Returns are still drained in order,
    so `mean_return` / `ep_count` are exact at every emit boundary.

    Same-target means `env_target == train_target`. Routed through the
    `BatchedEnv` trait:

      env_target | train_target | N_ENVS | covered
      -----------|--------------|--------|--------
      cpu        | cpu          | >=1    | yes  (via BatchedCpuEnv)
      gpu        | gpu          | >=1    | yes  (via BatchedGpuEnv)

    Cross-target combinations are NOT covered here:
      - (cpu env, gpu train) reachable via `run_offpolicy_train`
        (Tier-1 Phase 3) at N_ENVS=1, or — for N_ENVS>=1 batched —
        via `run_offpolicy_train_cpu_env_gpu_agent` (Phase 6.1), which
        adds the H2D prev_obs/action/reward/obs/terminated boundary
        plumbing before `record_batch_gpu`.
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
    comptime if USE_TRAIN_CUDA_GRAPH:
        comptime assert train_target == "gpu", (
            "USE_TRAIN_CUDA_GRAPH requires train_target == 'gpu' (CUDA-graph"
            " capture is a GPU-only path; no-op on non-NVIDIA)"
        )
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

    # Slice 7 — lazily-captured train-step graph (None until first capture
    # past warmup). Declared at function level (Mojo `comptime if` bindings
    # don't bleed to sibling blocks; the slot is read/written inside the
    # loop's capture branch). Unused when USE_TRAIN_CUDA_GRAPH is False, and
    # a no-op on non-NVIDIA where `CUDAGraph` itself is a no-op.
    var train_graph: Optional[CUDAGraph] = None

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

    # Deferred episode-tracking readback (GPU env only). A ring of pinned host
    # buffers — one (reward, done) pair per slot — lets us enqueue the small
    # per-iteration D2H copies WITHOUT synchronizing, then drain `pending_eps`
    # buffered iterations with a single `synchronize` when the ring fills or an
    # emit boundary arrives. Declared at function level because Mojo nightly's
    # `comptime if` bindings don't bleed to sibling blocks; populated only for
    # the GPU-env branch. `sync_every == 1` reproduces the original
    # per-iteration sync exactly.
    var sync_every: Int = episode_sync_every if episode_sync_every >= 1 else 1
    var ring_reward = List[HostBuffer[DT]]()
    var ring_done = List[HostBuffer[DT]]()
    var pending_eps: Int = 0
    comptime if env_target == "gpu":
        if ctx:
            var c0 = ctx.value()
            for _ in range(sync_every):
                ring_reward.append(c0.enqueue_create_host_buffer[DT](N_ENVS))
                ring_done.append(c0.enqueue_create_host_buffer[DT](N_ENVS))

    env.reset_batch[N_ENVS](ctx=ctx, rng_seed=rng_seed)

    var ep_returns = List[Scalar[DT]]()
    var t_start = perf_counter_ns()
    var step_idx: Int = 0
    var iter_idx: Int = 0
    var next_print: Int = print_every
    # Independent counter for logger emit; only read inside the
    # `comptime if L.ENABLED` block so the default (NoOpLogger) path
    # never reads or writes it after this initialization.
    var next_log: Int = print_every
    # Independent counter for the diag-bundle flush cadence (mean_q /
    # critic_loss / alpha / train_steps / …). Disabled when diag_every == 0.
    var next_diag: Int = diag_every if diag_every > 0 else total_env_steps + 1

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
        # `base_step + step_idx` is the cumulative env-step counter (see
        # the `base_step` note in `run_offpolicy_train`).
        trainer.select_action_batched[N_ENVS](
            env.obs_ptr(),
            env.action_ptr(),
            ao.target_ptr[train_target](),
            alp.target_ptr[train_target](),
            base_step + step_idx,
        )

        # ── 3. Env step (writes env-internal obs/reward/done).
        env.step_batch[N_ENVS](
            ctx=ctx,
            rng_seed=rng_seed + UInt64(iter_idx + 1),
        )

        # ── 4. Replay push (env-target-specific).
        comptime if env_target == "cpu":
            # Replay stores `terminated` (natural termination only) so the TD
            # bootstrap is kept on truncation, dropped on termination. `done`
            # (terminated OR truncated) is still used below for episode
            # tracking and selective reset.
            trainer.record_batch_cpu[N_ENVS](
                prev_obs.host_ptr(),
                env.action_ptr(),
                env.reward_ptr(),
                env.obs_ptr(),
                env.terminated_ptr(),
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
            # Replay stores `terminated` (natural termination only), NOT the
            # combined `done`, so the TD bootstrap is kept on time-limit
            # truncation and dropped on real termination. Episode tracking and
            # selective reset below still use `done_ptr()`.
            var term_buf = DeviceBuffer[DT](
                c, env.terminated_ptr(), N_ENVS, owning=False,
            )
            comptime if NS > 1:
                trainer.record_batch_gpu_nstep[N_ENVS, NS](
                    c,
                    nstep_buf.value(),
                    prev_obs.dev.value(),
                    action_buf,
                    reward_buf,
                    obs_buf,
                    term_buf,
                )
            else:
                trainer.record_batch_gpu[N_ENVS](
                    c,
                    prev_obs.dev.value(),
                    action_buf,
                    reward_buf,
                    obs_buf,
                    term_buf,
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
            # GPU env: enqueue the small reward+done D2H into the next ring
            # slot WITHOUT synchronizing; drain (one sync) only when the ring
            # fills or an emit boundary is imminent. `step_idx` is still
            # pre-increment here, so the upcoming post-increment value is
            # `step_idx + N_ENVS` — the same value the print/diag blocks test.
            var c = ctx.value()
            var reward_view = DeviceBuffer[DT](
                c, env.reward_ptr(), N_ENVS, owning=False,
            )
            var done_view = DeviceBuffer[DT](
                c, env.done_ptr(), N_ENVS, owning=False,
            )
            c.enqueue_copy(ring_reward[pending_eps], reward_view)
            c.enqueue_copy(ring_done[pending_eps], done_view)
            pending_eps += 1

            var post_step = step_idx + N_ENVS
            var emit_now = print_every > 0 and post_step >= next_print
            if diag_every > 0 and post_step >= next_diag:
                emit_now = True
            if post_step >= total_env_steps:
                emit_now = True
            if pending_eps >= sync_every or emit_now:
                c.synchronize()
                for s in range(pending_eps):
                    var rewards_h = ring_reward[s].unsafe_ptr()
                    var dones_h = ring_done[s].unsafe_ptr()
                    for e in range(N_ENVS):
                        per_env_returns[e] = per_env_returns[e] + rewards_h[e]
                        if dones_h[e] > Scalar[DT](0.5):
                            trainer.add_complete_return(per_env_returns[e])
                            per_env_returns[e] = Scalar[DT](0.0)
                            ep_returns.append(trainer.mean_return())
                pending_eps = 0

        # ── 8. Selective env reset.
        env.selective_reset_batch[N_ENVS](
            ctx=ctx,
            rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7),
        )

        step_idx += N_ENVS
        iter_idx += 1

        # ── 9. Trainer updates.
        comptime if USE_TRAIN_CUDA_GRAPH:
            # Capture path: once the buffer is warm, the per-update device
            # kernel sequence (`train_device_kernels`) is captured into
            # `train_graph` on first call and replayed thereafter — host
            # bookkeeping advances via `note_train_update` so counters stay
            # correct across replays. During warmup we skip training, which
            # matches `train_step` gating to False. On non-NVIDIA the closure
            # simply runs each call (no-op capture) → identical to the
            # non-captured path. Capture requires uniform replay (the
            # caller must not enable PER with this flag).
            if base_step + step_idx >= trainer.learning_starts_count():
                var c = ctx.value()

                def _captured_step() capturing raises -> None:
                    trainer.train_device_kernels()

                for _ in range(updates_per_step):
                    maybe_capture_replay[_captured_step](train_graph, c)
                    trainer.note_train_update()
        else:
            for _ in range(updates_per_step):
                _ = trainer.train_step(base_step + step_idx)

        if verbose and print_every > 0 and step_idx >= next_print:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ",
                base_step + step_idx,
                "] mean_ret(10)=",
                trainer.mean_return(),
                " ep=",
                trainer.ep_count(),
                " elapsed=",
                elapsed,
                "s",
            )
            next_print += print_every

        # Logger emit at the same cadence (independent of verbose).
        # Comptime-elided when L=NoOpLogger (default).
        comptime if L.ENABLED:
            if (
                print_every > 0
                and step_idx >= next_log
                and Bool(logger)
            ):
                logger.value()[].log_scalar(
                    "avg_reward",
                    Float64(trainer.mean_return()),
                    base_step + step_idx,
                )
                logger.value()[].log_scalar(
                    "episodes",
                    Float64(trainer.ep_count()),
                    base_step + step_idx,
                )
                # Live flush so the dashboard updates DURING training. The
                # always-on stream is only 2 points per `print_every` — far
                # below `buffer_size` — so without this it would sit unsent
                # in the buffer until `logger.close()` (the symptom: a run
                # with no `diag_every` shows no remote logs until it ends).
                # `flush()` early-returns on an empty buffer, so pairing it
                # with the diag flush below is cheap (one POST per cadence).
                logger.value()[].flush()
                next_log += print_every

        # `diag_every` — drain the trainer's full metric bundle (mean_q,
        # critic_loss, alpha, train_steps, …) through the logger at its own
        # cadence, mirroring the single-env `run_offpolicy_train` driver.
        # Default trait impl is a no-op for trainers that haven't wired this
        # up; SACTrainer overrides it. A live `flush()` follows so the
        # dashboard updates DURING training (otherwise points only auto-flush
        # once the buffer hits `buffer_size`).
        comptime if L.ENABLED:
            if (
                diag_every > 0
                and step_idx >= next_diag
                and Bool(logger)
            ):
                trainer.flush_metrics_through_logger[L](
                    logger, base_step + step_idx
                )
                logger.value()[].flush()
                next_diag += diag_every

    # Defensive final drain of any buffered episode readbacks. With
    # `sync_every == 1` (or the last iteration hitting an emit boundary)
    # `pending_eps` is already 0; this only fires if the loop exited mid-ring.
    comptime if env_target == "gpu":
        if ctx and pending_eps > 0:
            var c = ctx.value()
            c.synchronize()
            for s in range(pending_eps):
                var rewards_h = ring_reward[s].unsafe_ptr()
                var dones_h = ring_done[s].unsafe_ptr()
                for e in range(N_ENVS):
                    per_env_returns[e] = per_env_returns[e] + rewards_h[e]
                    if dones_h[e] > Scalar[DT](0.5):
                        trainer.add_complete_return(per_env_returns[e])
                        per_env_returns[e] = Scalar[DT](0.0)
                        ep_returns.append(trainer.mean_return())

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_train_cpu_env_gpu_agent — Phase 6.1: batched
#   (cpu env, gpu train, N_ENVS>=1) GPU-agent / CPU-env hybrid.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_train_cpu_env_gpu_agent[
    A: OffPolicyAgentGpu,
    E: BatchedEnv,
    N_ENVS: Int = 1,
    L: Logger = NoOpLogger,
](
    ctx: DeviceContext,
    mut trainer: A,
    mut env: E,
    total_env_steps: Int,
    *,
    rng_seed: UInt64 = UInt64(42),
    updates_per_step: Int = 1,
    print_every: Int = 5_000,
    verbose: Bool = True,
    logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
    diag_every: Int = 0,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    base_step: Int = 0,
) raises -> List[Scalar[DT]]:
    """GPU-agent / CPU-env hybrid off-policy driver (Phase 6.1).

    Covers the one cross-target combination the other off-policy drivers
    deferred: `env_target == "cpu"` with `train_target == "gpu"` at any
    `N_ENVS >= 1`. Ported from the legacy
    `gpu_agent_cpu_env_train.mojo` (`run_offpolicy_continuous_train_cpu_env_gpu_agent`).

    Lets a GPU SAC/TD3/DDPG agent train against a CPU-stepped env (e.g. a
    Python Gymnasium MuJoCo env wrapped in `BatchedCpuEnv`) so a training
    failure can be attributed to "the env" or "the algorithm". The env
    steps on the host; obs/action/reward/terminated cross the H↔D
    boundary every iteration.

    Per-iteration loop:
      1. H2D current obs (env.obs_ptr → obs_dev) and D→D snapshot into
         prev_obs_dev (the `s_t` of the transition).
      2. `trainer.select_action_batched[N_ENVS]` on device (reads
         obs_dev, writes action_dev). GPU warmup uses the trainer's own
         Philox kernel, gated on `base_step + step_idx` vs
         `learning_starts`.
      3. D2H action (action_dev → env.action_ptr), then `synchronize()`
         so the host action slab is valid before the CPU env steps.
      4. `env.step_batch[N_ENVS]` on the host (writes host
         obs/reward/done/terminated).
      5. H2D next-obs / reward / terminated, then
         `trainer.record_batch_gpu[N_ENVS]`. The replay stores
         `terminated` (natural termination only) so the TD bootstrap is
         kept on time-limit truncation, dropped on real termination —
         the truncation-correct bootstrap Phase 6.1 calls for.
      6. `synchronize()` so all device reads of the env host buffers
         complete before the host mutates them in selective_reset.
      7. Per-env return accumulation; `add_complete_return` on `done`.
      8. `env.selective_reset_batch[N_ENVS]` on the host.
      9. `updates_per_step` × `trainer.train_step` (GPU).

    `ctx` MUST be the same `DeviceContext` the trainer was built with
    (Apple Metal's queue pool exhausts if a fresh context is constructed
    per call). The env's `reset_batch` / `step_batch` /
    `selective_reset_batch` receive `ctx=None` (CPU env ignores it).
    """
    comptime env_target: StaticString = E.ENV_TARGET
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM
    comptime ACT = A.AGENT_ACT_DIM

    comptime assert env_target == "cpu", (
        "run_offpolicy_train_cpu_env_gpu_agent: env_target must be 'cpu'"
        " (this is the CPU-env / GPU-agent hybrid; for a GPU env use"
        " run_offpolicy_train_batched)"
    )
    comptime assert train_target == "gpu", (
        "run_offpolicy_train_cpu_env_gpu_agent: train_target must be 'gpu'"
        " (for a CPU trainer use run_offpolicy_train_batched, which"
        " covers same-target cpu/cpu at any N_ENVS)"
    )
    comptime assert N_ENVS > 0, "N_ENVS must be > 0"
    comptime assert (
        E.OBS_DIM == OBS and E.ACT_DIM == ACT
    ), "BatchedEnv dimensions must match trainer dimensions"

    # All scratches live on the device (train_target == "gpu"). obs_dev
    # holds `s_t` then is overwritten with `s_{t+1}`; prev_obs_dev keeps
    # the pre-step `s_t` for the stored transition. reward_dev /
    # term_dev are H2D mirrors of the env's host reward / terminated
    # slabs. ao / alp are the actor-output + action-log-prob scratch the
    # trainer's select_action_batched writes through.
    var obs_dev = DriverScratch["he_obs", N_ENVS, OBS].make["gpu"](ctx=ctx)
    var prev_obs_dev = DriverScratch["he_prev_obs", N_ENVS, OBS].make["gpu"](
        ctx=ctx
    )
    var action_dev = DriverScratch["he_action", N_ENVS, ACT].make["gpu"](
        ctx=ctx
    )
    var reward_dev = DriverScratch["he_reward", N_ENVS, 1].make["gpu"](ctx=ctx)
    var term_dev = DriverScratch["he_term", N_ENVS, 1].make["gpu"](ctx=ctx)
    var ao = DriverScratch["he_ao", N_ENVS, 2 * ACT].make["gpu"](ctx=ctx)
    var alp = DriverScratch["he_alp", N_ENVS, ACT + 1].make["gpu"](ctx=ctx)

    var per_env_returns = List[Scalar[DT]](
        length=N_ENVS, fill=Scalar[DT](0.0),
    )

    # CPU env: ctx ignored. Wrap in Optional(None) for the trait sig.
    env.reset_batch[N_ENVS](ctx=None, rng_seed=rng_seed)

    var ep_returns = List[Scalar[DT]]()
    var t_start = perf_counter_ns()
    var step_idx: Int = 0
    var iter_idx: Int = 0
    var next_print: Int = print_every
    var next_log: Int = print_every

    while step_idx < total_env_steps:
        var abs_step = base_step + step_idx

        # ── 1. H2D current obs (s_t) + D→D snapshot into prev_obs.
        ctx.enqueue_copy(obs_dev.dev.value(), env.obs_ptr())
        ctx.enqueue_copy(prev_obs_dev.dev.value(), obs_dev.dev.value())

        # ── 2. Action selection on device (warmup gated inside trainer).
        trainer.select_action_batched[N_ENVS](
            obs_dev.dev_ptr(),
            action_dev.dev_ptr(),
            ao.dev_ptr(),
            alp.dev_ptr(),
            abs_step,
        )

        # ── 3. D2H action → env host slab; sync before CPU steps.
        ctx.enqueue_copy(env.action_ptr(), action_dev.dev.value())
        ctx.synchronize()

        # ── 4. Step CPU envs (writes host obs/reward/done/terminated).
        env.step_batch[N_ENVS](
            ctx=None, rng_seed=rng_seed + UInt64(iter_idx + 1),
        )

        # ── 5. H2D next-obs / reward / terminated, then GPU replay push.
        # `terminated` (natural termination only) is stored so the TD
        # bootstrap is kept on truncation, dropped on termination.
        ctx.enqueue_copy(obs_dev.dev.value(), env.obs_ptr())
        ctx.enqueue_copy(reward_dev.dev.value(), env.reward_ptr())
        ctx.enqueue_copy(term_dev.dev.value(), env.terminated_ptr())
        trainer.record_batch_gpu[N_ENVS](
            ctx,
            prev_obs_dev.dev.value(),
            action_dev.dev.value(),
            reward_dev.dev.value(),
            obs_dev.dev.value(),
            term_dev.dev.value(),
        )

        # ── 6. Sync so all device reads of the env host slabs finish
        # before selective_reset (host) mutates env.obs in place.
        ctx.synchronize()

        # ── 7. Per-env episode tracking (host reward + done, already
        # CPU-written by step_batch).
        var rewards_h = env.reward_ptr()
        var dones_h = env.done_ptr()
        for e in range(N_ENVS):
            per_env_returns[e] = per_env_returns[e] + rewards_h[e]
            if dones_h[e] > Scalar[DT](0.5):
                trainer.add_complete_return(per_env_returns[e])
                per_env_returns[e] = Scalar[DT](0.0)
                ep_returns.append(trainer.mean_return())

        # ── 8. Selective env reset (host).
        env.selective_reset_batch[N_ENVS](
            ctx=None, rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7),
        )

        step_idx += N_ENVS
        iter_idx += 1

        # ── 9. Trainer updates (GPU).
        for _ in range(updates_per_step):
            _ = trainer.train_step(base_step + step_idx)

        if verbose and print_every > 0 and step_idx >= next_print:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ",
                base_step + step_idx,
                "] mean_ret(10)=",
                trainer.mean_return(),
                " ep=",
                trainer.ep_count(),
                " elapsed=",
                elapsed,
                "s",
            )
            next_print += print_every

        # Logger emit at the print cadence (comptime-elided for NoOpLogger).
        comptime if L.ENABLED:
            if (
                print_every > 0
                and step_idx >= next_log
                and Bool(logger)
            ):
                logger.value()[].log_scalar(
                    "avg_reward",
                    Float64(trainer.mean_return()),
                    base_step + step_idx,
                )
                logger.value()[].log_scalar(
                    "episodes",
                    Float64(trainer.ep_count()),
                    base_step + step_idx,
                )
                next_log += print_every

        # `diag_every` — drain the trainer's metric bundle through the
        # logger. Default trait impl is a no-op for trainers that haven't
        # wired this up yet.
        comptime if L.ENABLED:
            if (
                diag_every > 0
                and (base_step + step_idx) % diag_every == 0
                and Bool(logger)
            ):
                trainer.flush_metrics_through_logger[L](
                    logger, base_step + step_idx
                )

        # `checkpoint_every` — overwrite `checkpoint_path` with the
        # trainer's one-file v2 envelope. Default trait impl is a no-op.
        if (
            checkpoint_every > 0
            and (base_step + step_idx) % checkpoint_every == 0
            and checkpoint_path.byte_length() > 0
        ):
            trainer.save_state(checkpoint_path)

    # Always overwrite the final checkpoint at end so resume gets the
    # freshest weights regardless of cadence alignment.
    if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
        trainer.save_state(checkpoint_path)

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
