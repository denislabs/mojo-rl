"""Shared cadence state for the BATCHED training drivers.

ONE copy of the print/log/diag/ckpt/eval threshold bookkeeping that every
batched driver previously carried inline (~60 lines each across the
off-policy continuous/discrete drivers, the two CPU-env/GPU-agent
hybrids, and the shared on-policy batched body).

Threshold counters, NOT `% cadence == 0`: `step_idx` advances by N_ENVS
per iteration, so a modulo check only fires when the cadence happens to
be divisible by N_ENVS (degraded to lcm intervals or never otherwise).
Each `*_due` method tests `step_idx >= next_*` and advances its counter
by one period when it fires. All cadences are relative to THIS call's
`step_idx` (not `base_step + step_idx`) — chunked facade callers restart
the cadence per chunk, exactly like the inline code did.

What stays at the call site (deliberately — it differs per driver):
  * the `comptime if L.ENABLED` gate + `Bool(logger)` check (so the
    NoOpLogger path never reads or advances the log counter),
  * the diag flush's x-axis (`abs_step` vs `trainer.total_train_steps()`),
  * whether the log emit force-flushes (`log_status[L, FLUSH=...]`),
  * the trainer calls themselves (flush_metrics / save_state / eval).

The SINGLE-ENV drivers keep their modulo (`abs_step % cadence == 0`)
gating and are NOT ported: their cadence is aligned to the CUMULATIVE
`base_step + step` counter, which chunked facade callers rely on.
"""

from std.time import perf_counter_ns

from mojo_rl.core.logger import Logger
from mojo_rl.nn.constants import DT
from mojo_rl.utils.progress import IntervalProgress


@fieldwise_init
struct DriverCadence(Movable):
    """Threshold-counter cadence state + status-line emission for a
    batched driver loop. Build with `make`; call the `*_due` methods in
    the same order the inline blocks ran."""

    var t_start: Int
    var prog: IntervalProgress
    var verbose: Bool
    var print_every: Int
    var next_print: Int
    var next_log: Int
    var diag_every: Int
    var next_diag: Int
    var ckpt_on: Bool
    var ckpt_every: Int
    var next_ckpt: Int
    var eval_on: Bool
    var eval_every: Int
    var next_eval: Int

    @staticmethod
    def make(
        print_every: Int,
        *,
        min_stride: Int = 1,
        label: String,
        verbose: Bool,
        diag_every: Int = 0,
        checkpoint_every: Int = 0,
        ckpt_enabled: Bool = False,
        eval_every: Int = 0,
        eval_enabled: Bool = False,
    ) raises -> Self:
        """`ckpt_enabled` / `eval_enabled` carry the caller's extra
        conditions (non-empty checkpoint path / eval env supplied)."""
        return Self(
            t_start=Int(perf_counter_ns()),
            prog=IntervalProgress(
                print_every,
                min_stride=min_stride,
                label=label,
                enabled=verbose,
            ),
            verbose=verbose,
            print_every=print_every,
            next_print=print_every,
            next_log=print_every,
            diag_every=diag_every,
            next_diag=diag_every,
            ckpt_on=checkpoint_every > 0 and ckpt_enabled,
            ckpt_every=checkpoint_every,
            next_ckpt=checkpoint_every,
            eval_on=eval_every > 0 and eval_enabled,
            eval_every=eval_every,
            next_eval=eval_every,
        )

    def tick(mut self, step_idx: Int, train_steps: Int):
        self.prog.tick(step_idx, train_steps)

    def clear(mut self):
        """Erase the in-place bar before an out-of-band status line
        (checkpoint / eval prints at the call site)."""
        self.prog.clear()

    # ─── due checks (advance their counter when they fire) ────────────

    def print_due(mut self, step_idx: Int) -> Bool:
        if self.verbose and self.print_every > 0 and step_idx >= self.next_print:
            self.next_print += self.print_every
            return True
        return False

    def log_due(mut self, step_idx: Int) -> Bool:
        """Call ONLY inside `comptime if L.ENABLED` and after a
        `Bool(logger)` check, so the NoOpLogger / no-logger paths never
        advance the counter (matches the inline blocks)."""
        if self.print_every > 0 and step_idx >= self.next_log:
            self.next_log += self.print_every
            return True
        return False

    def diag_due(mut self, step_idx: Int) -> Bool:
        """Same call-site contract as `log_due`."""
        if self.diag_every > 0 and step_idx >= self.next_diag:
            self.next_diag += self.diag_every
            return True
        return False

    def ckpt_due(mut self, step_idx: Int) -> Bool:
        if self.ckpt_on and step_idx >= self.next_ckpt:
            self.next_ckpt += self.ckpt_every
            return True
        return False

    def eval_due(mut self, step_idx: Int) -> Bool:
        if self.eval_on and step_idx >= self.next_eval:
            self.next_eval += self.eval_every
            return True
        return False

    # ─── boundary peeks (do NOT advance — for emit_now decisions) ─────

    def emit_boundary_imminent(self, post_step: Int, total: Int) -> Bool:
        """True when the POST-increment step is at/past any armed cadence
        boundary or the end of the run — the `emit_now` input to
        `EpisodeReturnRing.due` (drain before the boundary's readers)."""
        if post_step >= total:
            return True
        if self.print_every > 0 and post_step >= self.next_print:
            return True
        if self.diag_every > 0 and post_step >= self.next_diag:
            return True
        if self.ckpt_on and post_step >= self.next_ckpt:
            return True
        if self.eval_on and post_step >= self.next_eval:
            return True
        return False

    # ─── status emission ──────────────────────────────────────────────

    def print_status(mut self, abs_step: Int, mean_ret: Scalar[DT], ep: Int):
        """The `[step N] mean_ret(10)= …` line every driver printed."""
        self.prog.clear()
        var elapsed = Float64(Int(perf_counter_ns()) - self.t_start) / 1e9
        print(
            "[step ",
            abs_step,
            "] mean_ret(10)=",
            mean_ret,
            " ep=",
            ep,
            " elapsed=",
            elapsed,
            "s",
        )

    def log_status[
        L: Logger, FLUSH: Bool
    ](
        self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        abs_step: Int,
        mean_ret: Scalar[DT],
        ep: Int,
    ) raises:
        """The always-on `avg_reward` / `episodes` emit. `FLUSH=True`
        force-flushes at this (rare) cadence so the dashboard updates
        during training even when `diag_every == 0` — the off-policy
        batched drivers' behaviour; the on-policy + single-env drivers
        rely on `buffer_size` auto-flush (`FLUSH=False`)."""
        logger.value()[].log_scalar("avg_reward", Float64(mean_ret), abs_step)
        logger.value()[].log_scalar("episodes", Float64(ep), abs_step)
        comptime if FLUSH:
            logger.value()[].flush()
