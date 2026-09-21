"""In-place CLI progress bar — pure CPU, no GPU sync."""


def print_progress_bar(
    current: Int,
    total: Int,
    train_steps: Int,
    algorithm_name: String,
    bar_width: Int = 30,
):
    """Print an in-place progress bar using carriage return (no GPU sync).

    Uses only CPU-side counters so it adds zero overhead to GPU training.

    Args:
        current: Current step count.
        total: Target step count.
        train_steps: Total gradient updates so far.
        algorithm_name: Algorithm name prefix.
        bar_width: Width of the bar in characters (default 30).
    """
    var pct = current * 100 // total
    var filled = current * bar_width // total
    var bar = String("")
    for i in range(bar_width):
        if i < filled:
            bar += "█"
        else:
            bar += "░"
    print(
        "\r"
        + algorithm_name
        + " ["
        + bar
        + "] "
        + String(pct)
        + "% | Step "
        + String(current)
        + "/"
        + String(total)
        + " | Train: "
        + String(train_steps),
        end="",
    )


def _fmt1(x: Float64) -> String:
    """Fixed one-decimal format ("12.3") — String(Float64) prints full
    precision, which is unreadable on an in-place bar."""
    var neg = x < 0.0
    var ax = -x if neg else x
    var scaled = Int(ax * 10.0 + 0.5)
    var s = String(scaled // 10) + "." + String(scaled % 10)
    return "-" + s if neg else s


def print_bytes_progress(
    label: String,
    done_bytes: Int,
    total_bytes: Int,
    elapsed_s: Float64,
    bar_width: Int = 30,
):
    """In-place byte-transfer bar (downloads, file copies): percent, GB
    done/total, average MB/s and ETA from the average rate. Call once per
    chunk; finish with a plain `print()` to keep the last bar line."""
    var total = total_bytes if total_bytes > 0 else 1
    var pct = done_bytes * 100 // total
    var filled = done_bytes * bar_width // total
    var bar = String("")
    for i in range(bar_width):
        bar += "█" if i < filled else "░"
    var mbs = (
        Float64(done_bytes) / 1e6 / elapsed_s if elapsed_s > 0.0 else 0.0
    )
    var eta_s = 0
    if done_bytes > 0 and mbs > 0.0:
        eta_s = Int(Float64(total - done_bytes) / 1e6 / mbs)
    print(
        "\r"
        + label
        + " ["
        + bar
        + "] "
        + String(pct)
        + "% | "
        + _fmt1(Float64(done_bytes) / 1e9)
        + "/"
        + _fmt1(Float64(total) / 1e9)
        + " GB | "
        + _fmt1(mbs)
        + " MB/s | ETA "
        + String(eta_s // 60)
        + "m"
        + String(eta_s % 60)
        + "s   ",
        end="",
    )


def clear_progress_bar():
    """Overwrite the current progress bar line with spaces and return to start.

    Call this before printing stats to ensure the progress bar is fully erased.
    """
    # 120 spaces is enough to cover any progress bar output
    print("\r" + String(" ") * 120 + "\r", end="")


struct IntervalProgress(Copyable, Movable):
    """Drop-in within-log-interval progress bar for training loops.

    Renders an in-place bar that fills 0% → 100% across each
    `print_every` window, so the loop is never "blind" between two
    stats lines. Pure CPU counters → zero GPU sync. Mirrors the old
    `deep_agents` driver behaviour (see
    `core/training/gpu_offpolicy_train.mojo`).

    Usage:
      ```
      var prog = IntervalProgress(print_every, min_stride=N_ENVS,
                                  label="SAC", enabled=verbose)
      while ...:
          ...
          prog.tick(step_idx, trainer.total_train_steps())
          if verbose and step_idx >= next_print:
              prog.clear()       # erase bar before the stats line
              print("[step ...] ...")
      ```

    The bar position is `cur_step % print_every`, which lines up with
    both the modulo print gate (single-env) and the `next_print`
    counter gate (batched, since boundaries are multiples of
    `print_every`). Pass the loop's env-step counter to `tick` — the
    same value the stats-print boundary is keyed on.
    """

    var print_every: Int
    var stride: Int
    var next_tick: Int
    var label: String
    var enabled: Bool

    def __init__(
        out self,
        print_every: Int,
        *,
        stride_div: Int = 20,
        min_stride: Int = 1,
        label: String = "train",
        enabled: Bool = True,
    ):
        """Build an interval progress bar.

        Args:
            print_every: Env-step log interval (the bar resets each one).
            stride_div: Number of bar updates per interval (default ~20).
            min_stride: Floor on the update stride in env steps (pass
                `N_ENVS` for batched loops so the bar never updates more
                than once per iteration).
            label: Algorithm name prefix shown on the bar.
            enabled: Master switch (pass the driver's `verbose` flag).
        """
        self.print_every = print_every
        var s = print_every // stride_div
        if s < min_stride:
            s = min_stride
        if s < 1:
            s = 1
        self.stride = s
        self.next_tick = s
        self.label = label
        self.enabled = enabled and print_every > 0

    def tick(mut self, cur_step: Int, train_steps: Int):
        """Maybe redraw the bar for the current env-step count.

        Cheap to call every iteration: redraws only when `cur_step`
        crosses the next stride boundary. `train_steps` is the
        cumulative gradient-update count shown in the bar's `Train:`
        field (pass `trainer.total_train_steps()`; 0 is fine for
        trainers that don't track it).
        """
        if not self.enabled:
            return
        if cur_step >= self.next_tick:
            var pos = cur_step % self.print_every
            print_progress_bar(pos, self.print_every, train_steps, self.label)
            self.next_tick += self.stride

    def clear(self):
        """Erase the bar line. Call before printing a full stats line."""
        if self.enabled:
            clear_progress_bar()
