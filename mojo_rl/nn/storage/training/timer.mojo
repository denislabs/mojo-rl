"""Timer — per-section wall-time accumulator for trainers (storage surface).

Lightweight introspection helper: each trainer attaches a Timer and accumulates
wall-time into N labeled sections. Caller drives the mark → accumulate cadence:

    var timer = Timer.new()
    timer.add_section("target_y")
    timer.add_section("critic")
    timer.add_section("actor")
    # ... in train_step:
    var t0 = perf_counter_ns()
    self.compute_target_y(...)
    timer.accumulate(0, t0)

    var t1 = perf_counter_ns()
    self.critic_update(...)
    timer.accumulate(1, t1)
    # ...

Hot-path overhead is one `perf_counter_ns()` + one List index + one UInt
subtraction per timed section per call — roughly 30 ns on Apple Silicon. For
sub-steps taking >30 μs (every realistic training block), overhead is well under
0.1 %.

Use `format_report()` to render a multi-line readable summary, or the per-section
accessors `total_seconds(idx)` / `mean_ms(idx)` / `call_count(idx)` to feed
external logging.

Framework-agnostic (pure `perf_counter_ns` + the shared `DT` scalar alias) —
moved verbatim from the legacy `nn/training/timer.mojo` so the storage-migrated
deep_agents trainers no longer import the legacy training package.
"""

from std.time import perf_counter_ns

from mojo_rl.nn.constants import DT


@fieldwise_init
struct Timer(Movable & ImplicitlyDeletable):
    """Section-indexed wall-time accumulator. Labels are declared via
    `add_section`; index order matches declaration order."""

    var times_ns: List[UInt]
    var counts: List[Int]
    var labels: List[String]

    @staticmethod
    def new() -> Self:
        return Self(
            times_ns=List[UInt](),
            counts=List[Int](),
            labels=List[String](),
        )

    def add_section(mut self, label: String):
        """Append one zero-initialised section with the given label."""
        self.times_ns.append(UInt(0))
        self.counts.append(0)
        self.labels.append(label)

    @always_inline
    def accumulate(mut self, idx: Int, start_ns: UInt):
        """Add (now − start_ns) to section `idx`, increment its count."""
        var end_ns = perf_counter_ns()
        self.times_ns[idx] += end_ns - start_ns
        self.counts[idx] += 1

    def reset(mut self):
        """Zero every section's accumulator and call count."""
        for i in range(len(self.times_ns)):
            self.times_ns[i] = UInt(0)
            self.counts[i] = 0

    def total_seconds(self, idx: Int) -> Scalar[DT]:
        return Scalar[DT](Float64(self.times_ns[idx]) * 1e-9)

    def mean_ms(self, idx: Int) -> Scalar[DT]:
        if self.counts[idx] == 0:
            return Scalar[DT](0)
        return Scalar[DT](
            Float64(self.times_ns[idx]) * 1e-6 / Float64(self.counts[idx])
        )

    def call_count(self, idx: Int) -> Int:
        return self.counts[idx]

    def n_sections(self) -> Int:
        return len(self.times_ns)

    def format_report(self) -> String:
        """Multi-line "label: total_s (mean_ms/call, N calls)" report."""
        var out = String("")
        for i in range(len(self.times_ns)):
            out = out + self.labels[i] + ": "
            out = out + String(self.total_seconds(i)) + " s ("
            out = out + String(self.mean_ms(i)) + " ms/call, "
            out = out + String(self.counts[i]) + " calls)\n"
        return out
