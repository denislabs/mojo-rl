"""Reusable GPU performance timing system.

Provides a comptime-gated PerfTimer struct that accumulates nanosecond
timings into dynamically-added slots. When ENABLED=False, all methods
are no-ops and the compiler eliminates them entirely (zero overhead).

Slots support parent-child relationships for hierarchical display:
  - add_slot("label") creates a top-level slot
  - add_slot("label", parent=idx) creates a child slot under parent idx
  - print_report() renders children indented under their parent

Usage:
    var timer = PerfTimer[True]()
    var s0 = timer.add_slot("phase_a")
    var s1 = timer.add_slot("phase_b")
    var s1a = timer.add_slot("sub_b1", parent=s1)
    var s1b = timer.add_slot("sub_b2", parent=s1)

    timer.sync_and_mark(ctx)
    do_phase_a(ctx)
    timer.sync_and_accumulate(s0, ctx)
    timer.mark()
    do_sub_b1(ctx)
    timer.sync_and_accumulate(s1a, ctx)
    timer.mark()
    do_sub_b2(ctx)
    timer.sync_and_accumulate(s1b, ctx)
    # s1 accumulates the sum of s1a + s1b (or time it separately)

    timer.print_report("My Profile")
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from mojo_rl.core.fmt import fit


struct PerfTimer[ENABLED: Bool]:
    """Compile-time gated performance timer with dynamic slots.

    When ENABLED=False, all methods are no-ops (zero overhead).
    When ENABLED=True, mark/accumulate pairs measure elapsed wall-clock
    nanoseconds between calls.

    Parameters:
        ENABLED: Whether timing is active. False = all no-ops.
    """

    var accum_ns: List[Int]
    var counts: List[Int]
    var labels: List[String]
    var parents: List[Int]  # -1 = top-level, otherwise parent slot index
    var _mark: Int

    def __init__(out self):
        self.accum_ns = List[Int]()
        self.counts = List[Int]()
        self.labels = List[String]()
        self.parents = List[Int]()
        self._mark = 0

    def add_slot(mut self, label: String, parent: Int = -1) -> Int:
        """Add a timing slot. Returns its index for use with accumulate().

        Args:
            label: Human-readable name for this phase.
            parent: Parent slot index (-1 = top-level).

        Returns:
            Index of the new slot.
        """
        var idx = len(self.accum_ns)
        self.accum_ns.append(0)
        self.counts.append(0)
        self.labels.append(label)
        self.parents.append(parent)
        return idx

    def mark(mut self):
        """Record start timestamp (CPU-side, no GPU sync)."""
        comptime if Self.ENABLED:
            self._mark = perf_counter_ns()

    def accumulate(mut self, idx: Int):
        """Add elapsed ns since last mark() to slot idx."""
        comptime if Self.ENABLED:
            var now = perf_counter_ns()
            self.accum_ns[idx] += now - self._mark
            self.counts[idx] += 1

    def sync_and_mark(mut self, ctx: DeviceContext) raises:
        """Synchronize GPU then record start timestamp."""
        comptime if Self.ENABLED:
            ctx.synchronize()
            self._mark = perf_counter_ns()

    def sync_and_accumulate(mut self, idx: Int, ctx: DeviceContext) raises:
        """Synchronize GPU then accumulate elapsed ns to slot idx."""
        comptime if Self.ENABLED:
            ctx.synchronize()
            var now = perf_counter_ns()
            self.accum_ns[idx] += now - self._mark
            self.counts[idx] += 1

    def total(self) -> Int:
        """Sum all top-level accumulator slots."""
        var s: Int = 0
        for i in range(len(self.accum_ns)):
            if self.parents[i] == -1:
                s += self.accum_ns[i]
        return s

    def merge_children(mut self, parent_idx: Int, other: Self):
        """Merge all slots from other as children of parent_idx.

        Top-level slots in `other` become children of `parent_idx` in self.
        Nested children in `other` are remapped to maintain hierarchy.
        """
        comptime if Self.ENABLED:
            var base = len(self.accum_ns)
            for i in range(len(other.accum_ns)):
                self.accum_ns.append(other.accum_ns[i])
                self.counts.append(other.counts[i])
                self.labels.append(other.labels[i])
                if other.parents[i] == -1:
                    self.parents.append(parent_idx)
                else:
                    self.parents.append(other.parents[i] + base)

    def merge_children_range(
        mut self, parent_idx: Int, other: Self, start: Int, end: Int
    ):
        """Merge slots [start, end) from other as children of parent_idx."""
        comptime if Self.ENABLED:
            for i in range(start, end):
                self.accum_ns.append(other.accum_ns[i])
                self.counts.append(other.counts[i])
                self.labels.append(other.labels[i])
                self.parents.append(parent_idx)

    def merge_subtree_range(
        mut self, parent_idx: Int, other: Self, start: Int, end: Int
    ):
        """Merge slots [start, end) from other as children of parent_idx,
        plus all their descendants with proper parent remapping."""
        comptime if Self.ENABLED:
            var base = len(self.accum_ns)
            # Pass 1: copy requested range as children of parent_idx
            for i in range(start, end):
                self.accum_ns.append(other.accum_ns[i])
                self.counts.append(other.counts[i])
                self.labels.append(other.labels[i])
                self.parents.append(parent_idx)
            # Pass 2: copy all slots whose parent is in [start, end)
            for i in range(len(other.accum_ns)):
                if i >= start and i < end:
                    continue
                var p = other.parents[i]
                if p >= start and p < end:
                    self.accum_ns.append(other.accum_ns[i])
                    self.counts.append(other.counts[i])
                    self.labels.append(other.labels[i])
                    self.parents.append(p - start + base)  # remap parent

    def print_report(self, title: String = "Performance Profile"):
        """Print hierarchical performance report.

        Top-level slots show percentage relative to total.
        Child slots show percentage relative to their parent.
        Supports arbitrary nesting depth.
        """
        comptime if Self.ENABLED:
            var sep = String(
                "------------------------------------------------------------"
            )
            print(sep)
            print(title)
            print(sep)

            var total_ns = self.total()

            # Print top-level slots, with children recursively indented
            for i in range(len(self.accum_ns)):
                if self.parents[i] != -1:
                    continue  # skip children, they're printed under parent
                _print_slot(
                    self.labels[i], self.accum_ns[i], total_ns, indent=2
                )
                self._print_children(i, self.accum_ns[i], depth=1)

            print(sep)
            var total_ms = Float64(total_ns) / 1_000_000.0
            print("  Total:" + _pad_to(24 - "Total:".byte_length()) + _fmt_ms(total_ms))
            print(sep)

    def _print_children(self, parent: Int, ref_ns: Int, depth: Int):
        """Recursively print children of a slot with increasing indentation."""
        comptime if Self.ENABLED:
            for j in range(len(self.accum_ns)):
                if self.parents[j] == parent:
                    _print_slot(
                        self.labels[j],
                        self.accum_ns[j],
                        ref_ns,
                        indent=2 + depth * 4,
                    )
                    self._print_children(j, self.accum_ns[j], depth + 1)


def _fmt_ms(ms: Float64) -> String:
    """Format milliseconds with consistent width."""
    return fit(String(ms), 9) + "ms"


def _fmt_pct(pct: Float64) -> String:
    """Format percentage."""
    return "(" + fit(String(pct), 4) + "%)"


def _pad_to(n: Int) -> String:
    """Return n spaces."""
    var s = String("")
    for _ in range(n):
        s += " "
    return s


def _print_slot(label: String, ns: Int, ref_ns: Int, indent: Int):
    """Print a single slot line with proper formatting."""
    var ms = Float64(ns) / 1_000_000.0
    var pct: Float64 = 0.0
    if ref_ns > 0:
        pct = Float64(ns) / Float64(ref_ns) * 100.0

    var pad_len = 24 - label.byte_length()
    if pad_len < 1:
        pad_len = 1

    print(
        _pad_to(indent)
        + label
        + _pad_to(pad_len)
        + _fmt_ms(ms)
        + "  "
        + _fmt_pct(pct)
    )
