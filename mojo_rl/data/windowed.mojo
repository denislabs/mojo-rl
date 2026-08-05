# +--------------------------------------------------------------------------+ #
# | Windowed residency — regime B, for stores that do not fit in memory
# +--------------------------------------------------------------------------+ #
"""A sliding resident window over one column of a `TrajectoryStore`.

**What this is NOT.** It does not make uniform random sampling work on a store
too large to hold. That is not a solvable problem: a uniform minibatch over
2.3 M rows touches 4096 scattered rows, so any window either contains them
(and is the whole store) or misses most of them (and every miss is a disk
read). §3b measured that pattern at 272 ms per minibatch against 0.060 ms
resident — 4500x. A window cannot repair it.

**What it is.** Support for samplers whose access pattern has LOCALITY:

  * sequence windows — a length-T clip is T contiguous rows;
  * shard-shuffled training — draw uniformly *within* the resident window,
    then advance. This is how large-dataset training actually works, and it
    trades a small amount of sampling independence for two orders of
    magnitude of throughput.

The trade is explicit rather than hidden: `draw_in_window` samples the
resident rows only, and moving on is a separate, deliberate `advance()`. A
silent "refill on miss" API would have looked like uniform sampling while
performing like disk.

**Sizing.** Every dataset mojo-rl generates is regime A (walker at 10 M rows
is 992 MiB — load it whole with `ResidentColumn` and ignore this file). This
exists for datasets we CONSUME: PushT is 2,336,736 rows of 224x224x3 uint8,
~44 GB.
"""

from std.memory import alloc
from std.random import random_float64

from .column import ColumnSpec, dtype_bytes
from .resident import IDX_DT, IndexBatch
from .store import TrajectoryStore


struct WindowedColumn[dtype: DType](Movable & ImplicitlyDeletable):
    """One column, with rows `[window_start, window_start + window_rows)`
    resident.

    Holds no reference to the store — the store is passed to the methods that
    need it. That keeps this struct movable and avoids owning an `H5File`
    handle whose lifetime would then have to match the window's.
    """

    var host: List[Scalar[Self.dtype]]
    var name: String
    var n_rows: Int
    """Rows in the whole column, not in the window."""
    var row_dim: Int
    var window_rows: Int
    var window_start: Int
    var refills: Int
    """Slab reads performed. A sampler that thrashes shows up here."""

    def __init__(
        out self,
        var host: List[Scalar[Self.dtype]],
        var name: String,
        n_rows: Int,
        row_dim: Int,
        window_rows: Int,
    ):
        self.host = host^
        self.name = name^
        self.n_rows = n_rows
        self.row_dim = row_dim
        self.window_rows = window_rows
        self.window_start = -1
        self.refills = 0

    def __init__(out self, *, deinit move: Self):
        self.host = move.host^
        self.name = move.name^
        self.n_rows = move.n_rows
        self.row_dim = move.row_dim
        self.window_rows = move.window_rows
        self.window_start = move.window_start
        self.refills = move.refills

    @staticmethod
    def make(
        store: TrajectoryStore, name: String, window_rows: Int
    ) raises -> Self:
        var spec = store.column(name)
        if spec.dtype != Self.dtype:
            raise Error(
                "WindowedColumn: column '" + name + "' is "
                + String(spec.dtype) + " but was opened as "
                + String(Self.dtype)
            )
        if window_rows <= 0:
            raise Error("WindowedColumn: window_rows must be > 0")
        var w = window_rows
        if w > store.n_rows():
            w = store.n_rows()
        var buf = List[Scalar[Self.dtype]](
            unsafe_uninit_length=w * spec.row_dim()
        )
        return Self(buf^, String(name), store.n_rows(), spec.row_dim(), w)

    def window_end(self) -> Int:
        return self.window_start + self.window_rows

    def resident(self, row: Int) -> Bool:
        if self.window_start < 0:
            return False
        return row >= self.window_start and row < self.window_end()

    def bytes_resident(self) raises -> Int:
        return self.window_rows * self.row_dim * dtype_bytes(Self.dtype)

    def seek(mut self, store: TrajectoryStore, start: Int) raises:
        """Make `[start, start + window_rows)` resident via ONE contiguous
        slab read — the access pattern §3b measured at GiB/s."""
        var s = start
        if s < 0:
            s = 0
        var max_start = self.n_rows - self.window_rows
        if max_start < 0:
            max_start = 0
        if s > max_start:
            s = max_start
        if self.window_start == s:
            return
        store.read_range[Self.dtype](
            self.name, s, s + self.window_rows,
            self.host.unsafe_ptr().as_unsafe_any_origin(),
        )
        self.window_start = s
        self.refills += 1

    def advance(mut self, store: TrajectoryStore) raises -> Bool:
        """Slide to the next non-overlapping window. Returns False when the
        column is exhausted (the caller decides whether to wrap)."""
        if self.window_start < 0:
            self.seek(store, 0)
            return True
        var nxt = self.window_start + self.window_rows
        if nxt + self.window_rows > self.n_rows:
            return False
        self.seek(store, nxt)
        return True

    def gather_host(
        self, ref idx: IndexBatch, mut out: List[Scalar[Self.dtype]]
    ) raises:
        """`out[i, :] = column[idx[i], :]` for rows inside the window.

        Raises on a row outside it — deliberately. Silently refilling per
        miss would turn one minibatch into thousands of slab reads while
        still looking like a gather.
        """
        if self.window_start < 0:
            raise Error(
                "WindowedColumn.gather_host: no window resident — call seek()"
                " or advance() first"
            )
        var batch = len(idx.host)
        var need = batch * self.row_dim
        if len(out) != need:
            out = List[Scalar[Self.dtype]](unsafe_uninit_length=need)
        for i in range(batch):
            var r = Int(idx.host[i])
            if not self.resident(r):
                raise Error(
                    "WindowedColumn.gather_host: row " + String(r)
                    + " is outside the resident window ["
                    + String(self.window_start) + ", "
                    + String(self.window_end()) + "). Windowed residency"
                    " serves samplers with LOCALITY; a scattered draw needs"
                    " the whole column resident (see this module's docstring)."
                )
            var src = (r - self.window_start) * self.row_dim
            var dst = i * self.row_dim
            for d in range(self.row_dim):
                out[dst + d] = self.host[src + d]


struct WindowSampler(Movable & ImplicitlyDeletable):
    """Uniform draw restricted to a resident window.

    The locality-respecting counterpart of `UniformSampler`. Sampling is
    uniform *within* the window; coverage of the whole store comes from
    advancing it. That is shard-shuffling, and the loss of independence is
    the price of not paying 272 ms per minibatch.
    """

    var window_start: Int
    var window_rows: Int

    def __init__(out self, window_start: Int, window_rows: Int):
        self.window_start = window_start
        self.window_rows = window_rows

    def __init__(out self, *, deinit move: Self):
        self.window_start = move.window_start
        self.window_rows = move.window_rows

    def draw(self, batch: Int) raises -> IndexBatch:
        if batch <= 0:
            raise Error("WindowSampler.draw: batch must be > 0")
        if self.window_rows <= 0:
            raise Error("WindowSampler.draw: empty window")
        var h = List[Scalar[IDX_DT]](unsafe_uninit_length=batch)
        for k in range(batch):
            # Same draw arithmetic as `UniformSampler`, offset into the
            # window — including the clamp, so the two agree on edge cases.
            var off = Int(random_float64() * Float64(self.window_rows))
            if off >= self.window_rows:
                off = self.window_rows - 1
            h[k] = Scalar[IDX_DT](self.window_start + off)
        return IndexBatch(h^)
