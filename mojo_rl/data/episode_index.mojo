# +--------------------------------------------------------------------------+ #
# | EpisodeIndex — the ragged structure over a flat row axis
# +--------------------------------------------------------------------------+ #
"""Episodes as `(ep_len, ep_offset)` over flat columns.

Columns are stored flat — one long row axis — and the episode structure lives
beside them as two small arrays. This is the layout the PushT/`stable_worldmodel`
files already use, and it is the right one: flat columns keep every bulk read
contiguous (measured 0.5-5 GiB/s, versus 43-58 MiB/s for strided reads), while
the index makes episode boundaries a lookup rather than a scan.

Boundaries are what stop a sampler from spanning two episodes — a
sequence-window sampler that ignores them silently trains on transitions that
never happened.
"""


struct EpisodeIndex(Movable & ImplicitlyDeletable):
    var ep_len: List[Int64]
    var ep_offset: List[Int64]

    def __init__(out self):
        self.ep_len = List[Int64]()
        self.ep_offset = List[Int64]()

    def __init__(out self, var ep_len: List[Int64], var ep_offset: List[Int64]):
        self.ep_len = ep_len^
        self.ep_offset = ep_offset^

    def __init__(out self, *, deinit move: Self):
        self.ep_len = move.ep_len^
        self.ep_offset = move.ep_offset^

    def n_episodes(self) -> Int:
        return len(self.ep_len)

    def total_rows(self) -> Int:
        var n = 0
        for i in range(len(self.ep_len)):
            n += Int(self.ep_len[i])
        return n

    def start_of(self, ep: Int) raises -> Int:
        if ep < 0 or ep >= len(self.ep_offset):
            raise Error("EpisodeIndex: episode out of range: " + String(ep))
        return Int(self.ep_offset[ep])

    def length_of(self, ep: Int) raises -> Int:
        if ep < 0 or ep >= len(self.ep_len):
            raise Error("EpisodeIndex: episode out of range: " + String(ep))
        return Int(self.ep_len[ep])

    def end_of(self, ep: Int) raises -> Int:
        """Exclusive end row of `ep`."""
        return self.start_of(ep) + self.length_of(ep)

    def episode_of(self, row: Int) raises -> Int:
        """Which episode a flat row belongs to. Binary search over offsets.

        Raises rather than clamping: a row outside every episode means the
        index and the columns disagree, which is a corrupt store, not a
        recoverable condition.
        """
        var n = len(self.ep_offset)
        if n == 0:
            raise Error("EpisodeIndex: empty index")
        var lo = 0
        var hi = n - 1
        var found = -1
        while lo <= hi:
            var mid = (lo + hi) // 2
            var s = Int(self.ep_offset[mid])
            var e = s + Int(self.ep_len[mid])
            if row < s:
                hi = mid - 1
            elif row >= e:
                lo = mid + 1
            else:
                found = mid
                break
        if found < 0:
            raise Error(
                "EpisodeIndex: row " + String(row) + " is in no episode"
            )
        return found

    def window_fits(self, row: Int, span: Int) raises -> Bool:
        """True when `[row, row+span)` stays inside one episode.

        The guard a sequence-window sampler needs.
        """
        if span <= 0:
            raise Error("EpisodeIndex.window_fits: span must be > 0")
        var ep = self.episode_of(row)
        return row + span <= self.end_of(ep)

    def validate(self, n_rows: Int) raises:
        """Check the index is internally consistent and covers exactly
        `n_rows`. Called on open — a store whose index disagrees with its
        columns produces samplers that read the wrong rows, silently."""
        if len(self.ep_len) != len(self.ep_offset):
            raise Error(
                "EpisodeIndex: ep_len has " + String(len(self.ep_len))
                + " entries but ep_offset has " + String(len(self.ep_offset))
            )
        var expect = 0
        for i in range(len(self.ep_len)):
            var l = Int(self.ep_len[i])
            if l <= 0:
                raise Error(
                    "EpisodeIndex: episode " + String(i) + " has length "
                    + String(l)
                )
            if Int(self.ep_offset[i]) != expect:
                raise Error(
                    "EpisodeIndex: episode " + String(i) + " offset is "
                    + String(Int(self.ep_offset[i])) + ", expected "
                    + String(expect) + " (episodes must be contiguous and"
                    " in order)"
                )
            expect += l
        if expect != n_rows:
            raise Error(
                "EpisodeIndex: episodes cover " + String(expect)
                + " rows but the columns hold " + String(n_rows)
            )
