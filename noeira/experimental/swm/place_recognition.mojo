"""Learned place recognition — the last oracle, and the acknowledged weak link.

Every result from Phase 3 onward assumed an ORACLE place identity. The design
doc flags this as the weak point (it is SLAM's weak point too) and notes that E1
is adversarial for it in a specific way: after an ODD lap the same place is seen
with a MIRRORED frame, so a similarity computed on the whole latent misses
exactly the revisits that create the informative cycle. Recognition therefore
has to be invariant to the frame.

The frame channel cannot supply that invariance, and not for want of trying:
under the reflection `u -> H u`, and the only O(2)-invariant of a vector is its
norm, which is near-constant here. So the frame carries essentially no
frame-invariant place information — which is the design's own claim read in the
other direction, and it makes the CONTENT channel the place cue.

That would be a suspiciously easy conclusion on E1 as originally built, where
every cell has its own distinctive texture and content-based recognition is
correct by construction. So the interesting condition is texture ALIASING
(`MobiusConfig.aliased_mobius`): cells `c` and `c + N/2` share a texture, and a
content-only recogniser must then produce FALSE identifications. Those are worth
having, because a false identification creates a SPURIOUS CYCLE, which is
precisely the input the classification table's ABERRANT row exists for — and it
lets G9's "never file a fault as a world fact" be re-checked against a learned
recogniser rather than an injected fault.
"""

from std.math import sqrt

comptime MATCH_NONE: Int = -1


@fieldwise_init
struct RecogniserStats(Copyable, ImplicitlyCopyable, Movable):
    """Counts, never bare rates — a rate with no denominator hides vacuity."""

    var queries: Int
    var proposed: Int
    """How many queries produced a match at all."""
    var correct: Int
    """Matches whose stored place is genuinely the query's place."""
    var wrong: Int
    """Matches to a DIFFERENT place — these create spurious cycles."""
    var missed: Int
    """Queries whose place was in memory but produced no match."""
    var correct_parity_0: Int
    var n_parity_0: Int
    var correct_parity_1: Int
    var n_parity_1: Int


struct PlaceMemory[LAT: Int, D: Int, dtype: DType = DType.float64](
    Copyable, Movable
):
    """Stored place encodings, queried by appearance.

    Ground-truth ids are kept ALONGSIDE for scoring only; nothing in the
    matching reads them.
    """

    comptime CONTENT_DIM: Int = Self.LAT - Self.D

    var lat: List[Scalar[Self.dtype]]
    var truth_place: List[Int]
    var truth_parity: List[Int]

    def __init__(out self):
        self.lat = List[Scalar[Self.dtype]]()
        self.truth_place = List[Int]()
        self.truth_parity = List[Int]()

    def __init__(out self, *, copy: Self):
        self.lat = copy.lat.copy()
        self.truth_place = copy.truth_place.copy()
        self.truth_parity = copy.truth_parity.copy()

    def __init__(out self, *, deinit move: Self):
        self.lat = move.lat^
        self.truth_place = move.truth_place^
        self.truth_parity = move.truth_parity^

    def size(self) -> Int:
        return len(self.truth_place)

    def add(
        mut self,
        lat: List[Scalar[Self.dtype]],
        place: Int,
        parity: Int,
    ):
        for i in range(Self.LAT):
            self.lat.append(lat[i])
        self.truth_place.append(place)
        self.truth_parity.append(parity)

    def _dist(
        self,
        idx: Int,
        query: List[Scalar[Self.dtype]],
        content_only: Bool,
    ) -> Float64:
        var start = 0 if not content_only else Self.D
        var stop = Self.LAT
        var d = Float64(0)
        for i in range(start, stop):
            var e = Float64(self.lat[idx * Self.LAT + i] - query[i])
            d += e * e
        return sqrt(d)

    def query(
        self,
        lat: List[Scalar[Self.dtype]],
        threshold: Float64,
        content_only: Bool,
    ) -> Int:
        """Nearest stored entry within `threshold`, or `MATCH_NONE`.

        `content_only=False` is the naive baseline the design doc predicts will
        fail: it compares the whole latent, so a revisit at the opposite lap
        parity looks far away even though it is the same place.
        """
        var best = MATCH_NONE
        var best_d = threshold
        for i in range(self.size()):
            var d = self._dist(i, lat, content_only)
            if d < best_d:
                best_d = d
                best = i
        return best


def score_recogniser[
    LAT: Int, D: Int, dtype: DType = DType.float64
](
    memory: PlaceMemory[LAT, D, dtype],
    queries: List[Scalar[dtype]],
    q_place: List[Int],
    q_parity: List[Int],
    threshold: Float64,
    content_only: Bool,
) -> RecogniserStats:
    """Match every query against memory and count outcomes by TRUE parity.

    Split by parity because that is where the predicted failure lives: a
    whole-latent similarity should do fine on parity-0 revisits and fail on
    parity-1 ones, which are exactly the revisits that close an informative
    cycle.
    """
    var n = len(q_place)
    var st = RecogniserStats(n, 0, 0, 0, 0, 0, 0, 0, 0)
    for t in range(n):
        var lat = List[Scalar[dtype]](length=LAT, fill=0)
        for i in range(LAT):
            lat[i] = queries[t * LAT + i]
        var m = memory.query(lat, threshold, content_only)
        var in_memory = False
        for i in range(memory.size()):
            if memory.truth_place[i] == q_place[t]:
                in_memory = True
                break
        if q_parity[t] == 0:
            st.n_parity_0 += 1
        else:
            st.n_parity_1 += 1
        if m == MATCH_NONE:
            if in_memory:
                st.missed += 1
            continue
        st.proposed += 1
        if memory.truth_place[m] == q_place[t]:
            st.correct += 1
            if q_parity[t] == 0:
                st.correct_parity_0 += 1
            else:
                st.correct_parity_1 += 1
        else:
            st.wrong += 1
    return st^
