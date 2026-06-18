"""TensorPack[N] — an owned bag of N `Tensor` storages with a subscript that
returns an origin-erased ref. Used by orchestrators for their inter-module
buffer pools (SeqS's `act`/`grd`) AND as the multi-arity input / grad_input
container for ≥2-ary leaves (no §B0 origin-union constraint — these are
independent storages, not origin-pinned views).
"""

from .tensor import Tensor


struct TensorPack[N: Int](Defaultable & Movable & ImplicitlyDeletable):
    var tensors: List[Tensor]

    def __init__(out self):
        self.tensors = List[Tensor]()
        comptime for i in range(Self.N):
            self.tensors.append(Tensor())

    def __getitem__(mut self, index: Int) raises -> ref [MutAnyOrigin] Tensor:
        # MUST be `MutAnyOrigin`, NOT `MutUntrackedOrigin`. The returned ref
        # points into `self.tensors`' heap buffer; the wildcard's lifetime-
        # PINNING is what keeps `self` (the pack) alive for as long as the ref
        # is live. `MutUntrackedOrigin` does NOT pin — the ref then dangles
        # the moment it's bound to a var or re-read (proven: a stored+reloaded
        # element reads garbage under MutUntracked, correct under MutAny). The
        # bypass-exclusivity property both share is necessary for passing two
        # elements as ref+mut call args; only the wildcard ALSO supplies the
        # load-bearing pinning. This is the rare case where the wildcard's
        # lifetime extension is the safety mechanism, not a footgun (cf. a
        # FIELD-stored erased pointer, which no origin pins — design review
        # §7.9). Do not "narrow" this to MutUntracked.
        return self.tensors[index]
