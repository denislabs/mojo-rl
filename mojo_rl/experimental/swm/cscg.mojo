"""Clone-Structured Cognitive Graph (George et al. 2021) — the BASELINE for
the P5 comparison. Gates only; no execution path uses it.

A CSCG is an HMM whose hidden states are `n_clones` copies of every
observation symbol, with a deterministic emission (a clone emits its symbol)
and an action-conditioned transition `T[a][z, z']`. Learning is EM: the
E-step is forward-backward constrained to the clones of the observed
symbols, the M-step re-estimates `T` from the expected transition counts
(with a pseudocount). Perceptual aliasing is resolved when the two contexts
of one symbol settle on different clones.

Written independently of the reference implementation and kept small:
dense transition matrices (states are a few dozen here), scaled
forward-backward, Viterbi for the decode that the map score reads.
"""

from std.math import log

from .rng import Rng


struct Cscg(Copyable, Movable):
    var n_symbols: Int
    var n_clones: Int
    var n_actions: Int
    var n_states: Int
    var t: List[Float64]
    """`n_actions x n_states x n_states`, row-stochastic per (action, state)."""
    var pi: List[Float64]
    var pseudocount: Float64

    def __init__(
        out self,
        n_symbols: Int,
        n_clones: Int,
        n_actions: Int,
        seed: UInt64,
        pseudocount: Float64 = 1e-3,
    ):
        self.n_symbols = n_symbols
        self.n_clones = n_clones
        self.n_actions = n_actions
        self.n_states = n_symbols * n_clones
        self.pseudocount = pseudocount
        var rng = Rng(seed)
        var ns = self.n_states
        self.t = List[Float64](length=n_actions * ns * ns, fill=0)
        for a in range(n_actions):
            for z in range(ns):
                var row = Float64(0)
                for zp in range(ns):
                    var v = rng.uniform() + 0.1
                    self.t[(a * ns + z) * ns + zp] = v
                    row += v
                for zp in range(ns):
                    self.t[(a * ns + z) * ns + zp] /= row
        self.pi = List[Float64](length=ns, fill=1.0 / Float64(ns))

    def __init__(out self, *, copy: Self):
        self.n_symbols = copy.n_symbols
        self.n_clones = copy.n_clones
        self.n_actions = copy.n_actions
        self.n_states = copy.n_states
        self.t = copy.t.copy()
        self.pi = copy.pi.copy()
        self.pseudocount = copy.pseudocount

    def __init__(out self, *, deinit move: Self):
        self.n_symbols = move.n_symbols
        self.n_clones = move.n_clones
        self.n_actions = move.n_actions
        self.n_states = move.n_states
        self.t = move.t^
        self.pi = move.pi^
        self.pseudocount = move.pseudocount

    def state_of(self, symbol: Int, clone: Int) -> Int:
        return symbol * self.n_clones + clone

    def em(
        mut self,
        symbols: List[Int],
        actions: List[Int],
        iterations: Int,
    ) -> Float64:
        """Baum-Welch over the episodes of one walk (`actions[t] < 0` ends an
        episode). Returns the final mean log-likelihood per visit."""
        var ns = self.n_states
        var nc = self.n_clones
        var n = len(symbols)
        var ll = Float64(0)
        for _ in range(iterations):
            var counts = List[Float64](
                length=self.n_actions * ns * ns, fill=self.pseudocount
            )
            var pi_counts = List[Float64](length=ns, fill=self.pseudocount)
            ll = 0.0
            var start = 0
            while start < n:
                var end = start
                while end < n and actions[end] >= 0:
                    end += 1
                # episode = visits start..end inclusive
                var length = end - start + 1
                # forward over clones of the observed symbol only
                var alpha = List[Float64](length=length * nc, fill=0)
                var scale = List[Float64](length=length, fill=0)
                var s0 = symbols[start]
                var tot = Float64(0)
                for c in range(nc):
                    alpha[c] = self.pi[self.state_of(s0, c)]
                    tot += alpha[c]
                scale[0] = tot
                for c in range(nc):
                    alpha[c] /= tot
                for i in range(1, length):
                    var t_idx = start + i
                    var a = actions[t_idx - 1]
                    var sp = symbols[t_idx - 1]
                    var sc = symbols[t_idx]
                    tot = 0.0
                    for c in range(nc):
                        var acc = Float64(0)
                        for cp in range(nc):
                            acc += alpha[(i - 1) * nc + cp] * self.t[
                                (a * ns + self.state_of(sp, cp)) * ns
                                + self.state_of(sc, c)
                            ]
                        alpha[i * nc + c] = acc
                        tot += acc
                    if tot <= 0:
                        tot = 1e-300
                    scale[i] = tot
                    for c in range(nc):
                        alpha[i * nc + c] /= tot
                for i in range(length):
                    ll += log(scale[i])
                # backward
                var beta = List[Float64](length=length * nc, fill=1.0)
                for i in range(length - 2, -1, -1):
                    var t_idx = start + i
                    var a = actions[t_idx]
                    var sc = symbols[t_idx]
                    var sn = symbols[t_idx + 1]
                    for c in range(nc):
                        var acc = Float64(0)
                        for cn in range(nc):
                            acc += self.t[
                                (a * ns + self.state_of(sc, c)) * ns
                                + self.state_of(sn, cn)
                            ] * beta[(i + 1) * nc + cn]
                        beta[i * nc + c] = acc / scale[i + 1]
                # expected counts
                for c in range(nc):
                    pi_counts[self.state_of(s0, c)] += alpha[c] * beta[c]
                for i in range(length - 1):
                    var t_idx = start + i
                    var a = actions[t_idx]
                    var sc = symbols[t_idx]
                    var sn = symbols[t_idx + 1]
                    for c in range(nc):
                        var zc = self.state_of(sc, c)
                        for cn in range(nc):
                            var zn = self.state_of(sn, cn)
                            var xi = (
                                alpha[i * nc + c]
                                * self.t[(a * ns + zc) * ns + zn]
                                * beta[(i + 1) * nc + cn]
                                / scale[i + 1]
                            )
                            counts[(a * ns + zc) * ns + zn] += xi
                start = end + 1
            # M-step
            for a in range(self.n_actions):
                for z in range(ns):
                    var row = Float64(0)
                    for zp in range(ns):
                        row += counts[(a * ns + z) * ns + zp]
                    for zp in range(ns):
                        self.t[(a * ns + z) * ns + zp] = counts[
                            (a * ns + z) * ns + zp
                        ] / row
            var ptot = Float64(0)
            for z in range(ns):
                ptot += pi_counts[z]
            for z in range(ns):
                self.pi[z] = pi_counts[z] / ptot
        return ll / Float64(n)

    def decode(self, symbols: List[Int], actions: List[Int]) -> List[Int]:
        """Viterbi clone assignment per visit, as a global state id."""
        var ns = self.n_states
        var nc = self.n_clones
        var n = len(symbols)
        var out = List[Int](length=n, fill=0)
        var start = 0
        while start < n:
            var end = start
            while end < n and actions[end] >= 0:
                end += 1
            var length = end - start + 1
            var score = List[Float64](length=length * nc, fill=-1e300)
            var back = List[Int](length=length * nc, fill=0)
            var s0 = symbols[start]
            for c in range(nc):
                score[c] = log(self.pi[self.state_of(s0, c)] + 1e-300)
            for i in range(1, length):
                var t_idx = start + i
                var a = actions[t_idx - 1]
                var sp = symbols[t_idx - 1]
                var sc = symbols[t_idx]
                for c in range(nc):
                    var best = -1e300
                    var arg = 0
                    for cp in range(nc):
                        var v = score[(i - 1) * nc + cp] + log(
                            self.t[
                                (a * ns + self.state_of(sp, cp)) * ns
                                + self.state_of(sc, c)
                            ]
                            + 1e-300
                        )
                        if v > best:
                            best = v
                            arg = cp
                    score[i * nc + c] = best
                    back[i * nc + c] = arg
            var best_c = 0
            var best_v = -1e300
            for c in range(nc):
                if score[(length - 1) * nc + c] > best_v:
                    best_v = score[(length - 1) * nc + c]
                    best_c = c
            var c = best_c
            for i in range(length - 1, -1, -1):
                out[start + i] = self.state_of(symbols[start + i], c)
                if i > 0:
                    c = back[i * nc + c]
            start = end + 1
        return out^


def dense_labels(states: List[Int]) -> List[Int]:
    """Renumber decoded state ids densely, so `score_map` can read them."""
    var seen = List[Int]()
    var out = List[Int](length=len(states), fill=0)
    for t in range(len(states)):
        var idx = -1
        for k in range(len(seen)):
            if seen[k] == states[t]:
                idx = k
                break
        if idx < 0:
            seen.append(states[t])
            idx = len(seen) - 1
        out[t] = idx
    return out^
