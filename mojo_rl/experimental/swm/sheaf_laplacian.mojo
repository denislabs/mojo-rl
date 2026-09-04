"""Dense sheaf Laplacian and its spectrum — FOR GATES ONLY.

Nothing on an execution path calls this file, and that is a design decision,
not an omission. On a cycle of length N the Mobius obstruction shows up
spectrally as `dim ker L = 1 < D`, with the informative gap

    lambda_2 - lambda_1 = 2 (1 - cos(pi / N))

which is 0.068 at N=12 but 2.7e-4 at N=192 — it falls off as 1/N^2 and drowns
in noise on long loops, while `||H - I||_F` stays at exactly 2 regardless of N.
So the runtime reads holonomies; the spectrum's job is to be an independent,
closed-form check that the sheaf structure was assembled correctly at all.

For an edge `e = (u -> v)` with restriction `R_e` on the source side and `I` on
the target side, `(delta x)_e = x_v - R_e x_u`, and `L = delta^T delta` has

    L[u,u] += R_e^T R_e      L[v,v] += I
    L[u,v] += -R_e^T         L[v,u] += -R_e
"""

from std.math import abs, sqrt

from .so_d import SqMat
from .place_graph import PlaceGraph, EDGE_ACTION, EDGE_IDENTIFICATION


struct DenseSym[dtype: DType = DType.float64](Copyable, Movable):
    """Runtime-sized dense symmetric matrix. Heap-backed: `n` is not comptime."""

    var n: Int
    var data: List[Scalar[Self.dtype]]

    def __init__(out self, n: Int):
        self.n = n
        self.data = List[Scalar[Self.dtype]](length=n * n, fill=0)

    def __init__(out self, *, copy: Self):
        self.n = copy.n
        self.data = copy.data.copy()

    def __init__(out self, *, deinit move: Self):
        self.n = move.n
        self.data = move.data^

    def __getitem__(self, r: Int, c: Int) -> Scalar[Self.dtype]:
        return self.data[r * self.n + c]

    def __setitem__(mut self, r: Int, c: Int, v: Scalar[Self.dtype]):
        self.data[r * self.n + c] = v

    def add(mut self, r: Int, c: Int, v: Scalar[Self.dtype]):
        self.data[r * self.n + c] += v

    def symmetry_error(self) -> Float64:
        var worst = Float64(0)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                var d = abs(
                    Float64(self.data[i * self.n + j])
                    - Float64(self.data[j * self.n + i])
                )
                if d > worst:
                    worst = d
        return worst


def build_sheaf_laplacian[
    D: Int, dtype: DType = DType.float64
](graph: PlaceGraph[D, dtype], include_identifications: Bool = True) -> DenseSym[
    dtype
]:
    """Assemble `L = delta^T delta` over the graph's edges.

    `include_identifications=False` reproduces the "cut the non-trivial
    identifications out of the energy" rule of v2 §4.3.
    """
    var n = graph.n_places * D
    var lap = DenseSym[dtype](n)
    for e in range(len(graph.edges)):
        var ed = graph.edges[e]
        if ed.kind == EDGE_IDENTIFICATION and not include_identifications:
            continue
        if ed.kind != EDGE_ACTION and ed.kind != EDGE_IDENTIFICATION:
            continue
        var r = graph.transports[e]
        var u = ed.src * D
        var v = ed.dst * D
        var w = Scalar[dtype](ed.w if ed.in_energy else 0.0)
        if w == 0:
            continue
        var rtr = r.transpose() * r
        for i in range(D):
            for j in range(D):
                lap.add(u + i, u + j, w * rtr[i, j])
                lap.add(u + i, v + j, -w * r[j, i])  # -R^T
                lap.add(v + i, u + j, -w * r[i, j])  # -R
            lap.add(v + i, v + i, w)
    return lap^


def eigenvalues_ascending[
    dtype: DType = DType.float64
](m: DenseSym[dtype], max_sweeps: Int = 60, tol: Float64 = 1e-14) raises -> List[
    Float64
]:
    """All eigenvalues of a symmetric matrix by the cyclic Jacobi method.

    Jacobi and not a Krylov method on purpose: it is unconditionally accurate
    for the SMALL eigenvalues, and the small ones are the entire point here
    (`lambda_1 = 0` with multiplicity 1 vs D is the obstruction).
    """
    var n = m.n
    var a = m.data.copy()
    for _ in range(max_sweeps):
        var off = Float64(0)
        for p in range(n):
            for q in range(p + 1, n):
                var v = Float64(a[p * n + q])
                off += v * v
        if off <= tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                var apq = Float64(a[p * n + q])
                if abs(apq) < 1e-300:
                    continue
                var app = Float64(a[p * n + p])
                var aqq = Float64(a[q * n + q])
                var theta = (aqq - app) / (2.0 * apq)
                var t: Float64
                if theta >= 0:
                    t = 1.0 / (theta + sqrt(theta * theta + 1.0))
                else:
                    t = -1.0 / (-theta + sqrt(theta * theta + 1.0))
                var c = 1.0 / sqrt(t * t + 1.0)
                var s = t * c
                for k in range(n):
                    var akp = Float64(a[k * n + p])
                    var akq = Float64(a[k * n + q])
                    a[k * n + p] = Scalar[dtype](c * akp - s * akq)
                    a[k * n + q] = Scalar[dtype](s * akp + c * akq)
                for k in range(n):
                    var apk = Float64(a[p * n + k])
                    var aqk = Float64(a[q * n + k])
                    a[p * n + k] = Scalar[dtype](c * apk - s * aqk)
                    a[q * n + k] = Scalar[dtype](s * apk + c * aqk)

    var out = List[Float64]()
    for i in range(n):
        out.append(Float64(a[i * n + i]))
    # Insertion sort: n is a few hundred at most in the gates.
    for i in range(1, len(out)):
        var v = out[i]
        var j = i - 1
        while j >= 0 and out[j] > v:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = v
    return out^


def kernel_dimension(eigs: List[Float64], tol: Float64 = 1e-8) -> Int:
    """`dim ker L` = number of global sections. `D` iff transport is path-independent."""
    var k = 0
    for i in range(len(eigs)):
        if abs(eigs[i]) <= tol:
            k += 1
    return k
