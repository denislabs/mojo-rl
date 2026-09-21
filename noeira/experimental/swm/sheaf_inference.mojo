"""Inference as descent: minimise the sheaf Dirichlet energy over the frames.

This is the PCN's `pc_inference_step` restricted to the frame channel and
generalised from a chain to a runtime graph. The energy is

    E(u) = sum_{e in energy}  w_e ||u_dst - R_e u_src||^2
         + beta * sum_{p anchored} ||u_p - anchor_p||^2
         + ridge * ||u||^2

and one Jacobi step is `u <- u - lr * dE/du`. The ridge is not decoration: `E`
is only semi-definite without it (unanchored places and the kernels of the
restrictions leave flat directions), so the exact solve would be singular.

**Only edges marked `in_energy` participate.** An identification whose cycle
carries a non-trivial holonomy is not a constraint, it is a monodromy; including
it forces `u` into the cycle's fixed subspace — a rank collapse that destroys
exactly the information the method exists to report. That claim is measured, not
assumed: see the G11 gate.

CPU and serial by design. The edge-parallel version in the design doc
accumulates into vertex gradients with `atomic_add`, a shape that has already
produced a silent miscompute in this repo, so the CPU numbers get frozen as a
reference before any of that is attempted.
"""

from std.math import abs, sqrt

from .so_d import SqMat
from .place_graph import PlaceGraph, EDGE_ACTION, EDGE_IDENTIFICATION
from .sheaf_laplacian import DenseSym, build_sheaf_laplacian, dense_solve


@fieldwise_init
struct InferenceConfig(Copyable, ImplicitlyCopyable, Movable):
    var steps: Int
    var lr: Float64
    var beta: Float64
    """Weight on the encoder's anchors. The PCN's clamped readout."""
    var ridge: Float64
    """`eps_0`: pins the flat directions. Without it `E` is semi-definite."""

    @staticmethod
    def default() -> Self:
        return Self(400, 0.05, 1.0, 1e-6)


def frame_energy[
    D: Int, dtype: DType = DType.float64
](
    graph: PlaceGraph[D, dtype],
    u: List[Float64],
    anchors: List[Float64],
    has_anchor: List[Bool],
    cfg: InferenceConfig,
) -> Float64:
    var e = Float64(0)
    for idx in range(len(graph.edges)):
        var ed = graph.edges[idx]
        if not ed.in_energy:
            continue
        if ed.kind != EDGE_ACTION and ed.kind != EDGE_IDENTIFICATION:
            continue
        var r = graph.transports[idx]
        for i in range(D):
            var pred = Float64(0)
            for j in range(D):
                pred += Float64(r[i, j]) * u[ed.src * D + j]
            var d = u[ed.dst * D + i] - pred
            e += ed.w * d * d
    for p in range(graph.n_places):
        for i in range(D):
            var v = u[p * D + i]
            if has_anchor[p]:
                var d = v - anchors[p * D + i]
                e += cfg.beta * d * d
            e += cfg.ridge * v * v
    return e


def frame_gradient[
    D: Int, dtype: DType = DType.float64
](
    graph: PlaceGraph[D, dtype],
    u: List[Float64],
    anchors: List[Float64],
    has_anchor: List[Bool],
    cfg: InferenceConfig,
    mut grad: List[Float64],
):
    """`dE/du`. The source-side term is the transport applied BACKWARDS.

    `R^T eps` is `R^-1 eps` because the transport is orthogonal, so the
    predictive-coding pull-back and the parallel transport back along the edge
    are literally the same operation — which is why it costs nothing here.
    """
    for i in range(len(grad)):
        grad[i] = 0
    for idx in range(len(graph.edges)):
        var ed = graph.edges[idx]
        if not ed.in_energy:
            continue
        if ed.kind != EDGE_ACTION and ed.kind != EDGE_IDENTIFICATION:
            continue
        var r = graph.transports[idx]
        var eps = List[Float64](length=D, fill=0)
        for i in range(D):
            var pred = Float64(0)
            for j in range(D):
                pred += Float64(r[i, j]) * u[ed.src * D + j]
            eps[i] = u[ed.dst * D + i] - pred
        for i in range(D):
            grad[ed.dst * D + i] += 2.0 * ed.w * eps[i]
            for j in range(D):
                grad[ed.src * D + j] -= 2.0 * ed.w * Float64(r[i, j]) * eps[i]
    for p in range(graph.n_places):
        for i in range(D):
            var v = u[p * D + i]
            if has_anchor[p]:
                grad[p * D + i] += 2.0 * cfg.beta * (v - anchors[p * D + i])
            grad[p * D + i] += 2.0 * cfg.ridge * v


def infer_frames[
    D: Int, dtype: DType = DType.float64
](
    graph: PlaceGraph[D, dtype],
    anchors: List[Float64],
    has_anchor: List[Bool],
    cfg: InferenceConfig,
    warm_start: List[Float64],
) -> List[Float64]:
    """`steps` Jacobi iterations from `warm_start` (the amortised posterior)."""
    var u = warm_start.copy()
    var grad = List[Float64](length=len(u), fill=0)
    for _ in range(cfg.steps):
        frame_gradient[D, dtype](graph, u, anchors, has_anchor, cfg, grad)
        for i in range(len(u)):
            u[i] -= cfg.lr * grad[i]
    return u^


def solve_frames_exact[
    D: Int, dtype: DType = DType.float64
](
    graph: PlaceGraph[D, dtype],
    anchors: List[Float64],
    has_anchor: List[Bool],
    cfg: InferenceConfig,
) raises -> List[Float64]:
    """The stationary point in closed form: `(L_w + beta M + ridge I) u = beta M a`.

    Exists so the iterative path has something exact to be checked against.
    """
    var lap = build_sheaf_laplacian[D, dtype](graph)
    var n = graph.n_places * D
    var a = DenseSym[dtype](n)
    for i in range(n * n):
        a.data[i] = lap.data[i]
    var b = List[Float64](length=n, fill=0)
    for p in range(graph.n_places):
        for i in range(D):
            var k = p * D + i
            a.data[k * n + k] += Scalar[dtype](cfg.ridge)
            if has_anchor[p]:
                a.data[k * n + k] += Scalar[dtype](cfg.beta)
                b[k] = cfg.beta * anchors[k]
    return dense_solve[dtype](a, b)


def frame_covariance_anisotropy(
    samples: List[Float64], n: Int
) -> Float64:
    """`sqrt(lambda_min/lambda_max)` of the 2x2 covariance of stacked frames.

    The rank-collapse detector for G11: an inference that has been forced into
    a cycle's one-dimensional fixed subspace shows this going to zero.
    """
    if n < 2:
        return 0.0
    var m0 = Float64(0)
    var m1 = Float64(0)
    for t in range(n):
        m0 += samples[t * 2]
        m1 += samples[t * 2 + 1]
    m0 /= Float64(n)
    m1 /= Float64(n)
    var c00 = Float64(0)
    var c01 = Float64(0)
    var c11 = Float64(0)
    for t in range(n):
        var a = samples[t * 2] - m0
        var b = samples[t * 2 + 1] - m1
        c00 += a * a
        c01 += a * b
        c11 += b * b
    var d = Float64(n - 1)
    c00 /= d
    c01 /= d
    c11 /= d
    var tr = c00 + c11
    var det = c00 * c11 - c01 * c01
    var disc = tr * tr / 4.0 - det
    if disc < 0:
        disc = 0
    var root = sqrt(disc)
    var hi = tr / 2.0 + root
    var lo = tr / 2.0 - root
    if hi <= 1e-300 or lo <= 0:
        return 0.0
    return sqrt(lo / hi)
