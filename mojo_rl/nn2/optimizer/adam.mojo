"""Adam optimizer (Kingma & Ba 2014) — CPU Phase 1.

Storage shape:
  m_flat, v_flat : flat List[Scalar[DT]] holding all params' moments
                   concatenated in walk-order.
  offsets        : List[Int], offsets[k] = start of param k's slice
                   inside m_flat / v_flat.

`_AdamInitVisitor` walks the model once at construction time, appending
to all three Lists per parameter. `_AdamStepVisitor` walks at each step,
indexing into the flat moments by `offsets[idx] + i`.

Visitors borrow pointers to Adam's Lists rather than owning them — this
avoids the "move field out of struct mid-life" failure Mojo's flow
analysis rejects when an owned-visitor would need to hand its lists back
to Adam after the walk.

Bias corrections cached incrementally — `beta_pow_t` updates once per
step instead of recomputing `pow(beta, t)` each call.
"""

from std.math import sqrt
from layout import TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import Module, ParamVisitor


# ──────────────────────────────────────────────────────────────────────────
# Init visitor — appends to Adam's flat Lists per param.
# ──────────────────────────────────────────────────────────────────────────

@fieldwise_init
struct _AdamInitVisitor(ParamVisitor):
    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]

    def visit[L: TensorLayout](
        mut self,
        name: String,
        param: TileTensor[DT, L, MutAnyOrigin],
        grad: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
    ):
        # Record this param's start offset, then append n_elems zeros
        # to each flat list.
        var zero: Scalar[DT] = 0.0
        self.offsets_ptr[].append(len(self.m_flat_ptr[]))
        for _ in range(n_elems):
            self.m_flat_ptr[].append(zero)
            self.v_flat_ptr[].append(zero)


# ──────────────────────────────────────────────────────────────────────────
# Step visitor — applies Adam update per-param using offsets[idx] to
# index into the flat moment lists.
# ──────────────────────────────────────────────────────────────────────────

@fieldwise_init
struct _AdamStepVisitor(ParamVisitor):
    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var idx: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var bias_correction1: Scalar[DT]
    var bias_correction2: Scalar[DT]

    def visit[L: TensorLayout](
        mut self,
        name: String,
        param: TileTensor[DT, L, MutAnyOrigin],
        grad: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
    ):
        var off = self.offsets_ptr[][self.idx]
        var p_ptr = param.ptr
        var g_ptr = grad.ptr
        var one: Scalar[DT] = 1.0

        for i in range(n_elems):
            var g = g_ptr[i]
            var m_old = self.m_flat_ptr[][off + i]
            var v_old = self.v_flat_ptr[][off + i]
            var m_new = self.beta1 * m_old + (one - self.beta1) * g
            var v_new = self.beta2 * v_old + (one - self.beta2) * g * g
            self.m_flat_ptr[][off + i] = m_new
            self.v_flat_ptr[][off + i] = v_new
            var m_hat = m_new / self.bias_correction1
            var v_hat = v_new / self.bias_correction2
            p_ptr[i] = p_ptr[i] - self.lr * m_hat / (sqrt(v_hat) + self.eps)

        self.idx += 1


# ──────────────────────────────────────────────────────────────────────────
# Zero-grad visitor — walks param tree and zeros every gradient buffer.
# ──────────────────────────────────────────────────────────────────────────

struct _ZeroGradVisitor(ParamVisitor):
    def __init__(out self):
        pass

    def visit[L: TensorLayout](
        mut self,
        name: String,
        param: TileTensor[DT, L, MutAnyOrigin],
        grad: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
    ):
        var g_ptr = grad.ptr
        var zero: Scalar[DT] = 0.0
        for i in range(n_elems):
            g_ptr[i] = zero


# ──────────────────────────────────────────────────────────────────────────
# Adam — flat-list state + offsets table; visitors borrow via pointers.
# ──────────────────────────────────────────────────────────────────────────

@fieldwise_init
struct Adam(Movable & ImplicitlyDestructible):
    var m_flat: List[Scalar[DT]]
    var v_flat: List[Scalar[DT]]
    var offsets: List[Int]
    var step_count: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var beta1_pow_t: Scalar[DT]
    var beta2_pow_t: Scalar[DT]

    @staticmethod
    def make[M: Module](
        mut model: M,
        lr: Scalar[DT] = 0.001,
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
    ) -> Self:
        var adam = Self(
            m_flat=List[Scalar[DT]](),
            v_flat=List[Scalar[DT]](),
            offsets=List[Int](),
            step_count=0,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            beta1_pow_t=1.0, beta2_pow_t=1.0,
        )
        var visitor = _AdamInitVisitor(
            m_flat_ptr=UnsafePointer(to=adam.m_flat),
            v_flat_ptr=UnsafePointer(to=adam.v_flat),
            offsets_ptr=UnsafePointer(to=adam.offsets),
        )
        model.for_each_param(String(""), visitor)
        return adam^

    def zero_grad[M: Module](mut self, mut model: M):
        """Zero every gradient accumulator in the model. Call before
        backward so that gradients from this minibatch don't add to the
        previous one's."""
        var v = _ZeroGradVisitor()
        model.for_each_param(String(""), v)

    def step[M: Module](mut self, mut model: M):
        self.step_count += 1
        self.beta1_pow_t = self.beta1_pow_t * self.beta1
        self.beta2_pow_t = self.beta2_pow_t * self.beta2
        var bc1: Scalar[DT] = 1.0 - self.beta1_pow_t
        var bc2: Scalar[DT] = 1.0 - self.beta2_pow_t

        var visitor = _AdamStepVisitor(
            m_flat_ptr=UnsafePointer(to=self.m_flat),
            v_flat_ptr=UnsafePointer(to=self.v_flat),
            offsets_ptr=UnsafePointer(to=self.offsets),
            idx=0,
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            bias_correction1=bc1,
            bias_correction2=bc2,
        )
        model.for_each_param(String(""), visitor)
