"""Slim Optimizer trait spike (Follow-up #3).

Demonstrates the trait shape after dropping algorithm-specific
hyperparams from `make` in the trait. Hyperparams live as public mut
fields on the concrete optimizer.

# Slim trait (this spike)

  trait Optimizer:
      @staticmethod
      def make[M: Module](mut model: M) raises -> Self: ...
      def zero_grad[M: Module](mut self, mut model: M) raises: ...
      def step[M: Module](mut self, mut model: M) raises: ...

# vs. current nn2 (`core/optimizer.mojo:16-22`)

  @staticmethod
  def make[target, M: Module](
      mut model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
  ) raises -> Self: ...

The current trait bakes Adam's β₁/β₂/ε into the trait surface; SGD and
RMSprop don't share that vocabulary. Slim trait stays neutral; users
poke `opt.lr = 3e-4` after `make`.

# What the spike intentionally drops

  - `target: StaticString` comptime param — CPU-only here; real trait
    keeps it.
  - GPU storage / DeviceContext plumbing.
  - `apply_decay` flag on ParamVisitor — orthogonal to this spike.

The retrofit (Follow-up #6) brings these back. This spike isolates
"does the slim trait shape work end-to-end."
"""

from std.math import sqrt
from .spike_unified_buffers import DT, Module, ParamVisitor


# ──────────────────────────────────────────────────────────────────────
# Slim Optimizer trait — no algorithm-specific hyperparams in `make`.
# ──────────────────────────────────────────────────────────────────────


trait Optimizer(Defaultable & Movable & ImplicitlyDestructible):
    @staticmethod
    def make[M: Module](mut model: M) raises -> Self:
        ...

    def zero_grad[M: Module](mut self, mut model: M) raises:
        ...

    def step[M: Module](mut self, mut model: M) raises:
        ...


# ──────────────────────────────────────────────────────────────────────
# Param-walk visitors. Same shape as nn2's; just simpler signature
# because the spike Module/ParamVisitor pair is minimal.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _InitVisitor(ParamVisitor):
    """Lays out flat m / v storage in walk order."""

    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]

    def visit(
        mut self,
        name: String,
        param_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        grad_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_elems: Int,
    ) raises:
        self.offsets_ptr[].append(len(self.m_flat_ptr[]))
        var zero: Scalar[DT] = 0.0
        for _ in range(n_elems):
            self.m_flat_ptr[].append(zero)
            self.v_flat_ptr[].append(zero)


@fieldwise_init
struct _StepVisitor(ParamVisitor):
    """Per-leaf Adam update. Reads hyperparams off the optimizer."""

    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var idx: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var bc1: Scalar[DT]
    var bc2: Scalar[DT]

    def visit(
        mut self,
        name: String,
        param_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        grad_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_elems: Int,
    ) raises:
        var off = self.offsets_ptr[][self.idx]
        var m = self.m_flat_ptr[].unsafe_ptr() + off
        var v = self.v_flat_ptr[].unsafe_ptr() + off
        var one: Scalar[DT] = 1.0
        for i in range(n_elems):
            var g = grad_ptr[i]
            var m_new = self.beta1 * m[i] + (one - self.beta1) * g
            var v_new = self.beta2 * v[i] + (one - self.beta2) * g * g
            m[i] = m_new
            v[i] = v_new
            var m_hat = m_new / self.bc1
            var v_hat = v_new / self.bc2
            param_ptr[i] = param_ptr[i] - self.lr * m_hat / (sqrt(v_hat) + self.eps)
        self.idx += 1


@fieldwise_init
struct _ZeroGradVisitor(ParamVisitor):
    def visit(
        mut self,
        name: String,
        param_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        grad_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_elems: Int,
    ) raises:
        var zero: Scalar[DT] = 0.0
        for i in range(n_elems):
            grad_ptr[i] = zero


# ──────────────────────────────────────────────────────────────────────
# Adam — implements the slim trait. Hyperparams are public mut fields.
# ──────────────────────────────────────────────────────────────────────


struct Adam(Optimizer):
    var m_flat: List[Scalar[DT]]
    var v_flat: List[Scalar[DT]]
    var offsets: List[Int]
    var step_count: Int

    # Public mut hyperparams. Defaults match Kingma & Ba 2014. User
    # overrides after `make`: `opt.lr = 3e-4`.
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]

    var beta1_pow_t: Scalar[DT]
    var beta2_pow_t: Scalar[DT]

    def __init__(out self):
        self.m_flat = List[Scalar[DT]]()
        self.v_flat = List[Scalar[DT]]()
        self.offsets = List[Int]()
        self.step_count = 0
        self.lr = Scalar[DT](0.001)
        self.beta1 = Scalar[DT](0.9)
        self.beta2 = Scalar[DT](0.999)
        self.eps = Scalar[DT](1e-8)
        self.beta1_pow_t = Scalar[DT](1.0)
        self.beta2_pow_t = Scalar[DT](1.0)

    @staticmethod
    def make[M: Module](mut model: M) raises -> Self:
        """Trait factory — no hyperparams. User sets `opt.lr` etc. after."""
        var opt = Self()
        var init = _InitVisitor(
            m_flat_ptr=UnsafePointer(to=opt.m_flat),
            v_flat_ptr=UnsafePointer(to=opt.v_flat),
            offsets_ptr=UnsafePointer(to=opt.offsets),
        )
        model.for_each_param(String(""), init)
        return opt^

    def zero_grad[M: Module](mut self, mut model: M) raises:
        var v = _ZeroGradVisitor()
        model.for_each_param(String(""), v)

    def step[M: Module](mut self, mut model: M) raises:
        self.step_count += 1
        self.beta1_pow_t *= self.beta1
        self.beta2_pow_t *= self.beta2
        var bc1: Scalar[DT] = Scalar[DT](1.0) - self.beta1_pow_t
        var bc2: Scalar[DT] = Scalar[DT](1.0) - self.beta2_pow_t
        var step = _StepVisitor(
            m_flat_ptr=UnsafePointer(to=self.m_flat),
            v_flat_ptr=UnsafePointer(to=self.v_flat),
            offsets_ptr=UnsafePointer(to=self.offsets),
            idx=0,
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            bc1=bc1,
            bc2=bc2,
        )
        model.for_each_param(String(""), step)
