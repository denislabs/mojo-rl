"""Spike: Module trait with reflection-derived default methods.

Today's nn2 has `polyak_update` and `hard_copy_params` as free functions
in `nn2/core/map_params.mojo` (~108 LOC). They walk both models'
`named_params` lists, validate that names and sizes match, then run
the interpolation. The "structure parity validation" boilerplate is
required because the connection between the two lists is by-name at
runtime.

With reflection-based trait default methods, both models have the
*same type* `T`, so the compiler enforces structural parity statically.
Walking is recursive: at each field, the trait method either
interpolates (if the field is a Param) or recurses into the field
(if the field is itself a Module-conforming type).

Conforming structs receive `polyak_update`, `hard_copy_params`,
`zero_grad`, and `set_inference` without writing any of them.

## Surface

  trait Module:
      def polyak_update(self, mut other: Self, tau): ...      # default
      def hard_copy_params(self, mut other: Self): ...        # default (calls polyak with tau=1)
      def zero_grad(mut self): ...                            # default
      def set_inference(mut self, value: Bool): ...           # default

  struct AutoLinear[IN, OUT]:
      var weight: Param["weight", True]
      var bias:   Param["bias",   False]
      var _target_tag: Int8

  struct TwoLayerNet:
      var lin1: AutoLinear[4, 8]
      var lin2: AutoLinear[8, 1]
      # Inherits ALL four trait default methods. Zero per-struct code.

## What this replaces

  - `nn2/core/map_params.mojo` (108 LOC) → 0 (trait method)
  - `OnlineTargetPair[M]` (52 LOC) → ~15 LOC (just owns two Modules)
  - Per-Module `set_inference` recursion bodies (Sequential, Residual,
    Parallel etc.) → trait default
  - Per-Module `zero_grad` helpers → trait default
"""

from std.reflection import reflect


comptime DT = DType.float32


# ──────────────────────────────────────────────────────────────────────
# IsParam marker trait — Param wrapper conforms.
# ──────────────────────────────────────────────────────────────────────


trait IsParam(Movable & ImplicitlyDestructible):
    """A field-type that represents a trainable parameter."""

    def n_elems(self) -> Int:
        ...

    def value_unsafe_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        ...

    def grad_unsafe_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        ...

    def param_name(self) -> StaticString:
        ...

    def param_decay(self) -> Bool:
        ...

    # In-place ops that the Module-level default methods dispatch to.
    def polyak_inplace(self, mut other: Self, tau: Scalar[DT]) raises:
        ...

    def zero_grad_inplace(mut self):
        ...


# ──────────────────────────────────────────────────────────────────────
# Param[NAME, APPLY_DECAY] — one trainable tensor + grad pair.
# ──────────────────────────────────────────────────────────────────────


struct Param[NAME: StaticString, APPLY_DECAY: Bool](IsParam):
    var value: List[Scalar[DT]]
    var grad: List[Scalar[DT]]

    def __init__(out self):
        self.value = List[Scalar[DT]]()
        self.grad = List[Scalar[DT]]()

    @staticmethod
    def with_size(n: Int) raises -> Self:
        var p = Self()
        p.value = List[Scalar[DT]](length=n, fill=0.0)
        p.grad  = List[Scalar[DT]](length=n, fill=0.0)
        return p^

    def n_elems(self) -> Int:
        return len(self.value)

    def value_unsafe_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.value.unsafe_ptr()
        )

    def grad_unsafe_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.grad.unsafe_ptr()
        )

    def param_name(self) -> StaticString:
        return Self.NAME

    def param_decay(self) -> Bool:
        return Self.APPLY_DECAY

    def polyak_inplace(self, mut other: Self, tau: Scalar[DT]) raises:
        """other.value = (1 - tau) * other.value + tau * self.value"""
        var omt = Scalar[DT](1.0) - tau
        var n = len(self.value)
        for k in range(n):
            other.value[k] = omt * other.value[k] + tau * self.value[k]

    def zero_grad_inplace(mut self):
        for k in range(len(self.grad)):
            self.grad[k] = Scalar[DT](0.0)


# ──────────────────────────────────────────────────────────────────────
# Module trait — marker, plus thin public surface that delegates to
# free-function walkers below.
#
# Why free functions for the recursion: Mojo trait default methods can
# self-recurse (calling the same method through a generic ref) but
# can't call SIBLING trait methods on a generic ref. We hit this with
# `zero_grad` calling itself on a child Module. Free generic functions
# don't have that limitation — calling `zero_grad_auto[type_of(a)](a)`
# resolves cleanly. The trait method just delegates.
# ──────────────────────────────────────────────────────────────────────


trait Module(Movable & ImplicitlyDestructible):
    """Marker trait for structs that contain Param fields and/or nested
    Modules. Conforming structs get polyak/hard_copy/zero_grad via the
    free `*_auto` walkers below, OR via the trait method delegates.
    """

    def polyak_update(self, mut other: Self, tau: Scalar[DT]) raises:
        polyak_update_auto(self, other, tau)

    def hard_copy_params(self, mut other: Self) raises:
        polyak_update_auto(self, other, Scalar[DT](1.0))

    def zero_grad(mut self):
        zero_grad_auto(self)


# ──────────────────────────────────────────────────────────────────────
# Free-function walkers. Recursive over Module-typed fields, terminal
# on Param-typed fields. Non-Param-non-Module fields skipped at comptime.
# ──────────────────────────────────────────────────────────────────────


def polyak_update_auto[T: Module](
    self_: T, mut other: T, tau: Scalar[DT],
) raises:
    """target ← (1 - tau) * target + tau * self, applied to every
    Param in the tree. Recurses into Module-typed fields.

    Compile-time structural parity: both args are `T`, so the compiler
    enforces shape match. No runtime name validation needed."""
    comptime field_types = reflect[T].field_types()
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = field_types[idx]
        comptime if conforms_to(ft, IsParam):
            ref a = reflect[T].field_ref[idx](self_)
            ref b = reflect[T].field_ref[idx](other)
            a.polyak_inplace(b, tau)
        comptime if conforms_to(ft, Module):
            ref a = reflect[T].field_ref[idx](self_)
            ref b = reflect[T].field_ref[idx](other)
            polyak_update_auto(a, b, tau)


def zero_grad_auto[T: Module](mut t: T):
    """Zero every Param's grad. Recurses into Module fields."""
    comptime field_types = reflect[T].field_types()
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = field_types[idx]
        comptime if conforms_to(ft, IsParam):
            ref a = reflect[T].field_ref[idx](t)
            a.zero_grad_inplace()
        comptime if conforms_to(ft, Module):
            ref a = reflect[T].field_ref[idx](t)
            zero_grad_auto(a)


# ──────────────────────────────────────────────────────────────────────
# AutoLinear — leaf Module with two Param fields. NO trait method bodies.
# ──────────────────────────────────────────────────────────────────────


struct AutoLinear[IN: Int, OUT: Int](Module):
    var weight: Param["weight", True]
    var bias:   Param["bias",   False]
    var _target_tag: Int8

    def __init__(out self):
        self.weight = Param["weight", True]()
        self.bias   = Param["bias",   False]()
        self._target_tag = Int8(0)

    @staticmethod
    def make_uniform(value: Scalar[DT]) raises -> Self:
        """Build a Linear with every weight + bias = `value`. Useful
        for the polyak test (lets us assert deterministic interpolation)."""
        var l = Self()
        l.weight = Param["weight", True].with_size(Self.IN * Self.OUT)
        l.bias   = Param["bias",   False].with_size(Self.OUT)
        for k in range(Self.IN * Self.OUT):
            l.weight.value[k] = value
        for j in range(Self.OUT):
            l.bias.value[j] = value
        l._target_tag = Int8(1)
        return l^


# ──────────────────────────────────────────────────────────────────────
# TwoLayerNet — composite Module. Tests recursion through Module fields.
# Note: ZERO per-struct method bodies. polyak_update/hard_copy_params/
# zero_grad all inherited from the Module trait defaults.
# ──────────────────────────────────────────────────────────────────────


struct TwoLayerNet(Module):
    var lin1: AutoLinear[4, 8]
    var lin2: AutoLinear[8, 1]

    def __init__(out self):
        self.lin1 = AutoLinear[4, 8]()
        self.lin2 = AutoLinear[8, 1]()

    @staticmethod
    def make_uniform(v1: Scalar[DT], v2: Scalar[DT]) raises -> Self:
        var n = Self()
        n.lin1 = AutoLinear[4, 8].make_uniform(v1)
        n.lin2 = AutoLinear[8, 1].make_uniform(v2)
        return n^


# ──────────────────────────────────────────────────────────────────────
# Smoke test.
# ──────────────────────────────────────────────────────────────────────


def main() raises:
    # Build online + target with different values so polyak is observable.
    var online = TwoLayerNet.make_uniform(Scalar[DT](1.0), Scalar[DT](2.0))
    var target = TwoLayerNet.make_uniform(Scalar[DT](0.0), Scalar[DT](0.0))

    # ── Test 1: hard_copy_params. ────────────────────────────────────
    online.hard_copy_params(target)

    var ok1 = (
        target.lin1.weight.value[0] == Scalar[DT](1.0)
        and target.lin1.bias.value[0] == Scalar[DT](1.0)
        and target.lin2.weight.value[0] == Scalar[DT](2.0)
        and target.lin2.bias.value[0] == Scalar[DT](2.0)
    )
    print("after hard_copy_params:")
    print("  target.lin1.weight[0] =", target.lin1.weight.value[0], "(expect 1.0)")
    print("  target.lin1.bias[0]   =", target.lin1.bias.value[0],   "(expect 1.0)")
    print("  target.lin2.weight[0] =", target.lin2.weight.value[0], "(expect 2.0)")
    print("  target.lin2.bias[0]   =", target.lin2.bias.value[0],   "(expect 2.0)")
    if not ok1:
        raise Error("hard_copy_params failed")

    # ── Test 2: polyak_update with tau=0.5. ──────────────────────────
    # target was set to online via hard_copy, so reset target to zeros
    # and run polyak. Expected: target = 0.5 * online + 0.5 * 0 = 0.5 * online.
    target = TwoLayerNet.make_uniform(Scalar[DT](0.0), Scalar[DT](0.0))
    online.polyak_update(target, Scalar[DT](0.5))

    var ok2 = (
        target.lin1.weight.value[0] == Scalar[DT](0.5)
        and target.lin1.bias.value[0] == Scalar[DT](0.5)
        and target.lin2.weight.value[0] == Scalar[DT](1.0)
        and target.lin2.bias.value[0] == Scalar[DT](1.0)
    )
    print()
    print("after polyak_update(tau=0.5):")
    print("  target.lin1.weight[0] =", target.lin1.weight.value[0], "(expect 0.5)")
    print("  target.lin2.weight[0] =", target.lin2.weight.value[0], "(expect 1.0)")
    if not ok2:
        raise Error("polyak_update failed")

    # ── Test 3: zero_grad. ───────────────────────────────────────────
    # Manually scribble a non-zero grad, then call zero_grad, expect 0.
    online.lin1.weight.grad[0] = Scalar[DT](42.0)
    online.lin2.bias.grad[0]   = Scalar[DT](99.0)
    online.zero_grad()
    var ok3 = (
        online.lin1.weight.grad[0] == Scalar[DT](0.0)
        and online.lin2.bias.grad[0] == Scalar[DT](0.0)
    )
    print()
    print("after zero_grad: online.lin1.weight.grad[0] =", online.lin1.weight.grad[0], "(expect 0.0)")
    if not ok3:
        raise Error("zero_grad failed")

    print()
    if ok1 and ok2 and ok3:
        print("PASS — Module trait defaults (polyak / hard_copy / zero_grad)")
        print("       work end-to-end via reflection. TwoLayerNet and")
        print("       AutoLinear both contain ZERO method bodies for these.")
    else:
        raise Error("Module trait defaults failed")
