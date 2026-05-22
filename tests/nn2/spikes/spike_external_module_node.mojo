"""Spike Phase 3 — ExternalUnaryNode dispatch through UnsafePointer[M].

Goal: confirm that a graph-node-like struct can carry an
`UnsafePointer[M, MutAnyOrigin]` to an externally-owned Module instance
and invoke `module_ptr[].forward[target, BATCH](in_t, out_t)` from
inside its own `forward_via` method — with `M` being a trait-typed
comptime parameter (`M: Module`).

If this works, Phase 3 (β External-Module GraphNode) is unblocked:
the trainer keeps ownership of `actor`, `pair1.online`, `pair2.online`,
and the graph node references them per-call via a stored pointer set
by `set_external_via`.

Minimal validation:
  1. A tiny `IdentityM[DIM]` struct conforming to Module (forward is
     `out = in`, backward is `grad_in = grad_out`).
  2. A `Holder[M]` struct that stores `UnsafePointer[M, MutAnyOrigin]`
     and exposes `set_external_via` + `forward_via` + `vjp_via`.
  3. A driver that allocates an IdentityM externally, sets the pointer,
     and verifies forward/backward results match direct calls.
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Module, Initializer
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP


# ──────────────────────────────────────────────────────────────────────
# Tiny Module-conforming struct for the spike — out = in, grad_in = grad_out.
# ──────────────────────────────────────────────────────────────────────


struct IdentityM[DIM_: Int](Module):
    comptime IN_DIM = Self.DIM_
    comptime OUT_DIM = Self.DIM_

    # Sentinel field so we can verify the pointer routing is hitting THIS
    # instance and not a default-constructed sibling. forward writes to
    # output; backward writes to grad_input; both add `tag` so we can
    # distinguish.
    var tag: Scalar[DT]

    def __init__(out self):
        self.tag = Scalar[DT](0.0)

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        return Self()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        return Self()

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert input.flat_rank == 2, "input rank-2"
        comptime assert output.flat_rank == 2, "output rank-2"
        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
        for i in range(BATCH * Self.DIM_):
            out_p[i] = in_p[i] + self.tag

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_input: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
        for i in range(BATCH * Self.DIM_):
            gi_p[i] = go_p[i] + self.tag


# ──────────────────────────────────────────────────────────────────────
# Holder — minimal stand-in for ExternalUnaryNode. Stores a pointer to
# an external module, dispatches forward/backward through it.
# ──────────────────────────────────────────────────────────────────────


struct Holder[M: Module]:
    var _module_ptr: UnsafePointer[Self.M, MutAnyOrigin]

    def __init__(out self):
        self._module_ptr = UnsafePointer[Self.M, MutAnyOrigin](
            unsafe_from_address=0,
        )

    def set_external_via(
        mut self, ptr: UnsafePointer[Self.M, MutAnyOrigin],
    ):
        self._module_ptr = ptr

    def forward_via[
        target: StaticString,
        BATCH: Int,
    ](
        mut self,
        in_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var in_t = TileTensor(in_ptr, row_major[BATCH, Self.M.IN_DIM]())
        var out_t = TileTensor(out_ptr, row_major[BATCH, Self.M.OUT_DIM]())
        # The key question: can we dereference UnsafePointer[M, MutAnyOrigin]
        # to a mut Module instance and call forward on it?
        self._module_ptr[].forward[target, BATCH](in_t, out_t)

    def vjp_via[
        target: StaticString,
        BATCH: Int,
        mode: StaticString = "all",
    ](
        mut self,
        go_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        gi_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var go_t = TileTensor(go_ptr, row_major[BATCH, Self.M.OUT_DIM]())
        var gi_t = TileTensor(gi_ptr, row_major[BATCH, Self.M.IN_DIM]())
        self._module_ptr[].vjp[
            target, BATCH, mode=mode,
        ](go_t, gi_t)


def test_holder_forward() raises:
    print("test_holder_forward ...")
    comptime BATCH = 2
    comptime DIM = 3

    # External-owned module instance, sentinel tag set to 7.
    var ext = IdentityM[DIM]()
    ext.tag = Scalar[DT](7.0)

    var h = Holder[IdentityM[DIM]]()
    h.set_external_via(
        UnsafePointer[IdentityM[DIM], MutAnyOrigin](to=ext)
    )

    var x = alloc[Scalar[DT]](BATCH * DIM)
    var y = alloc[Scalar[DT]](BATCH * DIM)
    for i in range(BATCH * DIM):
        x[i] = Scalar[DT](Float64(i))

    h.forward_via["cpu", BATCH](
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x),
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](y),
    )

    # y[i] = x[i] + 7
    for i in range(BATCH * DIM):
        assert_true(
            y[i] == Scalar[DT](Float64(i) + 7.0),
            String("forward_via tag-add failed at i=") + String(i),
        )
    print("  ok")


def test_holder_backward() raises:
    print("test_holder_backward ...")
    comptime BATCH = 2
    comptime DIM = 3

    var ext = IdentityM[DIM]()
    ext.tag = Scalar[DT](-2.0)

    var h = Holder[IdentityM[DIM]]()
    h.set_external_via(
        UnsafePointer[IdentityM[DIM], MutAnyOrigin](to=ext)
    )

    var go = alloc[Scalar[DT]](BATCH * DIM)
    var gi = alloc[Scalar[DT]](BATCH * DIM)
    for i in range(BATCH * DIM):
        go[i] = Scalar[DT](Float64(i) * 0.5)

    h.vjp_via["cpu", BATCH](
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go),
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi),
    )

    # gi[i] = go[i] + (-2.0)
    for i in range(BATCH * DIM):
        var expect = Scalar[DT](Float64(i) * 0.5 - 2.0)
        assert_true(
            gi[i] == expect,
            String("vjp_via tag-add failed at i=") + String(i),
        )
    print("  ok")


def test_pointer_isolation() raises:
    """Two Holders pointing to two different IdentityM instances must
    dispatch to the correct one. Confirms the pointer carries identity."""
    print("test_pointer_isolation ...")
    comptime BATCH = 1
    comptime DIM = 2

    var a = IdentityM[DIM]()
    a.tag = Scalar[DT](100.0)
    var b = IdentityM[DIM]()
    b.tag = Scalar[DT](-100.0)

    var ha = Holder[IdentityM[DIM]]()
    var hb = Holder[IdentityM[DIM]]()
    ha.set_external_via(
        UnsafePointer[IdentityM[DIM], MutAnyOrigin](to=a)
    )
    hb.set_external_via(
        UnsafePointer[IdentityM[DIM], MutAnyOrigin](to=b)
    )

    var x = alloc[Scalar[DT]](BATCH * DIM)
    var ya = alloc[Scalar[DT]](BATCH * DIM)
    var yb = alloc[Scalar[DT]](BATCH * DIM)
    x[0] = Scalar[DT](1.0); x[1] = Scalar[DT](2.0)

    ha.forward_via["cpu", BATCH](
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x),
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](ya),
    )
    hb.forward_via["cpu", BATCH](
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x),
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](yb),
    )

    assert_true(ya[0] == Scalar[DT](101.0), "Holder ha did not route to a")
    assert_true(ya[1] == Scalar[DT](102.0), "Holder ha did not route to a")
    assert_true(yb[0] == Scalar[DT](-99.0), "Holder hb did not route to b")
    assert_true(yb[1] == Scalar[DT](-98.0), "Holder hb did not route to b")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Spike Phase 3 — ExternalUnaryNode dispatch via UnsafePointer[M]")
    print("=" * 70)
    test_holder_forward()
    test_holder_backward()
    test_pointer_isolation()
    print("=" * 70)
    print("SPIKE PASSED — Phase 3 β path is viable")
    print("=" * 70)
