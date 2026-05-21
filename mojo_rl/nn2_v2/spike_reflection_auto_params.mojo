"""Auto-derive `for_each_param` from struct field reflection.

Today's nn2 leaves write `for_each_param` by hand — Linear, GaussianHead,
LayerNorm each have a method that names every (weight, grad_w) pair and
calls visitor.visit. Reflection lets us derive it.

# Approach: wrap parameters in a `Param[NAME, APPLY_DECAY]` field type

  struct Linear[IN, OUT]:
      var weight: Param["weight", True]    # value + grad
      var bias:   Param["bias", False]
      var cache:  List[Scalar[DT]]         # NOT a Param
      var _target_tag: Int8                # NOT a Param

  # Reflection walks fields, picks the Param ones, calls visitor.

`Param` carries its `NAME` and `APPLY_DECAY` as comptime params, so each
field has a distinct type. We use `conforms_to(field_type, IsParam)` to
filter; only Param-typed fields get visited.

# Why this matters

Today's `Linear.for_each_param` is ~25 lines (CPU branch + GPU branch +
naming). Auto-derived: **zero lines per leaf**. The leaf just declares
fields with `Param[...]` types. Same for every other parameterized
leaf (LayerNorm with γ/β, GaussianHead with W/b/log_std, etc.).
"""

from std.reflection import reflect


comptime DT = DType.float32


# ──────────────────────────────────────────────────────────────────────
# IsParam — marker trait every Param[NAME, DECAY] conforms to. Used by
# `conforms_to` to filter reflected fields.
# ──────────────────────────────────────────────────────────────────────


trait IsParam(Movable & ImplicitlyDestructible):
    """Marker trait — a field-type that the param-walker should visit."""

    def param_name(self) -> StaticString:
        ...

    def param_decay(self) -> Bool:
        ...

    def value_unsafe_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        ...

    def grad_unsafe_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        ...

    def n_elems(self) -> Int:
        ...


# ──────────────────────────────────────────────────────────────────────
# Param[NAME, APPLY_DECAY] — one trainable tensor + its gradient.
# Each (NAME, APPLY_DECAY) combination is a distinct type.
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

    def param_name(self) -> StaticString:
        return Self.NAME

    def param_decay(self) -> Bool:
        return Self.APPLY_DECAY

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

    def n_elems(self) -> Int:
        return len(self.value)


# ──────────────────────────────────────────────────────────────────────
# ParamVisitor + auto-derived walker.
# ──────────────────────────────────────────────────────────────────────


trait ParamVisitor(ImplicitlyDestructible):
    def visit(
        mut self,
        name: String,
        param_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        grad_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        ...


def for_each_param_auto[T: AnyType, V: ParamVisitor](
    mut t: T, prefix: String, mut visitor: V,
) raises:
    """Walk every Param-typed field of T and dispatch the visitor.

    The leaf just declares `var weight: Param["weight", True]`; this
    function does the rest. No per-leaf method needed.
    """
    comptime field_types = reflect[T].field_types()
    comptime field_names = reflect[T].field_names()
    var sep = "." if prefix.byte_length() > 0 else ""
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = field_types[idx]
        comptime if conforms_to(ft, IsParam):
            ref p = reflect[T].field_ref[idx](t)
            visitor.visit(
                prefix + sep + String(p.param_name()),
                p.value_unsafe_ptr(),
                p.grad_unsafe_ptr(),
                p.n_elems(),
                p.param_decay(),
            )


# ──────────────────────────────────────────────────────────────────────
# Auto-Linear — leaf with ZERO hand-written for_each_param body.
# ──────────────────────────────────────────────────────────────────────


struct AutoLinear[IN: Int, OUT: Int](Movable & ImplicitlyDestructible):
    """Like spike_unified_buffers.Linear but parameters live in Param[]
    wrappers so the param-walker auto-derives."""

    var weight: Param["weight", True]
    var bias:   Param["bias",   False]
    var cache_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]   # NOT a Param
    var _target_tag: Int8                                    # NOT a Param

    def __init__(out self):
        self.weight = Param["weight", True]()
        self.bias   = Param["bias",   False]()
        self.cache_ptr = UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)
        self._target_tag = Int8(0)

    @staticmethod
    def make() raises -> Self:
        var l = Self()
        l.weight = Param["weight", True].with_size(Self.IN * Self.OUT)
        l.bias   = Param["bias",   False].with_size(Self.OUT)
        l._target_tag = Int8(1)
        return l^

    # NOTE: NO `def for_each_param` method here. The user calls
    # `for_each_param_auto(linear, "", visitor)` and reflection
    # discovers `weight` and `bias` automatically.


# ──────────────────────────────────────────────────────────────────────
# Smoke test visitor — just records names + sizes + decay flags.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _RecordVisitor(ParamVisitor):
    var names_ptr: UnsafePointer[List[String], MutAnyOrigin]
    var sizes_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var decays_ptr: UnsafePointer[List[Bool], MutAnyOrigin]

    def visit(
        mut self,
        name: String,
        param_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        grad_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        self.names_ptr[].append(name)
        self.sizes_ptr[].append(n_elems)
        self.decays_ptr[].append(apply_decay)


def main() raises:
    var lin = AutoLinear[4, 3].make()

    var names  = List[String]()
    var sizes  = List[Int]()
    var decays = List[Bool]()
    var v = _RecordVisitor(
        names_ptr=UnsafePointer(to=names),
        sizes_ptr=UnsafePointer(to=sizes),
        decays_ptr=UnsafePointer(to=decays),
    )

    for_each_param_auto[AutoLinear[4, 3], _RecordVisitor](
        lin, String("layer0"), v,
    )

    print("AutoLinear[4,3] params discovered via reflection:")
    for i in range(len(names)):
        print(
            "  name=", names[i],
            "  n_elems=", sizes[i],
            "  apply_decay=", decays[i],
        )

    # Expected: layer0.weight (12 elems, decay=True), layer0.bias (3, False).
    var ok = (
        len(names) == 2
        and names[0] == "layer0.weight" and sizes[0] == 12 and decays[0]
        and names[1] == "layer0.bias"   and sizes[1] == 3  and not decays[1]
    )
    if ok:
        print()
        print("PASS — reflection auto-derived for_each_param works.")
        print("       AutoLinear declares ZERO lines of `for_each_param` body.")
    else:
        print("FAIL — reflection-walked param list does not match expected.")
        raise Error("reflection auto-for_each_param failed")
