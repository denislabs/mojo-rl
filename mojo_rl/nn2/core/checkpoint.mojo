"""Checkpoint — save / load of nn2 state.

Two formats coexist:

  - **v1** (`save_params` / `load_params`): walker-derived, flat list of
    all Scalar[DT] values in walk order, header `nn2-ckpt v1 <N>`.
    Order-dependent. Predates Phase A. Kept for backward compat.
  - **v2** (`save_state_v2` / `load_state_v2`): reflection-derived,
    named per-Param sections, header `nn2-ckpt v2`. Robust to topology
    drift between save and load — each section's `<prefix>#size=<N>`
    header is verified on load.

v2 uses the `Saveable` walker (`state_walker.mojo`). Currently CPU-only
for both formats. GPU checkpoints must download device → host before
save and re-upload after load.
"""

from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from .module import Module
from .param_visitor import ParamVisitor
from .state_walker import dump_state, load_state


# ──────────────────────────────────────────────────────────────────────
# _SaveVisitor — accumulates every visited param's values into a
# local List field. The walker passes the visitor by reference, so
# mutations to `self.values` persist across visits.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _SaveVisitor(ParamVisitor):
    var values: List[Scalar[DT]]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var p_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        for k in range(n_elems):
            self.values.append(p_ptr[k])


# ──────────────────────────────────────────────────────────────────────
# _LoadVisitor — consumes pre-parsed values in walk order via a
# struct-local `idx` field. Writes each param's n_elems consecutive
# values into the param's underlying buffer.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _LoadVisitor(ParamVisitor):
    var values: List[Scalar[DT]]
    var idx: Int

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var p_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        for k in range(n_elems):
            p_ptr[k] = self.values[self.idx + k]
        self.idx += n_elems


# ──────────────────────────────────────────────────────────────────────
# Helpers.
# ──────────────────────────────────────────────────────────────────────


def _split_lines(content: String) -> List[String]:
    """Split a String on '\\n'. Mirrors `nn/checkpoint`'s helper since
    `String.split` semantics in Mojo nightly aren't fully nailed down."""
    var lines = List[String]()
    var current_line = String("")
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord("\n")):
            lines.append(current_line)
            current_line = String("")
        else:
            current_line += chr(Int(c))
    if current_line.byte_length() > 0:
        lines.append(current_line)
    return lines^


def _read_file(path: String) raises -> String:
    """Read the whole file as a String. Wrapped in its own function
    because Mojo nightly's `with open(...) as f:` block scope doesn't
    propagate reassignments cleanly — return from inside the block."""
    with open(path, "r") as f:
        return String(f.read())


# ──────────────────────────────────────────────────────────────────────
# Public API.
# ──────────────────────────────────────────────────────────────────────


def save_params[M: Module](mut model: M, path: String) raises:
    """Walk every `Param` field of `model` and write its values to `path`.

    Format: a header line `nn2-ckpt v1 <total_n_elems>`, then one
    Scalar[DT] value per line in walk order.

    Bound on `Module`: combinators (`Sequential`, `Residual`, …) override
    `for_each_param` to recurse into children, so this dispatches correctly
    via the trait method (rather than the leaf-only `for_each_param_auto`
    free-function walker).
    """
    var v = _SaveVisitor(values=List[Scalar[DT]]())
    model.for_each_param[target="cpu", V=_SaveVisitor](
        String(""), v,
    )

    var content = String("nn2-ckpt v1 ") + String(len(v.values)) + String("\n")
    for k in range(len(v.values)):
        content += String(v.values[k]) + String("\n")
    with open(path, "w") as f:
        f.write(content)


def load_params[M: Module](mut model: M, path: String) raises:
    """Read `path` (saved by `save_params`) and overwrite each `Param`
    field of `model` in walk order. The model's network topology MUST
    match the one saved or behaviour is undefined."""
    var content = _read_file(path)
    var lines = _split_lines(content)

    var values = List[Scalar[DT]]()
    for li in range(len(lines)):
        var line = lines[li]
        if line.byte_length() == 0:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("nn2-ckpt"):
            continue
        values.append(Scalar[DT](atof(line)))

    var lv = _LoadVisitor(values=values^, idx=0)
    model.for_each_param[target="cpu", V=_LoadVisitor](
        String(""), lv,
    )


# ──────────────────────────────────────────────────────────────────────
# Checkpoint v2 — reflection-walked, named-section format.
#
# Format:
#   nn2-ckpt v2
#   <body lines emitted by dump_state>
#
# Body lines are produced by Saveable conformers via `dump_state` (which
# walks the struct's fields via reflection and calls each Saveable
# field's `save(out, prefix)`). For `dump_state[Module]`, this finds
# every `Param[NAME, DECAY, SIZE]` field (each is Saveable since Phase
# A.2) and emits a section per Param:
#
#   <field_path>#size=<SIZE>
#   v0
#   v1
#   ...
#
# Where `<field_path>` is the dotted accumulated prefix (e.g.
# `model.layers.0.weight`). On load, each section's header is verified
# against the in-memory Param's compile-time SIZE — catches topology
# drift between save and load early instead of silently corrupting state.
#
# CPU-only. Trainer must download device → host before save / re-upload
# after load when training on GPU.
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# v2 emits the same on-disk format as `Param.save` (since A.2: section
# header `<path>#size=<N>` followed by N value lines). We can't reach
# every Module's Params via reflection alone — combinators like
# `Sequential` hold children as `Tuple[*MODULES]`, which isn't Saveable,
# so reflection skips it. But every Module already implements
# `for_each_param`, which knows how to walk Params recursively (via the
# `IsParam` visitor surface from `walkers.mojo`).
#
# So v2 piggybacks on `for_each_param` with a visitor that emits the
# `Param.save`-equivalent lines. Trainers-as-Saveable will eventually
# (A.5) wrap this with a top-level reflection walk that visits the
# `model` field via this helper plus other Saveable fields via
# `dump_state`. For A.2, the Module-only entry point is enough.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _SaveStateV2Visitor(ParamVisitor):
    """Emits one `<name>#size=<N>` section + N value lines per visited
    Param. Output matches `Param.save` byte-for-byte so dump_state and
    this visitor can be composed later."""
    var out_ptr: UnsafePointer[String, MutAnyOrigin]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var p_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var section = name + "#size=" + String(n_elems) + "\n"
        for k in range(n_elems):
            section += String(p_ptr[k]) + "\n"
        self.out_ptr[] += section


@fieldwise_init
struct _LoadStateV2Visitor(ParamVisitor):
    """Consumes one section from `lines[idx:]` per visited Param.
    Validates header `<name>#size=<n_elems>` matches the in-memory
    Param's name + compile-time SIZE."""
    var lines: List[String]
    var idx_ptr: UnsafePointer[Int, MutAnyOrigin]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var idx = self.idx_ptr[]
        if idx >= len(self.lines):
            raise Error(
                "load_state_v2: out of input. Expected `"
                + name + "#size=" + String(n_elems) + "` at line "
                + String(idx)
            )
        var header = self.lines[idx]
        var expected = name + "#size=" + String(n_elems)
        if header != expected:
            raise Error(
                "load_state_v2: header mismatch at line " + String(idx)
                + ". Expected `" + expected + "`, got `" + header + "`"
            )
        idx += 1
        var p_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        for k in range(n_elems):
            if idx >= len(self.lines):
                raise Error(
                    "load_state_v2: short read at element " + String(k)
                    + " of " + String(n_elems) + " for `" + name + "`"
                )
            p_ptr[k] = Scalar[DT](atof(self.lines[idx]))
            idx += 1
        self.idx_ptr[] = idx


def save_state_v2[M: Module](mut model: M, path: String) raises:
    """Write all Param values in `model` to `path` in v2 format."""
    var body = String("")
    var v = _SaveStateV2Visitor(out_ptr=UnsafePointer(to=body))
    model.for_each_param[target="cpu", V=_SaveStateV2Visitor](
        String(""), v,
    )
    _ = body^  # lifetime extender — visitor holds UnsafePointer(to=body)

    var content = String("nn2-ckpt v2\n") + body
    with open(path, "w") as f:
        f.write(content)


def load_state_v2[M: Module](mut model: M, path: String) raises:
    """Read a v2 checkpoint and overwrite every Param in `model`."""
    var content = _read_file(path)
    var lines = _split_lines(content)
    if len(lines) == 0 or lines[0] != String("nn2-ckpt v2"):
        raise Error(
            "load_state_v2: expected `nn2-ckpt v2` header, got `"
            + (lines[0] if len(lines) > 0 else String("<empty>")) + "`"
        )
    var idx: Int = 1
    var v = _LoadStateV2Visitor(
        lines=lines^, idx_ptr=UnsafePointer(to=idx),
    )
    model.for_each_param[target="cpu", V=_LoadStateV2Visitor](
        String(""), v,
    )
    _ = idx  # lifetime extender — visitor holds UnsafePointer(to=idx)
