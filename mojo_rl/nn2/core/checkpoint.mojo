"""Checkpoint v1 — walker-derived save/load of all Param fields.

Phase 1.6 minimum-viable checkpoint format. Reuses `for_each_param_auto`
(`core/walkers.mojo`) to dump every `Param[NAME, DECAY, SIZE]` field of
a model and read it back into a fresh instance.

Format: plain text, one Scalar[DT] value per line, in walk order.
Header line: `nn2-ckpt v1 <total_n_elems>`. No section names — load
expects the *same* network topology to ensure walk-order matches.
Comment lines (start with `#`) and empty lines are skipped on load.

Round-trip contract:
    var net  = MyNet.make[target="cpu", INIT=Kaiming]()
    # ... train ...
    save_params(net, "ckpt.txt")

    var fresh = MyNet.make[target="cpu", INIT=Kaiming]()
    load_params(fresh, "ckpt.txt")
    # forward(net, x) == forward(fresh, x) for any x

CPU-only in v1. GPU checkpoints will download to host first (Phase 5
follow-up if MBRL training needs cross-platform save / load).
"""

from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from .module import Module
from .param_visitor import ParamVisitor


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
