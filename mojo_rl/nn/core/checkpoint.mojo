"""Checkpoint — save / load of nn state.

Two formats coexist:

  - **v1** (`save_params` / `load_params`): walker-derived, flat list of
    all Scalar[DT] values in walk order, header `nn-ckpt v1 <N>`.
    Order-dependent. Predates Phase A. Kept for backward compat.
  - **v2** (`save_state_v2` / `load_state_v2`): reflection-derived,
    named per-Param sections, header `nn-ckpt v2`. Robust to topology
    drift between save and load — each section's `<prefix>#size=<N>`
    header is verified on load.

v2 uses the `Saveable` walker (`state_walker.mojo`). Currently CPU-only
for both formats. GPU checkpoints must download device → host before
save and re-upload after load.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from .module import mptr
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
        var p_ptr = mptr(param.ptr)
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
        var p_ptr = mptr(param.ptr)
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

    Format: a header line `nn-ckpt v1 <total_n_elems>`, then one
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

    var content = String("nn-ckpt v1 ") + String(len(v.values)) + String("\n")
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
        if line.startswith("nn-ckpt"):
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
#   nn-ckpt v2
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
        var p_ptr = mptr(param.ptr)
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
        var p_ptr = mptr(param.ptr)
        for k in range(n_elems):
            if idx >= len(self.lines):
                raise Error(
                    "load_state_v2: short read at element " + String(k)
                    + " of " + String(n_elems) + " for `" + name + "`"
                )
            p_ptr[k] = Scalar[DT](atof(self.lines[idx]))
            idx += 1
        self.idx_ptr[] = idx


def save_state_v2_body[M: Module](
    mut model: M, mut out: String, prefix: String,
) raises:
    """Append every Param's serialized section to `out`, with `prefix`
    accumulated into each section header. Used by composite checkpoints
    that pack several modules + optimizers into a single v2 envelope.

    No header is written — callers prepend `nn-ckpt v2\\n` themselves
    (or use `save_state_v2` for the single-module shortcut)."""
    var v = _SaveStateV2Visitor(out_ptr=UnsafePointer(to=out))
    model.for_each_param[target="cpu", V=_SaveStateV2Visitor](
        prefix, v,
    )
    # S5 Stage 3: persist State (e.g. BatchNorm running stats) right after
    # params, using the same visitor/section format.
    model.for_each_state[target="cpu", V=_SaveStateV2Visitor](
        prefix, v,
    )
    _ = out  # lifetime extender — visitor holds UnsafePointer(to=out)


def load_state_v2_body[M: Module](
    mut model: M,
    lines: List[String],
    mut idx: Int,
    prefix: String,
) raises:
    """Consume Param sections from `lines[idx:]` (advancing `idx`) using
    `prefix` for the expected section header. Counterpart of
    `save_state_v2_body`."""
    var v = _LoadStateV2Visitor(
        lines=lines.copy(), idx_ptr=UnsafePointer(to=idx),
    )
    model.for_each_param[target="cpu", V=_LoadStateV2Visitor](
        prefix, v,
    )
    # S5 Stage 3: load State right after params (same order as save).
    model.for_each_state[target="cpu", V=_LoadStateV2Visitor](
        prefix, v,
    )
    _ = idx  # lifetime extender — visitor holds UnsafePointer(to=idx)


# ──────────────────────────────────────────────────────────────────────
# GPU param save/load (Phase 2 — GPU checkpointing).
#
# Self-contained device→host (save) / host→device (load) visitors. Each
# downloads/uploads one Param's DeviceBuffer through a temp host List and
# emits / consumes the SAME `<name>#size=<N>` + value-lines section the
# CPU visitors produce. So a GPU-saved checkpoint is byte-identical to a
# CPU-saved one — train-on-GPU → eval-on-CPU just calls the normal CPU
# `load_state_v2_body`. One `synchronize` per Param keeps the staging
# buffer valid across the copy.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _SaveStateV2GpuVisitor(ParamVisitor):
    var out_ptr: UnsafePointer[String, MutAnyOrigin]
    var ctx: DeviceContext

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
        var d_ptr = mptr(param.ptr)
        var dev = DeviceBuffer[DT](self.ctx, d_ptr, n_elems, owning=False)
        var host = List[Scalar[DT]](length=n_elems, fill=Scalar[DT](0.0))
        self.ctx.enqueue_copy(host.unsafe_ptr(), dev)
        self.ctx.synchronize()
        var section = name + "#size=" + String(n_elems) + "\n"
        for k in range(n_elems):
            section += String(host[k]) + "\n"
        self.out_ptr[] += section


@fieldwise_init
struct _LoadStateV2GpuVisitor(ParamVisitor):
    var lines: List[String]
    var idx_ptr: UnsafePointer[Int, MutAnyOrigin]
    var ctx: DeviceContext

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
                "load_state_v2_gpu: out of input. Expected `"
                + name + "#size=" + String(n_elems) + "` at line "
                + String(idx)
            )
        var header = self.lines[idx]
        var expected = name + "#size=" + String(n_elems)
        if header != expected:
            raise Error(
                "load_state_v2_gpu: header mismatch at line " + String(idx)
                + ". Expected `" + expected + "`, got `" + header + "`"
            )
        idx += 1
        var host = List[Scalar[DT]](length=n_elems, fill=Scalar[DT](0.0))
        for k in range(n_elems):
            if idx >= len(self.lines):
                raise Error(
                    "load_state_v2_gpu: short read at element " + String(k)
                    + " of " + String(n_elems) + " for `" + name + "`"
                )
            host[k] = Scalar[DT](atof(self.lines[idx]))
            idx += 1
        self.idx_ptr[] = idx
        var d_ptr = mptr(param.ptr)
        var dev = DeviceBuffer[DT](self.ctx, d_ptr, n_elems, owning=False)
        self.ctx.enqueue_copy(dev, host.unsafe_ptr())
        self.ctx.synchronize()


def save_state_v2_body_gpu[M: Module](
    mut model: M, mut out: String, prefix: String, ctx: DeviceContext,
) raises:
    """GPU counterpart of `save_state_v2_body` — byte-identical output."""
    var v = _SaveStateV2GpuVisitor(out_ptr=UnsafePointer(to=out), ctx=ctx)
    model.for_each_param[target="gpu", V=_SaveStateV2GpuVisitor](prefix, v)
    model.for_each_state[target="gpu", V=_SaveStateV2GpuVisitor](prefix, v)
    _ = out  # lifetime extender — visitor holds UnsafePointer(to=out)


def load_state_v2_body_gpu[M: Module](
    mut model: M,
    lines: List[String],
    mut idx: Int,
    prefix: String,
    ctx: DeviceContext,
) raises:
    """GPU counterpart of `load_state_v2_body`."""
    var v = _LoadStateV2GpuVisitor(
        lines=lines.copy(), idx_ptr=UnsafePointer(to=idx), ctx=ctx,
    )
    model.for_each_param[target="gpu", V=_LoadStateV2GpuVisitor](prefix, v)
    model.for_each_state[target="gpu", V=_LoadStateV2GpuVisitor](prefix, v)
    _ = idx  # lifetime extender — visitor holds UnsafePointer(to=idx)


def save_state_v2[M: Module](mut model: M, path: String) raises:
    """Write all Param values in `model` to `path` in v2 format.
    Single-module convenience wrapper over `save_state_v2_body`."""
    var body = String("")
    save_state_v2_body(model, body, String(""))
    var content = String("nn-ckpt v2\n") + body
    with open(path, "w") as f:
        f.write(content)


def load_state_v2[M: Module](mut model: M, path: String) raises:
    """Read a v2 checkpoint and overwrite every Param in `model`.
    Single-module convenience wrapper over `load_state_v2_body`."""
    var content = _read_file(path)
    var lines = _split_lines(content)
    if len(lines) == 0 or lines[0] != String("nn-ckpt v2"):
        raise Error(
            "load_state_v2: expected `nn-ckpt v2` header, got `"
            + (lines[0] if len(lines) > 0 else String("<empty>")) + "`"
        )
    var idx: Int = 1
    load_state_v2_body(model, lines, idx, String(""))
