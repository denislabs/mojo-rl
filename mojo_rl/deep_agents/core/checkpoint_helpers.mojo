"""Per-section v2 save/load helpers for composite agent checkpoints.

Two API levels:

  * **Single-file convenience** (`save_optimizer_v2` / `load_optimizer_v2`
    / `save_scalar_adam_v2` / `load_scalar_adam_v2`): wraps one Saveable
    in its own `nn-ckpt v2\\n<body>` file. Used in tests + bring-up.
  * **Body-level** (`save_optimizer_v2_body` / `load_optimizer_v2_body`
    / `save_scalar_adam_v2_body` / `load_scalar_adam_v2_body`): append
    a prefixed section into an existing String / consume from a parsed
    line buffer. Used by Agent facades that pack several modules +
    optimizers into ONE `.ckpt` file under a single v2 envelope.

CPU only. GPU trainers must download device → host before calling
these.
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.save_scalar import _expect_kv_line
from mojo_rl.nn.core.saveable import Saveable
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.optimizer.adamw import AdamW
from mojo_rl.nn.optimizer.scalar_adam import ScalarAdam


# ──────────────────────────────────────────────────────────────────────
# Shared helpers (public — single-section file IO)
# ──────────────────────────────────────────────────────────────────────


def split_lines_v2(content: String) -> List[String]:
    """Split a String on '\\n' (matches nn/core/checkpoint.mojo)."""
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


def read_file_v2(path: String) raises -> String:
    with open(path, "r") as f:
        return String(f.read())


def expect_v2_header(lines: List[String]) raises:
    if len(lines) == 0 or lines[0] != String("nn-ckpt v2"):
        var got = String("<empty>") if len(lines) == 0 else lines[0]
        raise Error(
            "checkpoint_helpers: expected `nn-ckpt v2` header, got `"
            + got
            + "`"
        )


# ──────────────────────────────────────────────────────────────────────
# Body-level helpers — append/consume one prefixed section.
# ──────────────────────────────────────────────────────────────────────


def save_optimizer_v2_body[
    O: Saveable
](mut opt: O, mut out: String, prefix: String,) raises:
    """Append the Saveable optimizer's serialized section to `out` under
    `prefix`. No v2 header is written — the caller assembles the envelope
    once it has appended every section."""
    opt.save(out, prefix)


def load_optimizer_v2_body[
    O: Saveable
](mut opt: O, lines: List[String], mut idx: Int, prefix: String,) raises:
    """Consume the Saveable optimizer's section from `lines[idx:]` under
    `prefix`. Advances `idx` past the consumed section."""
    opt.load(lines, idx, prefix)


# ──────────────────────────────────────────────────────────────────────
# GPU Adam save/load (Phase 2 — GPU checkpointing).
#
# Byte-identical to the CPU optimizer section: GPU save D2Hs the device
# buffers into the Adam's host fields (`sync_to_host`), then runs the
# SAME CPU serializer; GPU load runs the SAME CPU parser, then H2Ds the
# restored host fields (`upload_from_host`). A GPU checkpoint therefore
# loads on a CPU trainer unchanged — train-on-GPU → eval-on-CPU.
# ──────────────────────────────────────────────────────────────────────


def save_optimizer_v2_body_gpu(
    mut opt: Adam,
    mut out: String,
    prefix: String,
) raises:
    """GPU `Adam` section. Downloads device state into the Adam's own host
    fields, then emits the identical CPU section. (`ctx` is read from the
    optimizer's own `TargetStorage`.)."""
    opt.sync_to_host()
    opt.save(out, prefix)


def load_optimizer_v2_body_gpu(
    mut opt: Adam,
    lines: List[String],
    mut idx: Int,
    prefix: String,
) raises:
    """Inverse of `save_optimizer_v2_body_gpu`: CPU parse into host
    fields, then upload to the device buffers.

    A GPU-built `Adam` has EMPTY `m_flat`/`v_flat` host lists (the live
    moments live in `m_dev`/`v_dev`). `Adam.load` writes
    `total_size` values straight into those lists' buffers, so they MUST
    be pre-sized or the writes corrupt the heap. Size them here before
    parsing, then upload to device."""
    if len(opt.m_flat) != opt.total_size:
        opt.m_flat = List[Scalar[DT]](
            length=opt.total_size, fill=Scalar[DT](0.0)
        )
        opt.v_flat = List[Scalar[DT]](
            length=opt.total_size, fill=Scalar[DT](0.0)
        )
    opt.load(lines, idx, prefix)
    opt.upload_from_host()


def save_optimizer_v2_body_gpu(
    mut opt: AdamW,
    mut out: String,
    prefix: String,
) raises:
    """GPU `AdamW` section (overload). Downloads device state into the
    optimizer's host fields, then emits the identical CPU section. Used by
    the MBPO dynamics ensemble, which optimises with decoupled weight
    decay."""
    opt.sync_to_host()
    opt.save(out, prefix)


def load_optimizer_v2_body_gpu(
    mut opt: AdamW,
    lines: List[String],
    mut idx: Int,
    prefix: String,
) raises:
    """Inverse of the `AdamW` `save_optimizer_v2_body_gpu` overload: CPU
    parse into host fields, then upload to the device buffers. A GPU-built
    `AdamW` has empty `m_flat`/`v_flat`; size them before parsing so
    `AdamW.load` writes into valid storage."""
    if len(opt.m_flat) != opt.total_size:
        opt.m_flat = List[Scalar[DT]](
            length=opt.total_size, fill=Scalar[DT](0.0)
        )
        opt.v_flat = List[Scalar[DT]](
            length=opt.total_size, fill=Scalar[DT](0.0)
        )
    opt.load(lines, idx, prefix)
    opt.upload_from_host()


def save_scalar_adam_v2_body(
    opt: ScalarAdam,
    mut out: String,
    prefix: String,
):
    """Append a ScalarAdam's serialized section to `out` under `prefix`."""
    out += prefix + ".value=" + String(opt.value) + "\n"
    out += prefix + ".m=" + String(opt.m) + "\n"
    out += prefix + ".v=" + String(opt.v) + "\n"
    out += prefix + ".t=" + String(opt.t) + "\n"
    out += prefix + ".lr=" + String(opt.lr) + "\n"
    out += prefix + ".beta1=" + String(opt.beta1) + "\n"
    out += prefix + ".beta2=" + String(opt.beta2) + "\n"
    out += prefix + ".eps=" + String(opt.eps) + "\n"


def save_scalar_adam_v2_body_gpu(
    mut opt: ScalarAdam,
    mut out: String,
    prefix: String,
) raises:
    """GPU `ScalarAdam` section. Syncs device state into the host fields,
    then runs the SAME CPU serializer (byte-identical, interchangeable
    format). See `ScalarAdam.sync_to_host` for the accepted bias-
    correction gap on the GPU path."""
    opt.sync_to_host()
    save_scalar_adam_v2_body(opt, out, prefix)


def load_scalar_adam_v2_body_gpu(
    mut opt: ScalarAdam,
    lines: List[String],
    mut idx: Int,
    prefix: String,
) raises:
    """Inverse of `save_scalar_adam_v2_body_gpu`: CPU parse into host
    fields, then upload to `state_dev`."""
    load_scalar_adam_v2_body(opt, lines, idx, prefix)
    opt.upload_from_host()


def load_scalar_adam_v2_body(
    mut opt: ScalarAdam,
    lines: List[String],
    mut idx: Int,
    prefix: String,
) raises:
    """Consume a ScalarAdam section (8 key=value lines) from `lines`."""
    opt.value = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".value")))
    idx += 1
    opt.m = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".m")))
    idx += 1
    opt.v = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".v")))
    idx += 1
    opt.t = atol(_expect_kv_line(lines, idx, prefix + ".t"))
    idx += 1
    opt.lr = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".lr")))
    idx += 1
    opt.beta1 = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".beta1")))
    idx += 1
    opt.beta2 = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".beta2")))
    idx += 1
    opt.eps = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".eps")))
    idx += 1
    # Reconstruct the incremental bias-correction products β₁ᵗ / β₂ᵗ from
    # the restored `t` by replaying the same running product the step loop
    # builds (1·β·β·… , t times). Bit-identical to having stepped t times,
    # so CPU save/resume stays byte-stable. (Serializing them directly
    # would be equivalent but bloat the v2 envelope; t is the source of
    # truth.)
    opt.beta1_pow_t = Scalar[DT](1.0)
    opt.beta2_pow_t = Scalar[DT](1.0)
    for _ in range(opt.t):
        opt.beta1_pow_t *= opt.beta1
        opt.beta2_pow_t *= opt.beta2


# ──────────────────────────────────────────────────────────────────────
# Scalar Int counter section — one `<prefix>=<value>` line.
#
# Used for cumulative trainer counters (e.g. `_total_train_steps`) that
# must survive save/resume. PER β-anneal schedules key on this counter,
# so dropping it restarts β annealing on every resume.
#
# `load` is TOLERANT of absence: a checkpoint written before the counter
# section existed simply leaves `value` unchanged (idx not advanced).
# The counter section is therefore always appended LAST in each envelope
# so older checkpoints still parse cleanly.
# ──────────────────────────────────────────────────────────────────────


def save_counter_v2_body(value: Int, mut out: String, prefix: String):
    """Append a single `<prefix>=<value>` counter line to `out`."""
    out += prefix + "=" + String(value) + "\n"


def load_counter_v2_body(
    mut value: Int, lines: List[String], mut idx: Int, prefix: String,
) raises:
    """Consume a `<prefix>=<value>` counter line from `lines[idx:]`,
    advancing `idx`. If the stream is exhausted (older checkpoint with no
    counter section) `value` is left unchanged and `idx` is not advanced."""
    if idx >= len(lines):
        return
    value = atol(_expect_kv_line(lines, idx, prefix))
    idx += 1


# ──────────────────────────────────────────────────────────────────────
# Single-section convenience wrappers — one Saveable per file.
# ──────────────────────────────────────────────────────────────────────


def save_optimizer_v2[O: Saveable](mut opt: O, path: String) raises:
    """Serialize a Saveable optimizer (Adam / AdamW) to its own `.ckpt`
    file in the `nn-ckpt v2` envelope. Uses `opt` as the dotted-path
    prefix. Kept for tests + single-module workflows."""
    var body = String("")
    save_optimizer_v2_body(opt, body, "opt")
    var content = String("nn-ckpt v2\n") + body
    with open(path, "w") as f:
        f.write(content)


def load_optimizer_v2[O: Saveable](mut opt: O, path: String) raises:
    """Inverse of `save_optimizer_v2`."""
    var content = read_file_v2(path)
    var lines = split_lines_v2(content)
    expect_v2_header(lines)
    var idx: Int = 1
    load_optimizer_v2_body(opt, lines, idx, "opt")


def save_scalar_adam_v2(opt: ScalarAdam, path: String) raises:
    var body = String("")
    save_scalar_adam_v2_body(opt, body, "scalar_adam")
    var content = String("nn-ckpt v2\n") + body
    with open(path, "w") as f:
        f.write(content)


def load_scalar_adam_v2(mut opt: ScalarAdam, path: String) raises:
    var content = read_file_v2(path)
    var lines = split_lines_v2(content)
    expect_v2_header(lines)
    var idx: Int = 1
    load_scalar_adam_v2_body(opt, lines, idx, "scalar_adam")
