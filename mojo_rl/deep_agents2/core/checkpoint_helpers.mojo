"""Per-section v2 save/load helpers for composite agent checkpoints.

Two API levels:

  * **Single-file convenience** (`save_optimizer_v2` / `load_optimizer_v2`
    / `save_scalar_adam_v2` / `load_scalar_adam_v2`): wraps one Saveable
    in its own `nn2-ckpt v2\\n<body>` file. Used in tests + bring-up.
  * **Body-level** (`save_optimizer_v2_body` / `load_optimizer_v2_body`
    / `save_scalar_adam_v2_body` / `load_scalar_adam_v2_body`): append
    a prefixed section into an existing String / consume from a parsed
    line buffer. Used by Agent facades that pack several modules +
    optimizers into ONE `.ckpt` file under a single v2 envelope.

CPU only. GPU trainers must download device → host before calling
these.
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.save_scalar import _expect_kv_line
from mojo_rl.nn2.core.saveable import Saveable
from mojo_rl.nn2.optimizer.scalar_adam import ScalarAdam


# ──────────────────────────────────────────────────────────────────────
# Shared helpers (public — single-section file IO)
# ──────────────────────────────────────────────────────────────────────


def split_lines_v2(content: String) -> List[String]:
    """Split a String on '\\n' (matches nn2/core/checkpoint.mojo)."""
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
    if len(lines) == 0 or lines[0] != String("nn2-ckpt v2"):
        var got = String("<empty>") if len(lines) == 0 else lines[0]
        raise Error(
            "checkpoint_helpers: expected `nn2-ckpt v2` header, got `"
            + got + "`"
        )


# ──────────────────────────────────────────────────────────────────────
# Body-level helpers — append/consume one prefixed section.
# ──────────────────────────────────────────────────────────────────────


def save_optimizer_v2_body[O: Saveable](
    mut opt: O, mut out: String, prefix: String,
) raises:
    """Append the Saveable optimizer's serialized section to `out` under
    `prefix`. No v2 header is written — the caller assembles the envelope
    once it has appended every section."""
    opt.save(out, prefix)


def load_optimizer_v2_body[O: Saveable](
    mut opt: O,
    lines: List[String],
    mut idx: Int,
    prefix: String,
) raises:
    """Consume the Saveable optimizer's section from `lines[idx:]` under
    `prefix`. Advances `idx` past the consumed section."""
    opt.load(lines, idx, prefix)


def save_scalar_adam_v2_body(
    opt: ScalarAdam, mut out: String, prefix: String,
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


# ──────────────────────────────────────────────────────────────────────
# Single-section convenience wrappers — one Saveable per file.
# ──────────────────────────────────────────────────────────────────────


def save_optimizer_v2[O: Saveable](mut opt: O, path: String) raises:
    """Serialize a Saveable optimizer (Adam / AdamW) to its own `.ckpt`
    file in the `nn2-ckpt v2` envelope. Uses `opt` as the dotted-path
    prefix. Kept for tests + single-module workflows."""
    var body = String("")
    save_optimizer_v2_body(opt, body, "opt")
    var content = String("nn2-ckpt v2\n") + body
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
    var content = String("nn2-ckpt v2\n") + body
    with open(path, "w") as f:
        f.write(content)


def load_scalar_adam_v2(mut opt: ScalarAdam, path: String) raises:
    var content = read_file_v2(path)
    var lines = split_lines_v2(content)
    expect_v2_header(lines)
    var idx: Int = 1
    load_scalar_adam_v2_body(opt, lines, idx, "scalar_adam")
