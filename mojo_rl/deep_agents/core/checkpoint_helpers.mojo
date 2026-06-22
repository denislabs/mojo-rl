"""Generic `nn-ckpt v2` envelope helpers (string + counter sections).

Framework-agnostic checkpoint-file utilities shared by agent facades:

  * **Envelope string IO** (`split_lines_v2` / `read_file_v2` /
    `expect_v2_header`): split a checkpoint body on '\\n', read a file, and
    validate the `nn-ckpt v2` header.
  * **Scalar Int counter section** (`save_counter_v2_body` /
    `load_counter_v2_body`): persist a cumulative trainer counter (e.g.
    `_total_train_steps`) as one `<prefix>=<value>` line; `load` is tolerant
    of absence (older checkpoints simply leave the value unchanged).

The legacy optimizer-state save/load helpers that used to live here (typed on
the legacy `nn.optimizer.{Adam,AdamW,ScalarAdam}`) were removed in the legacy-`nn`
removal (Phase 1): migrated agents checkpoint optimizer moments through the
storage `CheckpointWriter`/`CheckpointReader` param-arena path instead, so this
module no longer imports the legacy framework.

CPU only. GPU trainers must download device → host before serializing.
"""

from mojo_rl.nn.core.save_scalar import _expect_kv_line


# ──────────────────────────────────────────────────────────────────────
# Envelope string IO
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
