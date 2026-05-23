"""Parametric Saveable wrappers for scalar leaves.

Phase A.1. Codebase convention (`mojo_rl/nn/autodiff/fused/activation.mojo:120`
and elsewhere): a single parametric `[DT: DType]` wrapper gated by
`comptime assert Self.DT.is_floating_point()` covers every supported
float dtype (Float32, Float64, BF16, …). For integer counters (step
counters, replay cursors, etc.) that genuinely need `Int`, use the
separate `SaveI` wrapper.

Line format:
  <prefix>=<value>\n

Where <prefix> is the dotted path accumulated by the walker (e.g.
`actor_opt.m`). Reader uses `atof` / `atol` on the right-hand side.
"""

from .saveable import Saveable


# ──────────────────────────────────────────────────────────────────────
# Parametric floating-point scalar wrapper. Gate uses the codebase's
# `comptime assert dtype.is_floating_point()` idiom — see
# `mojo_rl/nn/autodiff/fused/activation.mojo:120`.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct SaveScalar[DT: DType](Saveable, Copyable):
    var v: Scalar[Self.DT]

    def save(self, mut out: String, prefix: String) raises:
        comptime assert Self.DT.is_floating_point(), "dtype must be floating point"
        out += prefix + "=" + String(self.v) + "\n"

    def load(
        mut self, lines: List[String], mut idx: Int, prefix: String,
    ) raises:
        comptime assert Self.DT.is_floating_point(), "dtype must be floating point"
        var value_str = _expect_kv_line(lines, idx, prefix)
        idx += 1
        self.v = Scalar[Self.DT](atof(value_str))


# ──────────────────────────────────────────────────────────────────────
# Integer counter wrapper. No DT — Int is Int.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct SaveI(Saveable, Copyable):
    var v: Int

    def save(self, mut out: String, prefix: String) raises:
        out += prefix + "=" + String(self.v) + "\n"

    def load(
        mut self, lines: List[String], mut idx: Int, prefix: String,
    ) raises:
        var value_str = _expect_kv_line(lines, idx, prefix)
        idx += 1
        self.v = atol(value_str)


# ──────────────────────────────────────────────────────────────────────
# Line-parser helper. Shared by the wrappers above (and by any future
# Saveable leaf that wants to consume a `name=value` line from the
# stream). Kept here rather than in state_walker.mojo so leaves don't
# pull in the walker module just to parse a line.
# ──────────────────────────────────────────────────────────────────────


def _expect_kv_line(
    lines: List[String], idx: Int, expected_prefix: String,
) raises -> String:
    """Reads `lines[idx]`, asserts it has the form `<expected_prefix>=<value>`,
    returns the <value> substring. Raises if the prefix doesn't match —
    this catches checkpoint format drift early rather than letting it
    silently corrupt loaded state."""
    if idx >= len(lines):
        raise Error(
            "Saveable.load: out of input. Expected line `"
            + expected_prefix + "=...` at idx " + String(idx)
            + " but only have " + String(len(lines)) + " lines."
        )
    var line = lines[idx]
    var sep_pos = line.find("=")
    if sep_pos < 0:
        raise Error(
            "Saveable.load: line " + String(idx)
            + " missing `=`: `" + line + "`"
        )
    var got_prefix = line[byte=:sep_pos]
    if String(got_prefix) != expected_prefix:
        raise Error(
            "Saveable.load: prefix mismatch at line " + String(idx)
            + ". Expected `" + expected_prefix
            + "`, got `" + String(got_prefix) + "`"
        )
    return String(line[byte=(sep_pos + 1):])
