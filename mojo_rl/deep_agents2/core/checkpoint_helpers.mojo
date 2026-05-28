"""Per-file v2 save/load helpers for non-Module state.

Wraps the `Saveable` and ScalarAdam serialization paths in the same
`nn2-ckpt v2\\n<body>` envelope that `save_state_v2[M: Module]` uses,
so an Agent's `save(path)` writes a small directory tree of `.ckpt`
files (one per module + one per optimizer + one for ScalarAdam) that
all share the same on-disk format.

CPU only. GPU trainers must download device → host before calling
these (Track 1 scope: SAC/DDPG/TD3/PPO/MBPO save/load are gated to
CPU train_target).

API:
  * `save_optimizer_v2[O: Saveable](opt, path)` — for `Adam` / `AdamW`.
  * `load_optimizer_v2[O: Saveable](opt, path)` — inverse.
  * `save_scalar_adam_v2(opt, path)` — for `ScalarAdam` (not Saveable today;
                                       hand-rolled to keep the change
                                       minimal in `nn2/`).
  * `load_scalar_adam_v2(opt, path)` — inverse.
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.save_scalar import _expect_kv_line
from mojo_rl.nn2.core.saveable import Saveable
from mojo_rl.nn2.optimizer.scalar_adam import ScalarAdam


# ──────────────────────────────────────────────────────────────────────
# Shared helpers (private)
# ──────────────────────────────────────────────────────────────────────


def _split_lines(content: String) -> List[String]:
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


def _read_file(path: String) raises -> String:
    with open(path, "r") as f:
        return String(f.read())


def _expect_v2_header(lines: List[String]) raises:
    if len(lines) == 0 or lines[0] != String("nn2-ckpt v2"):
        var got = String("<empty>") if len(lines) == 0 else lines[0]
        raise Error(
            "checkpoint_helpers: expected `nn2-ckpt v2` header, got `"
            + got + "`"
        )


# ──────────────────────────────────────────────────────────────────────
# Optimizer (Saveable) save/load — wraps any Saveable in a v2 file.
# ──────────────────────────────────────────────────────────────────────


def save_optimizer_v2[O: Saveable](mut opt: O, path: String) raises:
    """Serialize a Saveable optimizer (Adam / AdamW) to `path` in the
    `nn2-ckpt v2` envelope. Uses `opt` as the dotted-path prefix."""
    var body = String("")
    opt.save(body, "opt")
    var content = String("nn2-ckpt v2\n") + body
    with open(path, "w") as f:
        f.write(content)


def load_optimizer_v2[O: Saveable](mut opt: O, path: String) raises:
    """Inverse of `save_optimizer_v2`. The optimizer must have been
    `make(...)`-ed against the matching model before this call so its
    internal `total_size` is populated — `opt.load` validates the saved
    `m_flat#size=N` against this in-memory value."""
    var content = _read_file(path)
    var lines = _split_lines(content)
    _expect_v2_header(lines)
    var idx: Int = 1
    opt.load(lines, idx, "opt")


# ──────────────────────────────────────────────────────────────────────
# ScalarAdam save/load — hand-rolled (ScalarAdam isn't Saveable today).
#
# Format (inside v2 envelope):
#   nn2-ckpt v2
#   scalar_adam.value=<float>
#   scalar_adam.m=<float>
#   scalar_adam.v=<float>
#   scalar_adam.t=<int>
#   scalar_adam.lr=<float>
#   scalar_adam.beta1=<float>
#   scalar_adam.beta2=<float>
#   scalar_adam.eps=<float>
# ──────────────────────────────────────────────────────────────────────


def save_scalar_adam_v2(mut opt: ScalarAdam, path: String) raises:
    var content = String("nn2-ckpt v2\n")
    content += "scalar_adam.value=" + String(opt.value) + "\n"
    content += "scalar_adam.m=" + String(opt.m) + "\n"
    content += "scalar_adam.v=" + String(opt.v) + "\n"
    content += "scalar_adam.t=" + String(opt.t) + "\n"
    content += "scalar_adam.lr=" + String(opt.lr) + "\n"
    content += "scalar_adam.beta1=" + String(opt.beta1) + "\n"
    content += "scalar_adam.beta2=" + String(opt.beta2) + "\n"
    content += "scalar_adam.eps=" + String(opt.eps) + "\n"
    with open(path, "w") as f:
        f.write(content)


def load_scalar_adam_v2(mut opt: ScalarAdam, path: String) raises:
    var content = _read_file(path)
    var lines = _split_lines(content)
    _expect_v2_header(lines)
    opt.value = Scalar[DT](atof(_expect_kv_line(lines, 1, "scalar_adam.value")))
    opt.m     = Scalar[DT](atof(_expect_kv_line(lines, 2, "scalar_adam.m")))
    opt.v     = Scalar[DT](atof(_expect_kv_line(lines, 3, "scalar_adam.v")))
    opt.t     = atol(_expect_kv_line(lines, 4, "scalar_adam.t"))
    opt.lr    = Scalar[DT](atof(_expect_kv_line(lines, 5, "scalar_adam.lr")))
    opt.beta1 = Scalar[DT](atof(_expect_kv_line(lines, 6, "scalar_adam.beta1")))
    opt.beta2 = Scalar[DT](atof(_expect_kv_line(lines, 7, "scalar_adam.beta2")))
    opt.eps   = Scalar[DT](atof(_expect_kv_line(lines, 8, "scalar_adam.eps")))
