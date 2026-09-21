# +--------------------------------------------------------------------------+ #
# | A checkpoint load accounts for every byte — and outlives its own walk
# +--------------------------------------------------------------------------+ #
"""Gate `nn/core/checkpoint.mojo`'s `finish()`.

    pixi run mojo run -I . tests/nn/test_checkpoint_finish.mojo

`finish()` does two jobs, and only one of them is deterministically testable:

 1. ⚠ IT KEEPS THE READER ALIVE. `ParamVisitorRef.of(r)` hands the state pass
    a POINTER to `r`; with no later mention, that call is `r`'s last use and
    Mojo destroys it there, so the state pass walks freed memory. On
    2026-09-16 an ACT deployment loaded 253 tensors and then failed on the
    first BatchNorm buffer with "unexpected end of file" against a file of
    `0 bytes` — the 225 MB file was byte-perfect. A use-after-free is not
    reliably fatal (the same pattern passes with a 4-neuron model, whose freed
    pages are still intact), so this file CANNOT gate it. It is gated by the
    call existing, and by the note at `finish()`.

 2. Bytes left unread mean the file holds tensors this build does not walk.
    That IS deterministic, and it is what the checks below assert.
"""

from std.os.path import exists

from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import load_params, save_params
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.batch_norm_1d import BatchNorm1D
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential
from noeira.io.fileio import read_file_bytes, write_file_atomic


comptime D = 4
comptime H = 5
comptime O = 3
comptime NET = Sequential[Linear[D, H], BatchNorm1D[H], Linear[H, O]]
comptime PATH = "/tmp/noeira_ckpt_finish_gate.ckpt"


def main() raises:
    print("[checkpoint-finish] gate")
    var n = 0
    var a = NET.make["cpu", Deterministic](None)
    save_params["cpu"](a, String(PATH), None)

    # A plain round trip still loads — the state pass included.
    var b = NET.make["cpu", Deterministic](None)
    load_params["cpu"](b, String(PATH), None)
    print("  a checkpoint with BatchNorm state round-trips")
    n += 1

    # ⚠ A FILE WITH MORE IN IT THAN THIS BUILD WALKS. Appending a section the
    # model has no tensor for is exactly what a checkpoint from a bigger model
    # looks like from here, and it used to load "successfully".
    var bytes = read_file_bytes(String(PATH))
    var extra = String("S ghost.running_mean 2\n")
    for i in range(extra.byte_length()):
        bytes.append(extra.as_bytes()[i])
    for _ in range(2 * 4):
        bytes.append(UInt8(0))
    write_file_atomic(String(PATH) + ".fat", bytes)
    var c = NET.make["cpu", Deterministic](None)
    var raised = String("")
    try:
        load_params["cpu"](c, String(PATH) + ".fat", None)
    except e:
        raised = String(e)
    if raised.find("left unread") < 0:
        raise Error(
            "a checkpoint holding tensors this build does not walk was accepted"
            " (got: " + (raised if raised else String("no error")) + ")"
        )
    print("  a checkpoint with an extra section is refused: " + String(raised[byte=0:60]) + "...")
    n += 1
    print("  " + String(n) + " checks, 0 failures")
    print("[PASS] checkpoint-finish")
