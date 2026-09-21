"""THE BC POLICY SURVIVES THE TRIP FROM THE TRAINER TO THE DRIVER.

    pixi run mojo run -I . tests/tasks/test_bc_policy.mojo

`examples/libero/libero_bc_train.mojo` fits a network and writes a checkpoint;
`examples/libero/libero_eval_batched.mojo` builds the network again and runs it.
Between them sits `tasks/bc_policy.mojo`. What can go wrong is not the fit:

* the driver builds a DIFFERENT network and `load_params` fills the layers
  whose names and sizes happen to match, leaving the rest at their init — a
  policy that runs, produces plausible actions, and scores nothing like the
  one that trained;
* the normalisation is dropped, so the same weights are fed raw metres where
  they were trained on standardised ones;
* the sidecar belongs to another checkpoint and nothing says so.

## WHAT IT ASSERTS

1. **Round trip, bit for bit.** Weights are perturbed away from their init,
   saved, loaded into a FRESH network, and the two forwards must agree
   EXACTLY on the same input — not within a tolerance: this is a copy, and a
   tolerance would hide a layer that never loaded.
2. ⚠ **THE CONTROL.** A fresh network that did NOT load the checkpoint must
   DISAGREE on that same input. Without this, check 1 passes just as well when
   every weight is whatever `Kaiming` produced from the same seed.
3. **The normalisation round-trips** through the sidecar, and applying it
   reproduces `(x - mu) / sd` word for word.
4. **The refusals**: a sidecar whose widths are not the model's, and a missing
   sidecar, both raise. A policy run under the wrong normalisation is the
   failure mode with no symptom.
"""

from std.os import makedirs
from std.os.path import exists
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.checkpoint import save_params, load_params
from noeira.nn.core.initializer import Kaiming
from noeira.tasks.bc_policy import (
    BcNet, BC_HID, BcNorm, load_bc_norm, write_bc_norm,
)


comptime OBS = 91
comptime ACT = 7
comptime B = 3
comptime TMP = "build/test_bc_policy"


def _forward(mut net: BcNet[OBS, ACT], mut x: Tensor, mut y: Tensor) raises:
    # ⚠ `mut x`: `TensorRefs` borrows MUTABLY, so an immutable parameter does
    # not bind — the error names the pack's constructors, not the caller.
    net.forward["cpu", B](TensorRefs[1](x), y, None)


def main() raises:
    print("=== the BC policy from the trainer to the driver ===")
    var checks = 0
    var failures = 0
    if not exists(String(TMP)):
        makedirs(String(TMP))
    var ckpt = String(TMP) + "/policy.ckpt"

    # a deterministic input
    var x = Tensor.alloc(B * OBS)
    for i in range(B * OBS):
        x.data[i] = Scalar[DT](0.01 * Float64((i * 37) % 101) - 0.5)

    # ── 1. round trip ─────────────────────────────────────────────────────
    var trained = BcNet[OBS, ACT].make["cpu", Kaiming](None)
    # ⚠ PERTURBED AWAY FROM THE INIT: two Kaiming networks built from the same
    # seed would round-trip trivially. `walk_params` is how the checkpoint
    # sees them, and `for_each_param` is how a trainer's optimizer does; here
    # the first layer's weights are moved by hand through the module's field.
    for i in range(OBS * BC_HID):
        trained.children[0].weight.val.data[i] += Scalar[DT](
            0.001 * Float64(i % 17) + 0.05
        )
    var y_trained = Tensor.alloc(B * ACT)
    _forward(trained, x, y_trained)
    save_params["cpu"](trained, ckpt, None)

    var loaded = BcNet[OBS, ACT].make["cpu", Kaiming](None)
    var y_fresh = Tensor.alloc(B * ACT)
    _forward(loaded, x, y_fresh)
    load_params["cpu"](loaded, ckpt, None)
    var y_loaded = Tensor.alloc(B * ACT)
    _forward(loaded, x, y_loaded)

    var exact = True
    for i in range(B * ACT):
        if y_loaded.data[i] != y_trained.data[i]:
            exact = False
    checks += 1
    if not exact:
        failures += 1
        print("  FAIL: the loaded network does not reproduce the saved one")
    else:
        print("  ok: saved == loaded, bit for bit, on", B * ACT, "outputs")

    # ── 2. the control ────────────────────────────────────────────────────
    var differs = False
    for i in range(B * ACT):
        if y_fresh.data[i] != y_trained.data[i]:
            differs = True
    checks += 1
    if not differs:
        failures += 1
        print("  FAIL: an UNLOADED network already matches — check 1 is"
              " vacuous")
    else:
        print("  ok: the control — an unloaded network disagrees")

    # ── 3. the normalisation round trip ───────────────────────────────────
    var mu = List[Float64]()
    var sd = List[Float64]()
    for j in range(OBS):
        mu.append(0.5 - 0.01 * Float64(j))
        sd.append(1.0 + 0.02 * Float64(j % 7))
    write_bc_norm(ckpt + ".norm", mu, sd, ACT)
    var norm = load_bc_norm(ckpt + ".norm", OBS, ACT)
    var same = norm.obs_dim() == OBS and norm.act_dim == ACT
    for j in range(OBS):
        if norm.mu[j] != mu[j] or norm.sd[j] != sd[j]:
            same = False
    var raw = List[Float64]()
    for j in range(OBS):
        raw.append(0.25 * Float64(j % 5) - 0.3)
    var z = List[Float64]()
    norm.apply(raw, z)
    var applied = len(z) == OBS
    for j in range(OBS):
        if z[j] != (raw[j] - mu[j]) / sd[j]:
            applied = False
    checks += 1
    if not (same and applied):
        failures += 1
        print("  FAIL: the normalisation did not round-trip")
    else:
        print("  ok: mu/sd round-trip and (x - mu) / sd is exact over", OBS,
              "words")

    # ── 4. the refusals ───────────────────────────────────────────────────
    var refused_width = False
    try:
        _ = load_bc_norm(ckpt + ".norm", OBS + 1, ACT)
    except:
        refused_width = True
    var refused_missing = False
    try:
        _ = load_bc_norm(String(TMP) + "/nothing.norm", OBS, ACT)
    except:
        refused_missing = True
    checks += 2
    if not refused_width:
        failures += 1
        print("  FAIL: a sidecar of the wrong width loaded")
    else:
        print("  ok: a sidecar whose width is not the model's is refused")
    if not refused_missing:
        failures += 1
        print("  FAIL: a missing sidecar did not raise")
    else:
        print("  ok: a missing sidecar is refused")

    print()
    print("--- ran", checks, "checks,", failures, "failed ---")
    if failures != 0:
        raise Error(
            "bc policy: " + String(failures) + " of " + String(checks)
            + " check(s) failed"
        )
    print("=== PASS ===")
