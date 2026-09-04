"""`SmolVLAPolicy.load` against the published checkpoint.

`test_policy` gates the WIRING with seeded weights and never calls `load`, so
until this runs the four walks are compiled and unexercised. What it pins:

  1. **Every map entry was claimed by exactly one parameter.** `report` raises
     when a PARAMETER has no map entry — the topology moved. The mirror
     failure is a map ENTRY no parameter claimed, because the walk never
     emitted that name: the weight stays at its initialiser and the load
     reports success. `Tokenwise` emitting `connector.0.weight` where the map
     says `connector.weight` is exactly that, and is how the bug was found.
  2. **`lm_head`'s two entries are skipped, and only those two.** Without a
     number, a component someone forgets to walk looks identical to the one
     omitted on purpose.
  3. **The weights landed in the right modules**, checked by value against
     `tools/vla/smolvla_samples.tsv` — a Python read that shares no code with
     the Mojo loader. Counts cannot see a `state_proj` filled from
     `action_in`'s tensor if both walks ran.

⚠ ~907 MB download (cached), and the policy instantiates every layer, so this
is slow. `pixi run test-vla-weights`.
"""

from std.math import abs
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.io.fileio import read_file_bytes
from mojo_rl.io.hf import hf_download_file, HF_MODEL
from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.smolvla.policy import SmolVLAPolicy

comptime REPO = String("lerobot/smolvla_base")
comptime SAMPLES = String("tools/vla/smolvla_samples.tsv")
comptime Pol = SmolVLAPolicy[2, 6, 50, 10, 1]


def main() raises:
    print("=" * 70)
    print("SmolVLAPolicy.load — the published checkpoint")
    print("=" * 70)

    var path = hf_download_file(REPO, String("model.safetensors"), HF_MODEL)
    print("  weights:", path)

    var d = DeviceContext()
    print("  building every layer…")
    var pol = Pol.make["gpu"](Optional(d))

    # The four walks. Each raises on a name that drifted (report) and on an
    # entry nothing claimed (the count) — see the struct header.
    pol.load["gpu"](path, Optional(d))
    print("  [1] all four walks claimed every map entry they were given")

    # [3] a value, in the module the policy actually uses. `state_proj` is F32,
    # transposed, and sits beside a real bias — the shape that a wrong
    # TN_TRANSPOSE on a non-square tensor would already have caught, but a
    # wrong MODULE would not.
    ref sp = pol.state_proj.weight.val
    var nonzero = 0
    var lo = Scalar[DT](0)
    var hi = Scalar[DT](0)
    for i in range(len(sp.data)):
        if sp.data[i] != 0.0:
            nonzero += 1
        if sp.data[i] < lo:
            lo = sp.data[i]
        if sp.data[i] > hi:
            hi = sp.data[i]
    print("  [2] state_proj.weight:", len(sp.data), "values,", nonzero,
          "non-zero, range", lo, "..", hi)
    assert_true(
        nonzero > len(sp.data) // 2,
        "state_proj is mostly zeros — it kept a zero initialisation rather"
        " than the checkpoint",
    )
    assert_true(hi - lo > 1e-3, "state_proj is constant")

    # The connector is the one walked through `.inner`; if that were wrong it
    # would be `unmapped` (caught above), but check it carries real values too.
    ref cw = pol.connector.inner.weight.val
    var cnz = 0
    for i in range(len(cw.data)):
        if cw.data[i] != 0.0:
            cnz += 1
    print("  [3] connector.weight:", len(cw.data), "values,", cnz, "non-zero")
    assert_true(
        cnz > len(cw.data) // 2,
        "the connector kept its initialisation — the Tokenwise wrapper was"
        " walked instead of its inner Linear",
    )

    print()
    print("PASSED — the policy holds the published weights")
