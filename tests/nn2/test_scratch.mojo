"""Test: Scratch[NAME, SIZE] + init_scratch_auto walker.

Phase 1.4 verification. Three sub-tests:

  1. **CPU walker init**: a tiny struct with 3 `Scratch` fields is
     `init_scratch_auto[..., "cpu"]`-walked; every field's `cpu` List
     reaches `SIZE` length and is zero-filled.

  2. **GPU walker init**: same struct, `init_scratch_auto[..., "gpu"]`-
     walked with a real `DeviceContext`; every field's `dev` Optional
     becomes populated and the buffer is the right size.

  3. **Mixed Param + Scratch fields**: a struct holding both `Param`
     fields and `Scratch` fields — the walker picks up ONLY the
     `Scratch`-typed ones (a sanity check that `conforms_to(_, IsScratch)`
     doesn't accidentally match `Param` fields).
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.param import Param


# ──────────────────────────────────────────────────────────────────────
# Test structs.
# ──────────────────────────────────────────────────────────────────────


struct _ThreeScratch(Movable & ImplicitlyDestructible):
    """Test struct with 3 differently-sized Scratch fields."""
    var alpha: Scratch["alpha", 4]
    var beta:  Scratch["beta",  8]
    var gamma: Scratch["gamma", 16]

    def __init__(out self):
        self.alpha = Scratch["alpha", 4]()
        self.beta  = Scratch["beta",  8]()
        self.gamma = Scratch["gamma", 16]()


struct _ParamAndScratch(Movable & ImplicitlyDestructible):
    """Test struct mixing Param + Scratch — walker must skip the Param."""
    var w: Param["w", True, 10]
    var s: Scratch["s", 6]
    var b: Param["b", False, 3]
    var t: Scratch["t", 12]

    def __init__(out self):
        self.w = Param["w", True, 10]()
        self.s = Scratch["s", 6]()
        self.b = Param["b", False, 3]()
        self.t = Scratch["t", 12]()


def test_cpu_walker() raises:
    print("test_cpu_walker ...")
    var x = _ThreeScratch()
    # Pre-walk: all empty.
    assert_true(len(x.alpha.cpu) == 0, "alpha.cpu empty before walker")
    assert_true(len(x.beta.cpu)  == 0, "beta.cpu empty before walker")
    assert_true(len(x.gamma.cpu) == 0, "gamma.cpu empty before walker")

    init_scratch_auto[_ThreeScratch, target="cpu"](x, None)

    # Post-walk: each list is SIZE long, all zeros.
    assert_true(len(x.alpha.cpu) == 4, "alpha.cpu length wrong")
    assert_true(len(x.beta.cpu)  == 8, "beta.cpu length wrong")
    assert_true(len(x.gamma.cpu) == 16, "gamma.cpu length wrong")
    for i in range(4):
        assert_true(x.alpha.cpu[i] == Scalar[DT](0), "alpha not zero-filled")
    for i in range(8):
        assert_true(x.beta.cpu[i] == Scalar[DT](0), "beta not zero-filled")
    for i in range(16):
        assert_true(x.gamma.cpu[i] == Scalar[DT](0), "gamma not zero-filled")

    # Names + sizes accessible via the IsScratch interface.
    assert_true(String(x.alpha.scratch_name()) == "alpha", "alpha name")
    assert_true(x.alpha.scratch_size() == 4, "alpha size")
    print("  ok")


def test_gpu_walker() raises:
    print("test_gpu_walker ...")
    var ctx = DeviceContext()
    var x = _ThreeScratch()
    assert_true(not x.alpha.dev, "alpha.dev None before walker")
    assert_true(not x.beta.dev,  "beta.dev None before walker")
    assert_true(not x.gamma.dev, "gamma.dev None before walker")

    init_scratch_auto[_ThreeScratch, target="gpu"](x, Optional[DeviceContext](ctx))

    assert_true(Bool(x.alpha.dev), "alpha.dev populated after walker")
    assert_true(Bool(x.beta.dev),  "beta.dev populated after walker")
    assert_true(Bool(x.gamma.dev), "gamma.dev populated after walker")
    # dev_ptr() should be non-null.
    var ap = x.alpha.dev_ptr()
    assert_true(Int(ap) != 0, "alpha dev_ptr() must be non-null")
    print("  ok")


def test_mixed_param_and_scratch() raises:
    """Walker only initialises Scratch fields, skips Param fields."""
    print("test_mixed_param_and_scratch ...")
    var x = _ParamAndScratch()
    # Pre-walk: Param fields are also empty (their factories haven't
    # been called) and the Scratches are empty.
    assert_true(len(x.s.cpu) == 0, "s.cpu empty before walker")
    assert_true(len(x.t.cpu) == 0, "t.cpu empty before walker")
    assert_true(len(x.w.value) == 0, "w.value empty before walker")

    init_scratch_auto[_ParamAndScratch, target="cpu"](x, None)

    # Scratch fields initialised.
    assert_true(len(x.s.cpu) == 6, "s.cpu length wrong post-walker")
    assert_true(len(x.t.cpu) == 12, "t.cpu length wrong post-walker")
    # Param fields untouched.
    assert_true(
        len(x.w.value) == 0,
        "w (Param) should NOT be touched by Scratch walker"
    )
    assert_true(
        len(x.b.value) == 0,
        "b (Param) should NOT be touched by Scratch walker"
    )
    print("  ok")


struct _StagingScratch(Movable & ImplicitlyDestructible):
    """Phase 2.5: a mix of plain and STAGING=True scratches. GPU init
    must populate BOTH cpu and dev for the staging one."""
    var plain: Scratch["plain", 4]
    var staging: Scratch["staging", 8, True]

    def __init__(out self):
        self.plain = Scratch["plain", 4]()
        self.staging = Scratch["staging", 8, True]()


def test_staging_scratch_gpu() raises:
    """STAGING=True: init_with["gpu"] must allocate dev AND mirror cpu."""
    print("test_staging_scratch_gpu ...")
    var ctx = DeviceContext()
    var x = _StagingScratch()

    init_scratch_auto[_StagingScratch, target="gpu"](
        x, Optional[DeviceContext](ctx),
    )

    # Plain: dev populated, cpu empty.
    assert_true(Bool(x.plain.dev), "plain.dev populated on GPU init")
    assert_true(len(x.plain.cpu) == 0, "plain.cpu must stay empty on GPU init")

    # Staging: BOTH dev and cpu populated.
    assert_true(Bool(x.staging.dev), "staging.dev populated on GPU init")
    assert_true(
        len(x.staging.cpu) == 8,
        "staging.cpu must be allocated alongside dev (STAGING=True)"
    )
    for i in range(8):
        assert_true(
            x.staging.cpu[i] == Scalar[DT](0),
            "staging.cpu must be zero-filled"
        )
    # cpu_ptr() should work on the staging scratch even on GPU runs.
    var cp = x.staging.cpu_ptr()
    assert_true(Int(cp) != 0, "staging cpu_ptr() must be non-null on GPU init")
    print("  ok")


def test_staging_scratch_cpu() raises:
    """STAGING=True: init_with["cpu"] still only allocates cpu (no GPU
    side-effect)."""
    print("test_staging_scratch_cpu ...")
    var x = _StagingScratch()

    init_scratch_auto[_StagingScratch, target="cpu"](x, None)

    # Both fields should have cpu allocated, neither with dev.
    assert_true(len(x.plain.cpu) == 4, "plain.cpu allocated on CPU init")
    assert_true(not x.plain.dev, "plain.dev must stay None on CPU init")
    assert_true(len(x.staging.cpu) == 8, "staging.cpu allocated on CPU init")
    assert_true(not x.staging.dev, "staging.dev must stay None on CPU init")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Scratch[NAME, SIZE] + init_scratch_auto walker (Phase 1.4 + 2.5)")
    print("=" * 70)
    test_cpu_walker()
    test_gpu_walker()
    test_mixed_param_and_scratch()
    test_staging_scratch_cpu()
    test_staging_scratch_gpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
