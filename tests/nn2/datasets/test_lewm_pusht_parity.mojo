"""Byte-for-byte parity test: ``LewmPushTExpert.sample_window`` against the
Python reference ``lewm_pusht_python_ref.py`` (which replicates
``stable_worldmodel.data.dataset.Dataset.__getitem__`` using only
``h5py`` + ``numpy``).

Uses the synthetic fixture at ``/tmp/mojo_rl_hdf5_fixture.h5`` so we do
not have to download 13 GB to validate equivalence. The fixture shares
the same on-disk schema as the real ``pusht_expert_train.h5``.

Setup:
    pixi run python tests/io/hdf5/make_fixture.py
    pixi run mojo run -I . tests/nn/datasets/test_lewm_pusht_parity.mojo
"""

from std.python import Python, PythonObject
from std.testing import assert_equal, assert_true
from mojo_rl.nn2.datasets import LewmPushTExpert, LewmPushTWindow


comptime FIXTURE_PATH = "/tmp/mojo_rl_hdf5_fixture.h5"


def _import_ref() raises -> PythonObject:
    """Load tests/nn/datasets/lewm_pusht_python_ref.py via sys.path."""
    var sys = Python.import_module("sys")
    _ = sys.path.insert(
        PythonObject(0), PythonObject(String("tests/nn/datasets"))
    )
    return Python.import_module("lewm_pusht_python_ref")


def _check_window_parity(
    ds: LewmPushTExpert,
    mut win: LewmPushTWindow,
    py_ref: PythonObject,
    clip_idx: Int,
    label: String,
) raises:
    """Compare Mojo sample_window output against the Python reference."""
    ds.sample_window(clip_idx, win)
    var py_out = py_ref.sample_window(
        PythonObject(String(FIXTURE_PATH)),
        PythonObject(clip_idx),
        PythonObject(ds.frameskip),
        PythonObject(ds.num_steps),
    )

    # ── pixels (uint8) ───────────────────────────────────────────────
    var py_pixels = py_out["pixels"]
    var n_pixels = ds.num_steps * 3 * ds.pixel_h * ds.pixel_w
    var pixels_flat = py_pixels.reshape(-1)
    for i in range(n_pixels):
        var mojo_val = Int(win.pixels[i])
        var py_val = Int(py=pixels_flat[i])
        if mojo_val != py_val:
            raise Error(
                label + ": pixels mismatch at flat idx "
                + String(i) + " (mojo=" + String(mojo_val)
                + " py_ref=" + String(py_val) + ")"
            )

    # ── action (f32) ─────────────────────────────────────────────────
    var py_action = py_out["action"]
    var act_total = ds.num_steps * ds.frameskip * ds.action_dim
    var action_flat = py_action.reshape(-1)
    for i in range(act_total):
        var mojo_val = Float64(win.action[i])
        var py_val = Float64(py=action_flat[i])
        if abs(mojo_val - py_val) > 1e-6:
            raise Error(
                label + ": action mismatch at flat idx "
                + String(i) + " (mojo=" + String(mojo_val)
                + " py_ref=" + String(py_val) + ")"
            )

    # ── proprio (f32) ────────────────────────────────────────────────
    var py_proprio = py_out["proprio"]
    var pro_total = ds.num_steps * ds.proprio_dim
    var proprio_flat = py_proprio.reshape(-1)
    for i in range(pro_total):
        var mojo_val = Float64(win.proprio[i])
        var py_val = Float64(py=proprio_flat[i])
        if abs(mojo_val - py_val) > 1e-6:
            raise Error(
                label + ": proprio mismatch at idx "
                + String(i) + " (mojo=" + String(mojo_val)
                + " py_ref=" + String(py_val) + ")"
            )

    # ── state (f32) ──────────────────────────────────────────────────
    var py_state = py_out["state"]
    var st_total = ds.num_steps * ds.state_dim
    var state_flat = py_state.reshape(-1)
    for i in range(st_total):
        var mojo_val = Float64(win.state[i])
        var py_val = Float64(py=state_flat[i])
        if abs(mojo_val - py_val) > 1e-6:
            raise Error(
                label + ": state mismatch at idx "
                + String(i) + " (mojo=" + String(mojo_val)
                + " py_ref=" + String(py_val) + ")"
            )


def test_parity_fs1_ns1() raises:
    print("[parity] frameskip=1, num_steps=1 ...")
    var py_ref = _import_ref()
    var ds = LewmPushTExpert(
        frameskip=1, num_steps=1, path=String(FIXTURE_PATH)
    )
    var win = ds.make_window()
    var n_clips = Int(
        py=py_ref.num_clips(
            PythonObject(String(FIXTURE_PATH)),
            PythonObject(1),
            PythonObject(1),
        )
    )
    assert_equal(len(ds), n_clips, "clip count parity")

    # Spot-check first / mid / last
    var probes: List[Int] = [0, n_clips // 2, n_clips - 1]
    for i in range(len(probes)):
        _check_window_parity(
            ds, win, py_ref, probes[i], String("fs=1 ns=1 clip=") + String(probes[i])
        )
    print("       OK (", n_clips, " clips, 3 probed)")


def test_parity_fs1_ns2() raises:
    print("[parity] frameskip=1, num_steps=2 ...")
    var py_ref = _import_ref()
    var ds = LewmPushTExpert(
        frameskip=1, num_steps=2, path=String(FIXTURE_PATH)
    )
    var win = ds.make_window()
    var n_clips = Int(
        py=py_ref.num_clips(
            PythonObject(String(FIXTURE_PATH)),
            PythonObject(1),
            PythonObject(2),
        )
    )
    assert_equal(len(ds), n_clips, "clip count parity")

    # Test every clip — the fixture is small so this is cheap.
    for i in range(n_clips):
        _check_window_parity(
            ds, win, py_ref, i, String("fs=1 ns=2 clip=") + String(i)
        )
    print("       OK (", n_clips, " clips, all probed)")


def test_parity_fs2_ns2() raises:
    print("[parity] frameskip=2, num_steps=2 ...")
    var py_ref = _import_ref()
    var ds = LewmPushTExpert(
        frameskip=2, num_steps=2, path=String(FIXTURE_PATH)
    )
    var win = ds.make_window()
    var n_clips = Int(
        py=py_ref.num_clips(
            PythonObject(String(FIXTURE_PATH)),
            PythonObject(2),
            PythonObject(2),
        )
    )
    assert_equal(len(ds), n_clips, "clip count parity")
    for i in range(n_clips):
        _check_window_parity(
            ds, win, py_ref, i, String("fs=2 ns=2 clip=") + String(i)
        )
    print("       OK (", n_clips, " clips, all probed)")


def test_parity_fs3_ns1() raises:
    """Stride > 1 with single step — exercises strided pixel reads."""
    print("[parity] frameskip=3, num_steps=1 ...")
    var py_ref = _import_ref()
    var ds = LewmPushTExpert(
        frameskip=3, num_steps=1, path=String(FIXTURE_PATH)
    )
    var win = ds.make_window()
    var n_clips = Int(
        py=py_ref.num_clips(
            PythonObject(String(FIXTURE_PATH)),
            PythonObject(3),
            PythonObject(1),
        )
    )
    assert_equal(len(ds), n_clips, "clip count parity")
    for i in range(n_clips):
        _check_window_parity(
            ds, win, py_ref, i, String("fs=3 ns=1 clip=") + String(i)
        )
    print("       OK (", n_clips, " clips, all probed)")


def main() raises:
    test_parity_fs1_ns1()
    test_parity_fs1_ns2()
    test_parity_fs2_ns2()
    test_parity_fs3_ns1()
    print("[lewm_pusht parity test] all configs match Python reference.")
