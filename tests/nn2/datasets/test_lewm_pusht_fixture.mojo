"""LewmPushTExpert loader test against the synthetic FFI fixture.

The fixture has the same on-disk layout as the real ``pusht_expert_train.h5``
(see ``tests/io/hdf5/make_fixture.py``) so we can exercise the loader
end-to-end without downloading 13 GB. The deterministic generator lets
us assert exact values.

Setup:
    pixi run python tests/io/hdf5/make_fixture.py
    pixi run mojo run -I . tests/nn/datasets/test_lewm_pusht_fixture.mojo

Fixture layout (must match make_fixture.py):
  ep_lengths     = [4, 3, 5]
  ep_offsets     = [0, 4, 7]
  n_total_frames = 12, pixels (12, 8, 6, 3) uint8
  pixels[t,:,:,:] = (t * 7) % 256
  action[t]       = [t, t + 0.5]
  proprio[t]      = [t * 0.1, t * 0.2]
  state[t]        = [t, 2t, 3t, 4t, 5t]
"""

from std.testing import assert_equal, assert_true
from mojo_rl.nn2.datasets import LewmPushTExpert, LewmPushTWindow


comptime FIXTURE_PATH = "/tmp/mojo_rl_hdf5_fixture.h5"

# Mirror make_fixture.py
comptime N_EP: Int = 3
comptime N_TOTAL: Int = 12
comptime H: Int = 8
comptime W: Int = 6
comptime ACTION_DIM: Int = 2
comptime PROPRIO_DIM: Int = 2
comptime STATE_DIM: Int = 5


def test_shape_introspection() raises:
    print("[test] shape introspection (frameskip=1, num_steps=1)...")
    var ds = LewmPushTExpert(
        frameskip=1, num_steps=1, path=String(FIXTURE_PATH)
    )
    assert_equal(ds.n_episodes, N_EP, "n_episodes")
    assert_equal(ds.n_total_frames, N_TOTAL, "n_total_frames")
    assert_equal(ds.pixel_h, H, "pixel_h")
    assert_equal(ds.pixel_w, W, "pixel_w")
    assert_equal(ds.action_dim, ACTION_DIM, "action_dim")
    assert_equal(ds.proprio_dim, PROPRIO_DIM, "proprio_dim")
    assert_equal(ds.state_dim, STATE_DIM, "state_dim")
    # ep_len / ep_offset round-trip
    var ep_lens: List[Int] = [4, 3, 5]
    var ep_offsets: List[Int] = [0, 4, 7]
    for ep in range(N_EP):
        assert_equal(Int(ds.ep_len[ep]), ep_lens[ep], "ep_len match")
        assert_equal(Int(ds.ep_offset[ep]), ep_offsets[ep], "ep_offset match")
    print("       OK")


def test_clip_indices_frameskip1_steps2() raises:
    """span = num_steps * frameskip = 2 → for each ep length L,
    valid starts ∈ [0, L-2], i.e. (L-1) clips."""
    print("[test] clip indices (frameskip=1, num_steps=2)...")
    var ds = LewmPushTExpert(
        frameskip=1, num_steps=2, path=String(FIXTURE_PATH)
    )
    # ep lens [4,3,5] → valid clip counts [3, 2, 4] → total 9
    assert_equal(len(ds), 9, "total clip count")

    # Spot-check the (ep_idx, start) sequence:
    # ep0: starts 0,1,2 ; ep1: starts 0,1 ; ep2: starts 0,1,2,3
    var expected_ep: List[Int] = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    var expected_st: List[Int] = [0, 1, 2, 0, 1, 0, 1, 2, 3]
    for i in range(9):
        assert_equal(
            Int(ds.clip_ep_idx[i]), expected_ep[i], "ep idx at i"
        )
        assert_equal(
            Int(ds.clip_start[i]), expected_st[i], "start at i"
        )
    print("       OK")


def test_sample_window_first_clip() raises:
    """frameskip=1, num_steps=2, clip 0 → ep_idx=0, start=0, g_start=0.
    Window covers frames {0, 1}."""
    print("[test] sample_window: first clip, frameskip=1, num_steps=2...")
    var ds = LewmPushTExpert(
        frameskip=1, num_steps=2, path=String(FIXTURE_PATH)
    )
    var win = ds.make_window()
    ds.sample_window(0, win)

    # pixels[t,:,:,:] = (t*7) % 256, in CHW after permute → every value at
    # step n equals UInt8((n*7) % 256) (since plane is uniform over c,h,w).
    var per_step = 3 * H * W
    for n in range(2):
        var expected = UInt8((n * 7) % 256)
        for i in range(per_step):
            assert_equal(
                win.pixels[n * per_step + i],
                expected,
                "pixels[n,:,:,:]",
            )

    # action[t] = [t, t+0.5], frameskip=1 → window covers t∈{0,1}.
    # Output shape: (num_steps=2, frameskip*action_dim = 1*2 = 2)
    # So action[0,:] = action_flat at t=0 = [0, 0.5]
    # And action[1,:] = action_flat at t=1 = [1, 1.5]
    assert_equal(Float64(win.action[0]), 0.0, "action win[0,0]")
    assert_equal(Float64(win.action[1]), 0.5, "action win[0,1]")
    assert_equal(Float64(win.action[2]), 1.0, "action win[1,0]")
    assert_equal(Float64(win.action[3]), 1.5, "action win[1,1]")

    # proprio[t] = [t*0.1, t*0.2], subsampled by frameskip=1.
    # Float32 won't represent 0.1/0.2 exactly — compare with epsilon.
    assert_equal(Float64(win.proprio[0]), 0.0, "proprio[0,0]")
    assert_equal(Float64(win.proprio[1]), 0.0, "proprio[0,1]")
    assert_true(
        abs(Float64(win.proprio[2]) - 0.1) < 1e-6, "proprio[1,0]"
    )
    assert_true(
        abs(Float64(win.proprio[3]) - 0.2) < 1e-6, "proprio[1,1]"
    )

    # state[t] = [t, 2t, 3t, 4t, 5t]
    for j in range(STATE_DIM):
        assert_equal(
            Float64(win.state[j]), 0.0, "state[0,j]"
        )  # all zero at t=0
    for j in range(STATE_DIM):
        assert_equal(
            Float64(win.state[STATE_DIM + j]),
            Float64(j + 1),
            "state[1,j]",
        )
    print("       OK")


def test_sample_window_frameskip2_steps2() raises:
    """frameskip=2, num_steps=2, span=4. ep0 (len 4) → 1 clip with
    start=0. Window samples non-action at indices {0, 2}; action covers
    {0, 1, 2, 3} reshaped to (2, 4)."""
    print("[test] sample_window: frameskip=2, num_steps=2 on ep0...")
    var ds = LewmPushTExpert(
        frameskip=2, num_steps=2, path=String(FIXTURE_PATH)
    )
    # ep lens [4,3,5], span=4 → clip counts: ep0=1 (L-span+1=1), ep1=0, ep2=2
    assert_equal(len(ds), 3, "total clips with span=4")
    assert_equal(Int(ds.clip_ep_idx[0]), 0, "first clip ep")
    assert_equal(Int(ds.clip_start[0]), 0, "first clip start")

    var win = ds.make_window()
    ds.sample_window(0, win)

    # pixels: non-action subsampled by frameskip=2 → {t=0, t=2}
    var per_step = 3 * H * W
    for n in range(2):
        var t = n * 2
        var expected = UInt8((t * 7) % 256)
        for i in range(per_step):
            assert_equal(
                win.pixels[n * per_step + i], expected, "pixels CHW"
            )

    # action: DENSE span=4 → flat (span * action_dim = 8) values.
    # Shape (num_steps=2, frameskip*action_dim=4): row 0 = t∈{0,1}, row 1 = t∈{2,3}.
    # action[t] = [t, t+0.5]
    var expected_action: List[Float64] = [
        0.0, 0.5, 1.0, 1.5,   # row 0: t=0, t=1
        2.0, 2.5, 3.0, 3.5,   # row 1: t=2, t=3
    ]
    for i in range(8):
        assert_equal(
            Float64(win.action[i]), expected_action[i], "action row-major"
        )

    # state: subsampled → t∈{0, 2}. state[t] = [t, 2t, 3t, 4t, 5t]
    for j in range(STATE_DIM):
        assert_equal(Float64(win.state[j]), 0.0, "state[0,j] at t=0")
        assert_equal(
            Float64(win.state[STATE_DIM + j]),
            Float64((j + 1) * 2),
            "state[1,j] at t=2",
        )
    print("       OK")


def main() raises:
    test_shape_introspection()
    test_clip_indices_frameskip1_steps2()
    test_sample_window_first_clip()
    test_sample_window_frameskip2_steps2()
    print("[lewm_pusht loader fixture test] all passing.")
