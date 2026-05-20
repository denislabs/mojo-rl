"""Phase 7E smoke gate for the Full Craftax pixel obs pipeline.

Verifies:
  - obs length == PIXEL_OBS_DIM (3 * 130 * 110 = 42 900)
  - all values in [0, 1] (atlas is normalized + we never multiply > 1)
  - obs is not entirely zero (at least the player tile must be drawn)
  - inventory bar bottom rows have some non-background pixels (HP/FD/DR
    icons visible after reset, since intrinsics start full)
  - obs changes after a movement step (player position shifts → view rolls)

Run:
  pixi run mojo run -I . tests/envs/craftax_full/test_pixel_obs.mojo
"""

from mojo_rl.envs.craftax_full import (
    CraftaxFullPixelEnv,
    CraftaxFullAction,
    OBS_PIX_H,
    OBS_PIX_W,
    OBS_CHANNELS,
    PIXEL_OBS_DIM,
)
from mojo_rl.envs.craftax_full.craftax_full_pixel import VIEW_PIX_H
from mojo_rl.envs.craftax_full.constants import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_DOWN,
)


@always_inline
def check(mut counts: List[Int], name: String, ok: Bool):
    if ok:
        counts[0] += 1
        print("  PASS", name)
    else:
        counts[1] += 1
        print("  FAIL", name)


def test_obs_shape(mut counts: List[Int]) raises:
    print("test_obs_shape")
    check(counts, "OBS_PIX_H == 130", OBS_PIX_H == 130)
    check(counts, "OBS_PIX_W == 110", OBS_PIX_W == 110)
    check(counts, "OBS_CHANNELS == 3", OBS_CHANNELS == 3)
    check(counts, "PIXEL_OBS_DIM == 42900", PIXEL_OBS_DIM == 42900)


def test_obs_basic(mut counts: List[Int]) raises:
    print("test_obs_basic")
    var env = CraftaxFullPixelEnv()
    _ = env.reset_with_seed(UInt64(0xBEEF))
    var obs = env.get_obs_list()
    check(counts, "len(obs) == PIXEL_OBS_DIM", len(obs) == PIXEL_OBS_DIM)

    var all_in_range = True
    var any_nonzero = False
    var max_val = Float32(0.0)
    var min_val = Float32(1.0)
    for i in range(len(obs)):
        var v = Float32(obs[i])
        if v < Float32(0.0) or v > Float32(1.0):
            all_in_range = False
        if v > max_val:
            max_val = v
        if v < min_val:
            min_val = v
        if v != Float32(0.0):
            any_nonzero = True
    check(counts, "all pixels in [0,1]", all_in_range)
    check(counts, "obs has some non-zero pixels", any_nonzero)
    check(counts, "obs has bright pixels (max > 0.3)",
          max_val > Float32(0.3))


def test_inventory_bar_visible(mut counts: List[Int]) raises:
    """After reset, HP/FD/DR/EN/MN intrinsic icons should be drawn in
    the inventory row 0. The first BPS rows of the inventory bar should
    contain at least one bright pixel."""
    print("test_inventory_bar_visible")
    var env = CraftaxFullPixelEnv()
    _ = env.reset_with_seed(UInt64(42))
    var obs = env.get_obs_list()
    # Inventory bar starts at h == VIEW_PIX_H. Sample channel 0 of the
    # first inventory pixel-row.
    var found_bright = False
    comptime HW = OBS_PIX_H * OBS_PIX_W
    for w in range(OBS_PIX_W):
        var idx = 0 * HW + VIEW_PIX_H * OBS_PIX_W + w
        if Float32(obs[idx]) > Float32(0.3):
            found_bright = True
            break
    check(counts, "inventory bar has a drawn icon", found_bright)


def test_obs_changes_after_move(mut counts: List[Int]) raises:
    print("test_obs_changes_after_move")
    var env = CraftaxFullPixelEnv()
    _ = env.reset_with_seed(UInt64(2026))
    var obs0 = env.get_obs_list()
    var changed = False
    var actions = [ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN]
    for a in actions:
        _ = env.reset_with_seed(UInt64(2026))
        _ = env.step(CraftaxFullAction(value=a))
        var obs1 = env.get_obs_list()
        # Compare just the view portion — that's where movement shows up.
        comptime HW = OBS_PIX_H * OBS_PIX_W
        for h in range(VIEW_PIX_H):
            for w in range(OBS_PIX_W):
                var idx = 0 * HW + h * OBS_PIX_W + w
                if Float32(obs1[idx]) != Float32(obs0[idx]):
                    changed = True
                    break
            if changed:
                break
        if changed:
            break
    check(counts, "obs changes after a movement step", changed)


def main() raises:
    print("Craftax-Full Phase-7E pixel obs smoke gate")
    print("=" * 50)
    var counts = [0, 0]
    test_obs_shape(counts)
    test_obs_basic(counts)
    test_inventory_bar_visible(counts)
    test_obs_changes_after_move(counts)
    print()
    print("=" * 50)
    print("Passed:", counts[0], "Failed:", counts[1])
    if counts[1] > 0:
        raise Error("Phase-7E pixel obs gate FAILED")
    print("Phase-7E pixel obs gate PASS")
