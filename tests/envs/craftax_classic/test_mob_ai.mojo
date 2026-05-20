"""Phase-3B mob AI gate: zombie chase + attack, cow random walk, skeleton
arrow fire, arrow physics, natural spawn.

Each subtest pins the player position, hand-places mobs into specific
slots, then runs `env.step()` for one or more steps and asserts the
state evolved correctly.

Run:
  pixi run mojo run -I . tests/envs/craftax_classic/test_mob_ai.mojo
"""

from std.random.philox import Random as PhiloxRandom

from mojo_rl.envs.craftax_classic import CraftaxClassicEnv
from mojo_rl.envs.craftax_classic.constants import (
    MAP_W,
    BLOCK_GRASS,
    BLOCK_PATH,
    DIR_UP,
    ACTION_NOOP,
    ACTION_DOWN,
    ACH_WAKE_UP,
    INTRINSIC_HEALTH,
    INTRINSIC_ENERGY,
    MOB_FY,
    MOB_FX,
    MOB_HP,
    MOB_CD,
    MOB_FIELDS,
    ARROW_FIELDS,
    ARROW_FDIR,
    MAX_ZOMBIES,
    MAX_COWS,
    MAX_SKELETONS,
    MAX_ARROWS,
    MOB_DESPAWN_DISTANCE,
)
from mojo_rl.envs.craftax_classic.state import (
    S_MAP_BASE,
    S_PLAYER_POS,
    S_INTRINSICS_BASE,
    S_ZOMBIES_BASE,
    S_COWS_BASE,
    S_SKELETONS_BASE,
    S_ARROWS_BASE,
    S_ACHIEVEMENTS_BASE,
    S_IS_SLEEPING,
    S_LIGHT_LEVEL,
    STATE_SIZE,
)
from mojo_rl.nn import dtype


@always_inline
def setup_clear_env(mut env: CraftaxClassicEnv[dtype], seed: UInt64 = 42):
    """Reset env and clear a 21×21 grass region around the player so we
    have a known-walkable arena to test mob AI on."""
    _ = env.reset_with_seed(seed, False)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    for dy in range(-10, 11):
        for dx in range(-10, 11):
            var y = py + dy
            var x = px + dx
            if 0 <= y and y < 64 and 0 <= x and x < 64:
                env.state[S_MAP_BASE + y * MAP_W + x] = Float32(BLOCK_GRASS)


@always_inline
def place_mob_at(
    mut env: CraftaxClassicEnv[dtype],
    base: Int,
    slot: Int,
    y: Int,
    x: Int,
    hp: Int,
    cd: Int = 0,
):
    env.state[base + slot * MOB_FIELDS + MOB_FY] = Float32(y)
    env.state[base + slot * MOB_FIELDS + MOB_FX] = Float32(x)
    env.state[base + slot * MOB_FIELDS + MOB_HP] = Float32(hp)
    env.state[base + slot * MOB_FIELDS + MOB_CD] = Float32(cd)


@always_inline
def place_arrow_at(
    mut env: CraftaxClassicEnv[dtype],
    slot: Int,
    y: Int,
    x: Int,
    dir_code: Int,
):
    var base = S_ARROWS_BASE + slot * ARROW_FIELDS
    env.state[base + MOB_FY] = Float32(y)
    env.state[base + MOB_FX] = Float32(x)
    env.state[base + MOB_HP] = Float32(1)
    env.state[base + MOB_CD] = Float32(0)
    env.state[base + ARROW_FDIR] = Float32(dir_code)


@always_inline
def check(mut counts: List[Int], name: String, condition: Bool):
    if condition:
        counts[0] += 1
        print("  PASS", name)
    else:
        counts[1] += 1
        print("  FAIL", name)


# ----------------------------------------------------------------------------


def test_zombie_chases_player(mut counts: List[Int]) raises:
    """Zombie placed 3 tiles south should close the gap over a few steps.

    Reference 75% chase probability; over 20 steps, probability of never
    chasing once is (0.25)^20 ≈ 10^-12. So a distance reduction is
    near-certain.
    """
    print("test_zombie_chases_player")
    var env = CraftaxClassicEnv[dtype]()
    setup_clear_env(env)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    place_mob_at(env, S_ZOMBIES_BASE, 0, py + 3, px, 5)
    var d_initial = 3

    var d_min = d_initial
    for _ in range(20):
        _ = env.step_obs(ACTION_NOOP)
        var zy = Int(env.state[S_ZOMBIES_BASE + MOB_FY])
        var zx = Int(env.state[S_ZOMBIES_BASE + MOB_FX])
        if Int(env.state[S_ZOMBIES_BASE + MOB_HP]) <= 0:
            break
        var d = (zy - py if zy >= py else py - zy) + (
            zx - px if zx >= px else px - zx
        )
        if d < d_min:
            d_min = d
    check(counts, "zombie closed distance", d_min < d_initial)


def test_zombie_attacks_player(mut counts: List[Int]) raises:
    print("test_zombie_attacks_player")
    var env = CraftaxClassicEnv[dtype]()
    setup_clear_env(env)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    # Zombie adjacent (distance 1) with cooldown 0 → attacks this step.
    place_mob_at(env, S_ZOMBIES_BASE, 0, py + 1, px, 5, cd=0)
    var h0 = Int(env.state[S_INTRINSICS_BASE + INTRINSIC_HEALTH])
    _ = env.step_obs(ACTION_NOOP)
    var h1 = Int(env.state[S_INTRINSICS_BASE + INTRINSIC_HEALTH])
    check(counts, "player took zombie damage", h1 < h0)


def test_zombie_despawns_far_away(mut counts: List[Int]) raises:
    print("test_zombie_despawns_far_away")
    var env = CraftaxClassicEnv[dtype]()
    setup_clear_env(env)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    place_mob_at(env, S_ZOMBIES_BASE, 0, py + 20, px, 5)  # dist=20 ≥ 14
    _ = env.step_obs(ACTION_NOOP)
    var hp = Int(env.state[S_ZOMBIES_BASE + MOB_HP])
    check(counts, "zombie despawned (hp=0)", hp == 0)


def test_cow_random_walk(mut counts: List[Int]) raises:
    """Cow has 50% chance of staying still each step (4/8 directions are
    zero), so over 30 steps it will almost surely have moved at least
    once: P(stay still 30×) ≈ 9.3 × 10^-10.
    """
    print("test_cow_random_walk")
    var env = CraftaxClassicEnv[dtype]()
    setup_clear_env(env)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    var cy0 = py + 4
    var cx0 = px + 4
    place_mob_at(env, S_COWS_BASE, 0, cy0, cx0, 3)
    var moved = False
    for _ in range(30):
        _ = env.step_obs(ACTION_NOOP)
        var cy = Int(env.state[S_COWS_BASE + MOB_FY])
        var cx = Int(env.state[S_COWS_BASE + MOB_FX])
        if cy != cy0 or cx != cx0:
            moved = True
            break
    check(counts, "cow moved at least once", moved)


def test_skeleton_fires_arrow(mut counts: List[Int]) raises:
    """Skeleton in the firing band (dist 4..5) should spawn an arrow."""
    print("test_skeleton_fires_arrow")
    var env = CraftaxClassicEnv[dtype]()
    setup_clear_env(env)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    # Distance 5 along Y (clear grass arena).
    place_mob_at(env, S_SKELETONS_BASE, 0, py + 5, px, 3, cd=0)

    # Run a few steps — skeleton may pick random move instead of fire
    # ~15% of the time, so loop until we either see an arrow or
    # exhaust patience.
    var arrow_fired = False
    for _ in range(20):
        _ = env.step_obs(ACTION_NOOP)
        for i in range(MAX_ARROWS):
            var base = S_ARROWS_BASE + i * ARROW_FIELDS
            if Int(env.state[base + MOB_HP]) > 0:
                arrow_fired = True
                break
        if arrow_fired:
            break
    check(counts, "skeleton fired at least one arrow", arrow_fired)


def test_arrow_hits_player(mut counts: List[Int]) raises:
    """Manually placed arrow heading UP from (py+1, px) should hit the
    player at (py, px) on the next step."""
    print("test_arrow_hits_player")
    var env = CraftaxClassicEnv[dtype]()
    setup_clear_env(env)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    place_arrow_at(env, 0, py + 1, px, DIR_UP)
    var h0 = Int(env.state[S_INTRINSICS_BASE + INTRINSIC_HEALTH])
    _ = env.step_obs(ACTION_NOOP)
    var h1 = Int(env.state[S_INTRINSICS_BASE + INTRINSIC_HEALTH])
    check(counts, "player hit by arrow", h1 < h0)
    check(
        counts,
        "arrow consumed",
        Int(env.state[S_ARROWS_BASE + MOB_HP]) == 0,
    )


def test_natural_spawn(mut counts: List[Int]) raises:
    """Cows have ~10% spawn chance per step. Run 200 steps and expect at
    least one cow to appear (P_none ≈ 0.9^200 ≈ 7×10^-10)."""
    print("test_natural_spawn")
    var env = CraftaxClassicEnv[dtype]()
    setup_clear_env(env)
    # Force-empty all mob arrays (reset_with_seed already does this, but
    # explicit is safer).
    for i in range(MAX_COWS):
        env.state[S_COWS_BASE + i * MOB_FIELDS + MOB_HP] = Float32(0)
    var spawned = False
    for _ in range(200):
        _ = env.step_obs(ACTION_NOOP)
        for i in range(MAX_COWS):
            if Int(env.state[S_COWS_BASE + i * MOB_FIELDS + MOB_HP]) > 0:
                spawned = True
                break
        if spawned:
            break
    check(counts, "cow spawned naturally", spawned)


def test_zombie_wakes_sleeping_player(mut counts: List[Int]) raises:
    """An adjacent zombie attacks a sleeping player with 7 damage and
    wakes them — sets WAKE_UP achievement."""
    print("test_zombie_wakes_sleeping_player")
    var env = CraftaxClassicEnv[dtype]()
    setup_clear_env(env)
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    env.state[S_IS_SLEEPING] = Float32(1.0)
    env.state[S_INTRINSICS_BASE + INTRINSIC_ENERGY] = Float32(2)
    place_mob_at(env, S_ZOMBIES_BASE, 0, py + 1, px, 5, cd=0)
    _ = env.step_obs(ACTION_NOOP)
    var awake = env.state[S_IS_SLEEPING] < Float32(0.5)
    var ach = env.state[S_ACHIEVEMENTS_BASE + ACH_WAKE_UP] > Float32(0.5)
    check(counts, "is_sleeping cleared by zombie attack", awake)
    check(counts, "WAKE_UP set", ach)


def main() raises:
    print("Craftax-Classic Phase-3B mob AI gate")
    print("=" * 50)
    var counts = [0, 0]

    test_zombie_chases_player(counts)
    test_zombie_attacks_player(counts)
    test_zombie_despawns_far_away(counts)
    test_cow_random_walk(counts)
    test_skeleton_fires_arrow(counts)
    test_arrow_hits_player(counts)
    test_natural_spawn(counts)
    test_zombie_wakes_sleeping_player(counts)

    print()
    print("=" * 50)
    print("Passed:", counts[0])
    print("Failed:", counts[1])
    if counts[1] > 0:
        raise Error("Phase-3B gate FAILED")
    print("Phase-3B gate PASS")
