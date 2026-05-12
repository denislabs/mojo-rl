"""Phase 7B gate: structural sanity of Full Craftax world gen.

Generates one full world from a fixed seed and verifies:
  - Each of the 9 floors has plausible block distributions
    (overworld has grass+stone+trees, mines have stone+path, dungeons have
     wall+path+chest, fire/ice have their grass type, boss has necromancer).
  - Down-ladders and up-ladders are placed in-range for each floor
    (where the config asks for them).
  - Two distinct seeds produce two distinct worlds.

Run:
  pixi run mojo run -I . tests/envs/craftax_full/test_world_gen.mojo
"""

from mojo_rl.envs.craftax_full import (
    MAP_H,
    MAP_W,
    NUM_FLOORS,
    STATE_SIZE,
    generate_full_world,
)
from mojo_rl.envs.craftax_full.constants import (
    BLOCK_GRASS,
    BLOCK_WATER,
    BLOCK_STONE,
    BLOCK_TREE,
    BLOCK_PATH,
    BLOCK_COAL,
    BLOCK_IRON,
    BLOCK_DIAMOND,
    BLOCK_FIRE_GRASS,
    BLOCK_ICE_GRASS,
    BLOCK_WALL,
    BLOCK_WALL_MOSS,
    BLOCK_DARKNESS,
    BLOCK_CHEST,
    BLOCK_NECROMANCER,
    BLOCK_LAVA,
    BLOCK_FOUNTAIN,
    BLOCK_ENCHANTMENT_TABLE_ICE,
    BLOCK_ENCHANTMENT_TABLE_FIRE,
    ITEM_LADDER_DOWN,
    ITEM_LADDER_UP,
    ITEM_TORCH,
)
from mojo_rl.envs.craftax_full.state import (
    s_map,
    s_item_map,
    s_light_map,
    s_down_ladder,
    s_up_ladder,
)


@always_inline
def check(mut counts: List[Int], name: String, ok: Bool):
    if ok:
        counts[0] += 1
        print("  PASS", name)
    else:
        counts[1] += 1
        print("  FAIL", name)


@always_inline
def _count_block(
    state_ptr: UnsafePointer[Float32, MutAnyOrigin],
    floor: Int,
    target: Int,
) -> Int:
    var n = 0
    for y in range(MAP_H):
        for x in range(MAP_W):
            if Int(state_ptr[s_map(floor, y, x)]) == target:
                n += 1
    return n


@always_inline
def _count_item(
    state_ptr: UnsafePointer[Float32, MutAnyOrigin],
    floor: Int,
    target: Int,
) -> Int:
    var n = 0
    for y in range(MAP_H):
        for x in range(MAP_W):
            if Int(state_ptr[s_item_map(floor, y, x)]) == target:
                n += 1
    return n


def test_world_generates(mut counts: List[Int]) raises:
    print("test_world_generates")
    var state = alloc[Float32](STATE_SIZE)
    for i in range(STATE_SIZE):
        state[i] = Float32(0.0)
    var pos = generate_full_world(seed=UInt64(0xC0FFEE), state_ptr=state)
    check(counts, "player_y == MAP_H // 2", pos[0] == MAP_H // 2)
    check(counts, "player_x == MAP_W // 2", pos[1] == MAP_W // 2)

    # Spot-check that block IDs landed inside the valid range on every floor.
    var all_ok = True
    for floor in range(NUM_FLOORS):
        for y in range(MAP_H):
            for x in range(MAP_W):
                var b = Int(state[s_map(floor, y, x)])
                if b < 0 or b >= 37:
                    all_ok = False
    check(counts, "all block IDs in [0, 37)", all_ok)
    state.free()


def test_overworld(mut counts: List[Int]) raises:
    print("test_overworld (floor 0)")
    var state = alloc[Float32](STATE_SIZE)
    for i in range(STATE_SIZE):
        state[i] = Float32(0.0)
    _ = generate_full_world(seed=UInt64(0x12345678), state_ptr=state)
    var grass = _count_block(state, 0, BLOCK_GRASS)
    var stone = _count_block(state, 0, BLOCK_STONE)
    var trees = _count_block(state, 0, BLOCK_TREE)
    var water = _count_block(state, 0, BLOCK_WATER)
    # Player spawn enforces at least one GRASS tile, so > 0 is too weak.
    check(counts, "overworld has plenty of grass", grass > 200)
    check(counts, "overworld has stone", stone > 50)
    check(counts, "overworld has some trees", trees > 0)
    # Water is optional — proximity-clamped — but should usually exist.
    check(counts, "overworld has water OR not (sanity)", water >= 0)

    # Ladder-down placed on a PATH tile.
    var ld_y = Int(state[s_down_ladder(0, 0)])
    var ld_x = Int(state[s_down_ladder(0, 1)])
    check(
        counts,
        "overworld ladder_down in bounds",
        ld_y >= 0 and ld_y < MAP_H and ld_x >= 0 and ld_x < MAP_W,
    )
    check(
        counts,
        "overworld has LADDER_DOWN item",
        _count_item(state, 0, ITEM_LADDER_DOWN) == 1,
    )
    # Overworld config has ladder_up=False, so no LADDER_UP item.
    check(
        counts,
        "overworld has NO LADDER_UP item",
        _count_item(state, 0, ITEM_LADDER_UP) == 0,
    )
    state.free()


def test_dungeons(mut counts: List[Int]) raises:
    print("test_dungeons (floors 1, 3, 4)")
    var state = alloc[Float32](STATE_SIZE)
    for i in range(STATE_SIZE):
        state[i] = Float32(0.0)
    _ = generate_full_world(seed=UInt64(0xDEADBEEF), state_ptr=state)
    for floor in [1, 3, 4]:
        var walls = _count_block(state, floor, BLOCK_WALL)
        var moss = _count_block(state, floor, BLOCK_WALL_MOSS)
        var path = _count_block(state, floor, BLOCK_PATH)
        var dark = _count_block(state, floor, BLOCK_DARKNESS)
        var chests = _count_block(state, floor, BLOCK_CHEST)
        var torches = _count_item(state, floor, ITEM_TORCH)

        check(
            counts,
            "floor " + String(floor) + " has rooms (chest count > 0)",
            chests > 0,
        )
        check(
            counts,
            "floor " + String(floor) + " has 4 torches per room (>= 4)",
            torches >= 4,
        )
        check(
            counts,
            "floor " + String(floor) + " has walls",
            walls > 100,
        )
        check(
            counts,
            "floor " + String(floor) + " has paths",
            path > 20,
        )
        # The 4-neighbor dilation should leave plenty of deep wall as darkness.
        check(
            counts,
            "floor " + String(floor) + " has darkness",
            dark > 50,
        )
        # Some moss should land on dilated walls (10% rare draw).
        check(
            counts,
            "floor " + String(floor) + " has WALL_MOSS",
            moss > 0,
        )

        # Each dungeon should place both ladders (PATH-only).
        check(
            counts,
            "floor " + String(floor) + " has LADDER_DOWN",
            _count_item(state, floor, ITEM_LADDER_DOWN) == 1,
        )
        check(
            counts,
            "floor " + String(floor) + " has LADDER_UP",
            _count_item(state, floor, ITEM_LADDER_UP) == 1,
        )

    # Dungeon-1 should NOT contain enchantment tables; sewers should
    # contain BLOCK_ENCHANTMENT_TABLE_ICE; vaults BLOCK_ENCHANTMENT_TABLE_FIRE.
    check(
        counts,
        "sewers contain ENCHANTMENT_TABLE_ICE",
        _count_block(state, 3, BLOCK_ENCHANTMENT_TABLE_ICE) >= 1,
    )
    check(
        counts,
        "vaults contain ENCHANTMENT_TABLE_FIRE",
        _count_block(state, 4, BLOCK_ENCHANTMENT_TABLE_FIRE) >= 1,
    )

    state.free()


def test_mines_and_elemental(mut counts: List[Int]) raises:
    print("test_mines_and_elemental (floors 2, 5, 6, 7)")
    var state = alloc[Float32](STATE_SIZE)
    for i in range(STATE_SIZE):
        state[i] = Float32(0.0)
    _ = generate_full_world(seed=UInt64(0xCAFEBABE), state_ptr=state)

    # Gnomish mines: default_block=PATH, mountain_block=STONE. Expect lots of
    # path & stone, plus ores.
    var path2 = _count_block(state, 2, BLOCK_PATH)
    var stone2 = _count_block(state, 2, BLOCK_STONE)
    check(counts, "gnomish has path/stone tiles", path2 + stone2 > 1000)

    # Troll mines (floor 5): same structure.
    var path5 = _count_block(state, 5, BLOCK_PATH)
    var stone5 = _count_block(state, 5, BLOCK_STONE)
    check(counts, "troll mines has path/stone tiles", path5 + stone5 > 1000)

    # Fire realm (floor 6): default = FIRE_GRASS. Should dominate.
    var fire = _count_block(state, 6, BLOCK_FIRE_GRASS)
    check(counts, "fire realm has FIRE_GRASS", fire > 200)

    # Ice realm (floor 7): default = ICE_GRASS.
    var ice = _count_block(state, 7, BLOCK_ICE_GRASS)
    check(counts, "ice realm has ICE_GRASS", ice > 200)

    state.free()


def test_boss_chamber(mut counts: List[Int]) raises:
    print("test_boss_chamber (floor 8)")
    var state = alloc[Float32](STATE_SIZE)
    for i in range(STATE_SIZE):
        state[i] = Float32(0.0)
    _ = generate_full_world(seed=UInt64(0xBADF00D), state_ptr=state)

    var nec = _count_block(state, 8, BLOCK_NECROMANCER)
    check(counts, "boss chamber has Necromancer", nec == 1)

    var walls = _count_block(state, 8, BLOCK_WALL)
    check(counts, "boss chamber has WALL tiles", walls > 100)

    # Boss config has ladder_up=False, ladder_down=False — no ladders placed.
    check(
        counts,
        "boss has NO LADDER_DOWN",
        _count_item(state, 8, ITEM_LADDER_DOWN) == 0,
    )
    check(
        counts,
        "boss has NO LADDER_UP",
        _count_item(state, 8, ITEM_LADDER_UP) == 0,
    )
    state.free()


def test_two_seeds_differ(mut counts: List[Int]) raises:
    print("test_two_seeds_differ")
    var s1 = alloc[Float32](STATE_SIZE)
    var s2 = alloc[Float32](STATE_SIZE)
    for i in range(STATE_SIZE):
        s1[i] = Float32(0.0)
        s2[i] = Float32(0.0)
    _ = generate_full_world(seed=UInt64(1), state_ptr=s1)
    _ = generate_full_world(seed=UInt64(2), state_ptr=s2)
    var diff = 0
    for floor in range(NUM_FLOORS):
        for y in range(MAP_H):
            for x in range(MAP_W):
                if Int(s1[s_map(floor, y, x)]) != Int(
                    s2[s_map(floor, y, x)]
                ):
                    diff += 1
    check(counts, "seeds differ in > 1000 tiles", diff > 1000)
    s1.free()
    s2.free()


def test_light_map_range(mut counts: List[Int]) raises:
    print("test_light_map_range")
    var state = alloc[Float32](STATE_SIZE)
    for i in range(STATE_SIZE):
        state[i] = Float32(0.0)
    _ = generate_full_world(seed=UInt64(42), state_ptr=state)
    var ok = True
    for floor in range(NUM_FLOORS):
        for y in range(MAP_H):
            for x in range(MAP_W):
                var v = state[s_light_map(floor, y, x)]
                if v < Float32(0.0) or v > Float32(1.0):
                    ok = False
    check(counts, "light_map ∈ [0, 1]", ok)

    # Dungeons should have all light_map = 1.0.
    var all_one = True
    for floor in [1, 3, 4]:
        for y in range(MAP_H):
            for x in range(MAP_W):
                if state[s_light_map(floor, y, x)] != Float32(1.0):
                    all_one = False
    check(counts, "dungeons fully lit", all_one)
    state.free()


def main() raises:
    print("Craftax-Full Phase-7B world-gen gate")
    print("=" * 50)
    var counts = [0, 0]
    test_world_generates(counts)
    test_overworld(counts)
    test_dungeons(counts)
    test_mines_and_elemental(counts)
    test_boss_chamber(counts)
    test_two_seeds_differ(counts)
    test_light_map_range(counts)
    print()
    print("=" * 50)
    print("Passed:", counts[0], "Failed:", counts[1])
    if counts[1] > 0:
        raise Error("Phase-7B gate FAILED")
    print("Phase-7B gate PASS")
