"""ASCII dump of one Craftax-Classic generated world.

Generates a 64×64 map from a given seed and prints it using one glyph per
block type. Use this to eyeball-compare structure against the JAX
reference at the same seed.

Glyphs:
  .  GRASS         #  STONE         T  TREE          P  PATH
  ~  WATER         *  SAND          c  COAL          i  IRON
  D  DIAMOND       L  LAVA          @  player spawn

Run:
  pixi run mojo run -I . examples/craftax_classic/dump_world.mojo
  pixi run mojo run -I . examples/craftax_classic/dump_world.mojo -- 1234   (custom seed)
"""

from std.sys import argv

from mojo_rl.envs.craftax_classic.constants import (
    MAP_H,
    MAP_W,
    MAP_SIZE,
    BLOCK_GRASS,
    BLOCK_WATER,
    BLOCK_STONE,
    BLOCK_TREE,
    BLOCK_PATH,
    BLOCK_COAL,
    BLOCK_IRON,
    BLOCK_DIAMOND,
    BLOCK_SAND,
    BLOCK_LAVA,
)
from mojo_rl.envs.craftax_classic.world_gen import generate_world_cpu


def glyph_for(block: Int) -> String:
    if block == BLOCK_GRASS:
        return "."
    if block == BLOCK_WATER:
        return "~"
    if block == BLOCK_STONE:
        return "#"
    if block == BLOCK_TREE:
        return "T"
    if block == BLOCK_PATH:
        return "P"
    if block == BLOCK_COAL:
        return "c"
    if block == BLOCK_IRON:
        return "i"
    if block == BLOCK_DIAMOND:
        return "D"
    if block == BLOCK_SAND:
        return "*"
    if block == BLOCK_LAVA:
        return "L"
    return "?"


def main() raises:
    var seed: UInt64 = 0
    var args = argv()
    # Skip "--" passed by `mojo run` to separate its flags from script args.
    for i in range(1, len(args)):
        var a = String(args[i])
        if a != "--":
            try:
                seed = UInt64(Int(a))
                break
            except:
                pass

    print("Craftax-Classic — world dump (seed =", seed, ")")
    print()

    var map_buf = InlineArray[Float32, MAP_SIZE](fill=Float32(0.0))
    var spawn = generate_world_cpu(
        seed, map_buf.unsafe_ptr().bitcast[Float32](), always_diamond=True
    )

    var py = spawn[0]
    var px = spawn[1]

    # Tally blocks for a quick sanity-check.
    var counts = InlineArray[Int, 17](fill=0)
    for i in range(MAP_SIZE):
        var b = Int(map_buf[i])
        if 0 <= b and b < 17:
            counts[b] += 1

    # Print map.
    for y in range(MAP_H):
        var line = String("")
        for x in range(MAP_W):
            if y == py and x == px:
                line += "@"
            else:
                line += glyph_for(Int(map_buf[y * MAP_W + x]))
        print(line)

    print()
    print("Block counts (excluding player tile):")
    print(
        "  GRASS:",
        counts[BLOCK_GRASS],
        "  WATER:",
        counts[BLOCK_WATER],
        "  STONE:",
        counts[BLOCK_STONE],
        "  TREE:",
        counts[BLOCK_TREE],
    )
    print(
        "  PATH:",
        counts[BLOCK_PATH],
        "  COAL:",
        counts[BLOCK_COAL],
        "  IRON:",
        counts[BLOCK_IRON],
        "  DIAMOND:",
        counts[BLOCK_DIAMOND],
    )
    print(
        "  SAND:",
        counts[BLOCK_SAND],
        "  LAVA:",
        counts[BLOCK_LAVA],
    )

    # Phase 2 invariants
    var spawn_block = Int(map_buf[py * MAP_W + px])
    print()
    print("Player spawn:", "(", py, ",", px, ")  block =", spawn_block)
    if spawn_block != BLOCK_GRASS:
        raise Error("INVARIANT FAILED: player spawn tile must be GRASS")
    if counts[BLOCK_DIAMOND] == 0:
        raise Error(
            "INVARIANT FAILED: always_diamond=True but no diamond placed"
        )
    print("Invariants OK.")
