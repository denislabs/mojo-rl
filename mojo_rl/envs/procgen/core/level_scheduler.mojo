"""`LevelScheduler` — Procgen level selection (train/test generalization splits).

Ports the `vecgame.cpp` per-env seeding + `Game::reset` level-seed draw for a
single env:

- `num_levels == 0` → levels drawn from `[0, INT32_MAX)` (unbounded).
- `num_levels > 0`  → levels drawn from `[start_level, start_level + num_levels)`.

`game_level_seed_gen` is seeded with `rand_seed`; its first raw draw seeds
`level_seed_rand_gen`, which then yields one `current_level_seed` per episode via
`randint(low, high)`. This is the mechanism behind Procgen's core benchmark:
train on a finite level set, evaluate on held-out levels. Reusable by every
procgen game. Validated in `tests/envs/procgen/test_maze_env_parity.mojo`.
"""

from .randgen import RandGen

comptime INT32_MAX = 2147483647


struct LevelScheduler(Copyable, Movable):
    var game_level_seed_gen: RandGen
    var level_seed_rand_gen: RandGen
    var level_seed_low: Int
    var level_seed_high: Int

    def __init__(out self, rand_seed: Int, num_levels: Int, start_level: Int):
        if num_levels == 0:
            self.level_seed_low = 0
            self.level_seed_high = INT32_MAX
        else:
            self.level_seed_low = start_level
            self.level_seed_high = start_level + num_levels
        self.game_level_seed_gen = RandGen()
        self.game_level_seed_gen.seed(rand_seed)
        self.level_seed_rand_gen = RandGen()
        # C++ seeds with the raw (no-arg) draw; int<->uint32 round-trips exactly.
        self.level_seed_rand_gen.seed(Int(self.game_level_seed_gen.randint()))

    def next_level_seed(mut self) -> Int:
        return self.level_seed_rand_gen.randint(
            self.level_seed_low, self.level_seed_high
        )
