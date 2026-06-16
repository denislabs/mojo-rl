"""Measure the frame runner's bulk vs per-clock clock split per game.

Flip `comptime ATARI_PROFILE = True` in mojo_rl/envs/atari/cpu6502.mojo
first (zero-cost False by default), then:

    pixi run mojo run -I . -D ASSERT=none benchmarks/profile_atari_span_split.mojo

Runs 200 headless frames per game with cycling actions and prints the
dbg_prof_* breakdown: bulk vs per-clock clocks, the lit-caused share of
per-clock (selective-ticking ceiling + avg active objects), and bulk
sub-span granularity (what the deferred-advance optimization amortizes).
This is the probe that sized the `CycleTIA.pending_ticks` deferred bulk
advance (+48% RAM / +35% pixel) — see docs/ATARI_AUDIT.md §2.

Requires ROMs under `roms/` (run from the repo root).
"""

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame


def profile_game(name: String, game: AtariGame) raises:
    var rom = load_rom("roms/" + name + ".bin")
    var env = AtariEnvironment(rom.data.value(), rom.size)
    env.reset_game(game)
    # Reset accumulators after the reset frames.
    env.state.dbg_prof_bulk_clocks = 0
    env.state.dbg_prof_perclock = 0
    env.state.dbg_prof_perclock_target = 0
    env.state.dbg_prof_active_ticks = 0
    env.state.dbg_prof_bulk_spans = 0
    env.state.dbg_prof_bulk_visible_spans = 0

    for f in range(200):
        _ = env.step_game(game, f % game.num_actions())

    var bulk = env.state.dbg_prof_bulk_clocks
    var pc = env.state.dbg_prof_perclock
    var tgt = env.state.dbg_prof_perclock_target
    var act = env.state.dbg_prof_active_ticks
    var total = bulk + pc
    if total == 0:
        print(name, "| ATARI_PROFILE is False — flip it in cpu6502.mojo")
        return
    print(
        name,
        "| total clocks", total,
        "| bulk", bulk, "(", Float64(bulk) * 100.0 / Float64(total), "% )",
        "| per-clock", pc, "(", Float64(pc) * 100.0 / Float64(total), "% )",
    )
    print(
        "   per-clock lit-target", tgt,
        "(", Float64(tgt) * 100.0 / Float64(pc) if pc > 0 else 0.0,
        "% of per-clock ) | avg active objects",
        Float64(act) / Float64(tgt) if tgt > 0 else 0.0,
        "/ 5",
    )
    var spans = env.state.dbg_prof_bulk_spans
    var vspans = env.state.dbg_prof_bulk_visible_spans
    print(
        "   bulk sub-spans", spans,
        "( avg", Float64(bulk) / Float64(spans) if spans > 0 else 0.0,
        "clk ) | visible (paid advance/horizon when not deferred)", vspans,
        "( avg",
        Float64(bulk) / Float64(vspans) if vspans > 0 else 0.0,
        "clk )",
    )


def main() raises:
    profile_game("pong", AtariGame.PONG)
    profile_game("breakout", AtariGame.BREAKOUT)
    profile_game("space_invaders", AtariGame.SPACE_INVADERS)
    profile_game("seaquest", AtariGame.SEAQUEST)
    profile_game("ms_pacman", AtariGame.MS_PACMAN)
