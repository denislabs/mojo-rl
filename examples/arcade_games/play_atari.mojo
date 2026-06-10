"""Interactive Atari 2600 player — ONE binary, any registered game.

Usage:
    pixi run -e apple mojo run -I . examples/arcade_games/play_atari.mojo ms_pacman
    pixi run -e apple mojo run -I . examples/arcade_games/play_atari.mojo qbert

With no argument, lists the registered games.

Controls:
    Arrow keys  → Joystick directions
    Space       → FIRE button
    R           → Console RESET switch (start a game)
    P           → Pause/Unpause
    V           → Toggle video recording
    Escape/Q    → Quit

Requires ROM files in 'roms/' (ale-py naming).
"""

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.renderer import AtariRenderer
from mojo_rl.envs.atari.riot import set_action
from mojo_rl.envs.atari.cpu6502 import run_frame_video
from mojo_rl.envs.atari.games.registry import AtariGame, game_signals
from std.sys import argv


def main() raises:
    if len(argv()) < 2:
        print("Usage: play_atari <game>")
        print("Registered games:")
        for gid in range(AtariGame.NUM_GAMES):
            print("  " + AtariGame.from_id(gid).name())
        return

    var game = AtariGame.from_name(String(argv()[1]))
    var rom_path = game.rom_file()
    print("Loading ROM: " + rom_path)
    var rom_data = load_rom(rom_path)
    print("ROM loaded: " + String(rom_data.size) + " bytes")

    # Create environment (frame_skip=1 for interactive play)
    var env = AtariEnvironment(
        rom_data.data.value(), rom_data.size, frame_skip=1, max_frames=0
    )
    env.reset()
    print("Environment reset. Starting interactive play...")
    print("")
    print("Controls:")
    print("  Arrow keys = Move")
    print("  Space      = Fire")
    print("  R          = Console RESET (start game)")
    print("  P          = Pause")
    print("  V          = Record video")
    print("  Esc/Q      = Quit")

    var renderer = AtariRenderer(fps=60)
    if not renderer.init_display():
        print("Failed to initialize display")
        return

    var step_count = 0
    while not renderer.should_quit:
        if not renderer.handle_events():
            break

        if not renderer.paused:
            var act = renderer.current_action
            set_action(env.state, act)
            run_frame_video(
                env.state, env.rom, env.rom_size, renderer.get_pixel_buffer()
            )
            step_count += 1

            # Extract game state from RAM via the registry
            var sig = game_signals(
                game, env.state.ram, Int(env.state.score)
            )
            env.state.score = Int32(sig.score)
            env.state.lives = UInt8(sig.lives)
            env.state.terminal = sig.terminal

            # Auto-reset on terminal
            if env.state.terminal:
                print(
                    "Game over! Score: "
                    + String(sig.score)
                    + " (step "
                    + String(step_count)
                    + ")"
                )
                env.reset()
                step_count = 0

        renderer.display_buffer_with_hud(
            Int(env.state.score),
            Int(env.state.lives),
            Int(env.state.frame_number),
        )

    renderer.close()
    print("Done.")
