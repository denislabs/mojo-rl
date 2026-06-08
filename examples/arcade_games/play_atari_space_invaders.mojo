"""Interactive Atari 2600 Space Invaders player — keyboard controlled.

Usage:
    pixi run -e apple mojo run -I . examples/arcade_games/play_atari_space_invaders.mojo

Controls:
    Left / Right  → Move ship
    Space         → FIRE
    P             → Pause/Unpause
    V             → Toggle video recording
    Escape/Q      → Quit

Space Invaders is a joystick game: LEFT/RIGHT move the ship (read from SWCHA)
and Space fires (read from INPT4). Unlike Pong it does not use the paddle, so
this exercises the joystick/fire input path of the emulator.

Requires ROM files in 'roms/' (symlink to ale_py/roms/).
"""

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.renderer import AtariRenderer
from mojo_rl.envs.atari.riot import set_action
from mojo_rl.envs.atari.cpu6502 import run_frame_with_video
from mojo_rl.envs.atari.games.space_invaders import SpaceInvadersDef
from mojo_rl.envs.atari.flags import ACTION_NOOP, ACTION_RESET


def main() raises:
    # Load ROM
    var rom_path = "roms/space_invaders.bin"
    print("Loading ROM: " + rom_path)
    var rom_data = load_rom(rom_path)
    print("ROM loaded: " + String(rom_data.size) + " bytes")

    # Create environment (frame_skip=1 for interactive play)
    var env = AtariEnvironment(
        rom_data.data.value(), rom_data.size, frame_skip=1, max_frames=0
    )
    env.reset()
    env.state.lives = UInt8(SpaceInvadersDef.get_lives(env.state.ram))
    print("Environment reset. Starting interactive play...")
    print("")
    print("Controls:")
    print("  Left/Right = Move ship")
    print("  Space      = Fire")
    print("  P          = Pause")
    print("  V          = Record video")
    print("  Esc/Q      = Quit")

    # Create renderer
    var renderer = AtariRenderer(fps=60)
    if not renderer.init_display():
        print("Failed to initialize display")
        return

    # Main loop
    var step_count = 0
    while not renderer.should_quit:
        # Process input
        if not renderer.handle_events():
            break

        if not renderer.paused:
            # Set the action and run one frame with video rendering
            var act = renderer.current_action
            set_action(env.state, act)
            run_frame_with_video(
                env.state, env.rom, env.rom_size, renderer.get_pixel_buffer()
            )
            step_count += 1

            # Debug RAM state every 300 frames (~5 sec)
            if step_count % 300 == 1:
                print(
                    "score="
                    + String(SpaceInvadersDef.get_score(env.state.ram))
                    + " lives="
                    + String(SpaceInvadersDef.get_lives(env.state.ram))
                    + " P0="
                    + String(Int(env.state.pos_p0))
                    + " M0="
                    + String(Int(env.state.pos_m0))
                )

            # Extract game state from RAM
            var score = SpaceInvadersDef.get_score(env.state.ram)
            env.state.score = Int32(score)
            env.state.lives = UInt8(SpaceInvadersDef.get_lives(env.state.ram))
            env.state.terminal = SpaceInvadersDef.is_terminal(env.state.ram)

            # Auto-reset on terminal
            if env.state.terminal:
                print(
                    "Game over! Score: "
                    + String(score)
                    + " (step "
                    + String(step_count)
                    + ")"
                )
                env.reset()
                env.state.lives = UInt8(
                    SpaceInvadersDef.get_lives(env.state.ram)
                )
                step_count = 0

        # Display the frame buffer (already filled by run_frame_with_video)
        renderer.display_buffer_with_hud(
            Int(env.state.score),
            Int(env.state.lives),
            Int(env.state.frame_number),
        )

    renderer.close()
    print("Done.")
