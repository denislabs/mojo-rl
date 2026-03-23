"""Interactive Atari 2600 player — play games with keyboard input.

Usage:
    pixi run -e apple mojo run envs/atari/play_atari.mojo

Controls:
    Arrow keys  → Joystick directions
    Space       → FIRE button
    P           → Pause/Unpause
    V           → Toggle video recording
    Escape/Q    → Quit

Requires ROM files in 'roms/' (symlink to ale_py/roms/).
"""

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.renderer import AtariRenderer
from mojo_rl.envs.atari.riot import set_action
from mojo_rl.envs.atari.cpu6502 import run_frame_with_video
from mojo_rl.envs.atari.games.pong import PongDef
from mojo_rl.envs.atari.flags import ACTION_NOOP, ACTION_RESET


def main() raises:
    # Load ROM
    var rom_path = "roms/pong.bin"
    print("Loading ROM: " + rom_path)
    var rom_data = load_rom(rom_path)
    print("ROM loaded: " + String(rom_data.size) + " bytes")

    # Create environment (frame_skip=1 for interactive play)
    var env = AtariEnvironment(
        rom_data.data, rom_data.size, frame_skip=1, max_frames=0
    )
    env.reset()
    print("Environment reset. Starting interactive play...")
    print("")
    print("Controls:")
    print("  Arrow keys = Move")
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

            # Debug TIA state every 300 frames (~5 sec)
            if step_count % 300 == 1:
                print(
                    "P0="
                    + String(Int(env.state.pos_p0))
                    + " P1="
                    + String(Int(env.state.pos_p1))
                    + " BL="
                    + String(Int(env.state.pos_bl))
                    + " paddle="
                    + String(Int(env.state.paddle_pos))
                    + " RAM13="
                    + String(Int(env.state.ram[13]))
                    + " RAM14="
                    + String(Int(env.state.ram[14]))
                )

            # Extract game state from RAM
            var score = PongDef.get_score(env.state.ram)
            env.state.score = Int32(score)
            env.state.terminal = PongDef.is_terminal(env.state.ram)

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
                step_count = 0

        # Display the frame buffer (already filled by run_frame_with_video)
        renderer.display_buffer_with_hud(
            Int(env.state.score),
            Int(env.state.lives),
            Int(env.state.frame_number),
        )

    renderer.close()
    print("Done.")
