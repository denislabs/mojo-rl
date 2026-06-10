"""TIA data-bus noise semantics (Stella TIA::peek `noise = dataBusState & 0x3F`).

TIA reads only drive bits 7/6; the low 6 bits leak the last byte on the data
bus. For a zero-page read the last bus byte is the operand itself, so reading
unmapped TIA $0F returns $0F (= 15). Haunted House's display kernel divides
by 15 with `SBC $0f` at $F44F — without the leak the divisor reads 0 and the
subtract loop never terminates (boot freeze, 2x262-line capped frames).

Run: pixi run mojo run -I . tests/arcade_games/test_tia_bus_noise.mojo
"""

from mojo_rl.envs.atari.atari_state import AtariState
from mojo_rl.envs.atari.tia import tia_read
from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.cpu6502 import cpu_reset, run_frame
from mojo_rl.envs.atari.cartridge import init_bank
from mojo_rl.envs.atari.riot import set_action
from mojo_rl.envs.atari.flags import ACTION_NOOP, ROM_AUTO, CX_BLPF


def test_unmapped_read_returns_bus_noise() raises:
    var state = AtariState()
    state.data_bus = 0x0F
    if tia_read(state, 0x0F) != 0x0F:
        raise Error("unmapped TIA $0F must return bus noise (operand byte)")
    state.data_bus = 0xFF  # only the low 6 bits leak
    if tia_read(state, 0x0E) != 0x3F:
        raise Error("noise must be masked to the low 6 bits")
    print("PASS unmapped TIA reads return data-bus noise")


def test_mapped_reads_mix_noise_low_bits() raises:
    var state = AtariState()
    state.data_bus = 0x06
    state.collision = CX_BLPF
    if tia_read(state, 0x06) != UInt8(0x80 | 0x06):  # CXBLPF
        raise Error("collision read must be bits7/6 | bus noise")
    state.data_bus = 0x0C
    if tia_read(state, 0x0C) != UInt8(0x80 | 0x0C):  # INPT4, not pressed
        raise Error("INPT4 read must be bit7 | bus noise")
    print("PASS mapped TIA reads carry bus noise in the low 6 bits")


def test_haunted_house_boots() raises:
    # End-to-end gate: without the bus leak the divide-by-15 at $F44F spins
    # forever and every frame hits the 2x262 scanline cap.
    var rom_data = load_rom("roms/haunted_house.bin")
    var env = AtariEnvironment(
        rom_data.data.value(), rom_data.size, frame_skip=1, max_frames=0
    )
    init_bank(env.state, env.rom_size, ROM_AUTO)
    cpu_reset(env.state, env.rom, env.rom_size)
    for _ in range(120):
        set_action(env.state, ACTION_NOOP)
        run_frame(env.state, env.rom, env.rom_size)
    var lines = Int(env.state.dbg_frame_lines)
    if lines > 400:
        raise Error(
            "haunted_house frame hit the scanline cap (boot freeze): "
            + String(lines)
        )
    print("PASS haunted_house boots (frame =", lines, "scanlines)")


def main() raises:
    test_unmapped_read_returns_bus_noise()
    test_mapped_reads_mix_noise_low_bits()
    test_haunted_house_boots()
    print("ALL PASS")
