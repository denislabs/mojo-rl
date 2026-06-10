"""Runtime game registry — one env, any game.

`AtariGame` is a DType-style value enum: a trivially-copyable id wrapped in a
struct with comptime constants, so call sites write `AtariGame.PONG` instead of
a magic number. Everything game-specific is dispatched at RUNTIME from it:

  - `game.rom_file()`        → "roms/<name>.bin" (ale-py ROM naming)
  - `game.num_actions()`     → minimal action set size
  - `game.action(idx)`       → idx-th ALE action of the minimal set
  - `game_signals(game,...)` → score / reward / lives / terminal from RAM

Runtime (vs comptime) dispatch is deliberate: this logic runs once per STEP (a
handful of RAM reads) while the emulator runs ~30k instructions per frame, so
the cost is invisible — and one compiled binary can play every game.

Minimal action sets are encoded as an 18-bit mask over the standard ALE action
ids (flags.mojo ACTION_* == ALE enum values) and enumerated in ASCENDING id
order, exactly like ALE's getMinimalActionSet(). NOTE: for Space Invaders this
ordering differs from the legacy SpaceInvadersDef.map_action (which puts LEFT
before FIRE); agents trained on one mapping are not action-compatible with the
other.

Per-game RAM logic is a faithful port of ALE's games/supported/<Game>.cpp
step() bodies (GPL reference, re-implemented). RAM addresses ≥ 0x80 are
mirrors of the 128-byte RIOT RAM — `_rb` masks to 7 bits like ALE's readRam.
"""

from ..flags import (
    RAM_SIZE,
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_UP,
    ACTION_RIGHT,
    ACTION_LEFT,
    ACTION_DOWN,
    ACTION_UPRIGHT,
    ACTION_UPLEFT,
    ACTION_DOWNRIGHT,
    ACTION_DOWNLEFT,
    ACTION_UPFIRE,
    ACTION_RIGHTFIRE,
    ACTION_LEFTFIRE,
    ACTION_DOWNFIRE,
    ACTION_UPRIGHTFIRE,
    ACTION_UPLEFTFIRE,
    ACTION_DOWNRIGHTFIRE,
    ACTION_DOWNLEFTFIRE,
    ACTION_RESET,
    ROM_AUTO,
    ROM_E0,
    ROM_FE,
    ROM_F8SC,
    ROM_F6SC,
)
from ..atari_state import AtariState
from .helpers import (
    get_decimal_score,
    get_decimal_score_2,
    get_decimal_score_3,
)


@always_inline
def _bit(a: UInt8) -> UInt32:
    return UInt32(1) << UInt32(a)


@always_inline
def _rb(ram: InlineArray[UInt8, RAM_SIZE], addr: Int) -> Int:
    """Read a RAM byte by ALE address (mirrors masked to the 128-byte RAM)."""
    return Int(ram[addr & 0x7F])


@always_inline
def _sb(ram: InlineArray[UInt8, RAM_SIZE], addr: Int) -> Int:
    """Read a RAM byte as a SIGNED value (-128..127), like a C++
    signed-char cast (Backgammon piece counts, WordZapper rounds)."""
    var v = Int(ram[addr & 0x7F])
    return v - 256 if v >= 128 else v


# Mask of all 18 standard actions.
comptime ALL_ACTIONS_MASK: UInt32 = 0x3FFFF


struct AtariGame(Copyable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    """Value-enum of supported games (DType-style)."""

    var id: UInt8

    comptime PONG = AtariGame(0)
    comptime BREAKOUT = AtariGame(1)
    comptime SPACE_INVADERS = AtariGame(2)
    comptime MS_PACMAN = AtariGame(3)
    comptime SEAQUEST = AtariGame(4)
    comptime QBERT = AtariGame(5)
    comptime ASTEROIDS = AtariGame(6)
    comptime FROSTBITE = AtariGame(7)
    comptime FREEWAY = AtariGame(8)
    comptime BOXING = AtariGame(9)
    comptime ENDURO = AtariGame(10)
    comptime AMIDAR = AtariGame(11)
    comptime ATLANTIS = AtariGame(12)
    # Wave 1 (alphabetical from here on)
    comptime ALIEN = AtariGame(13)
    comptime ASSAULT = AtariGame(14)
    comptime ASTERIX = AtariGame(15)
    comptime BANK_HEIST = AtariGame(16)
    comptime BATTLE_ZONE = AtariGame(17)
    comptime BEAM_RIDER = AtariGame(18)
    comptime BERZERK = AtariGame(19)
    comptime BOWLING = AtariGame(20)
    comptime CENTIPEDE = AtariGame(21)
    comptime CHOPPER_COMMAND = AtariGame(22)
    comptime CRAZY_CLIMBER = AtariGame(23)
    comptime DARK_CHAMBERS = AtariGame(24)
    comptime DEMON_ATTACK = AtariGame(25)
    comptime ELEVATOR_ACTION = AtariGame(26)
    comptime FISHING_DERBY = AtariGame(27)
    comptime JAMESBOND = AtariGame(28)
    comptime KLAX = AtariGame(29)
    comptime MONTEZUMA_REVENGE = AtariGame(30)
    comptime ROBOTANK = AtariGame(31)
    comptime TUTANKHAM = AtariGame(32)
    # Wave 2: completes the Atari-57 benchmark set (+ pooyan)
    comptime DEFENDER = AtariGame(33)
    comptime DOUBLE_DUNK = AtariGame(34)
    comptime GOPHER = AtariGame(35)
    comptime GRAVITAR = AtariGame(36)
    comptime HERO = AtariGame(37)
    comptime ICE_HOCKEY = AtariGame(38)
    comptime KANGAROO = AtariGame(39)
    comptime KRULL = AtariGame(40)
    comptime KUNG_FU_MASTER = AtariGame(41)
    comptime NAME_THIS_GAME = AtariGame(42)
    comptime PHOENIX = AtariGame(43)
    comptime PITFALL = AtariGame(44)
    comptime POOYAN = AtariGame(45)
    comptime PRIVATE_EYE = AtariGame(46)
    comptime RIVERRAID = AtariGame(47)
    comptime ROAD_RUNNER = AtariGame(48)
    comptime SKIING = AtariGame(49)
    comptime SOLARIS = AtariGame(50)
    comptime STAR_GUNNER = AtariGame(51)
    comptime SURROUND = AtariGame(52)
    comptime TENNIS = AtariGame(53)
    comptime TIME_PILOT = AtariGame(54)
    comptime UP_N_DOWN = AtariGame(55)
    comptime VENTURE = AtariGame(56)
    comptime VIDEO_PINBALL = AtariGame(57)
    comptime WIZARD_OF_WOR = AtariGame(58)
    comptime YARS_REVENGE = AtariGame(59)
    comptime ZAXXON = AtariGame(60)
    # Wave 3: every remaining ALE-supported game in the ROM set
    comptime ADVENTURE = AtariGame(61)
    comptime AIR_RAID = AtariGame(62)
    comptime ATLANTIS2 = AtariGame(63)
    comptime BACKGAMMON = AtariGame(64)
    comptime BASIC_MATH = AtariGame(65)
    comptime BLACKJACK = AtariGame(66)
    comptime CARNIVAL = AtariGame(67)
    comptime CASINO = AtariGame(68)
    comptime CROSSBOW = AtariGame(69)
    comptime DONKEY_KONG = AtariGame(70)
    comptime EARTHWORLD = AtariGame(71)
    comptime ENTOMBED = AtariGame(72)
    comptime ET = AtariGame(73)
    comptime FLAG_CAPTURE = AtariGame(74)
    comptime FROGGER = AtariGame(75)
    comptime GALAXIAN = AtariGame(76)
    comptime HANGMAN = AtariGame(77)
    comptime HAUNTED_HOUSE = AtariGame(78)
    comptime HUMAN_CANNONBALL = AtariGame(79)
    comptime JOURNEY_ESCAPE = AtariGame(80)
    comptime KABOOM = AtariGame(81)
    comptime KEYSTONE_KAPERS = AtariGame(82)
    comptime KING_KONG = AtariGame(83)
    comptime KOOLAID = AtariGame(84)
    comptime LASER_GATES = AtariGame(85)
    comptime LOST_LUGGAGE = AtariGame(86)
    comptime MARIO_BROS = AtariGame(87)
    comptime MINIATURE_GOLF = AtariGame(88)
    comptime MR_DO = AtariGame(89)
    comptime OTHELLO = AtariGame(90)
    comptime PACMAN = AtariGame(91)
    comptime SIR_LANCELOT = AtariGame(92)
    comptime SPACE_WAR = AtariGame(93)
    comptime SUPERMAN = AtariGame(94)
    comptime TETRIS = AtariGame(95)
    comptime TIC_TAC_TOE_3D = AtariGame(96)
    comptime TRONDEAD = AtariGame(97)
    comptime TURMOIL = AtariGame(98)
    comptime VIDEO_CHECKERS = AtariGame(99)
    comptime VIDEO_CHESS = AtariGame(100)
    comptime VIDEO_CUBE = AtariGame(101)
    comptime WORD_ZAPPER = AtariGame(102)

    comptime NUM_GAMES: Int = 103

    @always_inline
    def __init__(out self, id: UInt8):
        self.id = id

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.id == other.id

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self.id != other.id

    @staticmethod
    def from_id(id: Int) -> AtariGame:
        """Construct from a plain index (e.g. CLI arg); must be < NUM_GAMES."""
        return AtariGame(UInt8(id))

    @staticmethod
    def from_name(name: String) raises -> AtariGame:
        """Look up a game by its registry name (e.g. "ms_pacman")."""
        for gid in range(AtariGame.NUM_GAMES):
            var g = AtariGame.from_id(gid)
            if g.name() == name:
                return g
        raise Error("unknown game: " + name)

    def name(self) -> String:
        if self == AtariGame.PONG:
            return "pong"
        elif self == AtariGame.BREAKOUT:
            return "breakout"
        elif self == AtariGame.SPACE_INVADERS:
            return "space_invaders"
        elif self == AtariGame.MS_PACMAN:
            return "ms_pacman"
        elif self == AtariGame.SEAQUEST:
            return "seaquest"
        elif self == AtariGame.QBERT:
            return "qbert"
        elif self == AtariGame.ASTEROIDS:
            return "asteroids"
        elif self == AtariGame.FROSTBITE:
            return "frostbite"
        elif self == AtariGame.FREEWAY:
            return "freeway"
        elif self == AtariGame.BOXING:
            return "boxing"
        elif self == AtariGame.ENDURO:
            return "enduro"
        elif self == AtariGame.AMIDAR:
            return "amidar"
        elif self == AtariGame.ATLANTIS:
            return "atlantis"
        elif self == AtariGame.ALIEN:
            return "alien"
        elif self == AtariGame.ASSAULT:
            return "assault"
        elif self == AtariGame.ASTERIX:
            return "asterix"
        elif self == AtariGame.BANK_HEIST:
            return "bank_heist"
        elif self == AtariGame.BATTLE_ZONE:
            return "battle_zone"
        elif self == AtariGame.BEAM_RIDER:
            return "beam_rider"
        elif self == AtariGame.BERZERK:
            return "berzerk"
        elif self == AtariGame.BOWLING:
            return "bowling"
        elif self == AtariGame.CENTIPEDE:
            return "centipede"
        elif self == AtariGame.CHOPPER_COMMAND:
            return "chopper_command"
        elif self == AtariGame.CRAZY_CLIMBER:
            return "crazy_climber"
        elif self == AtariGame.DARK_CHAMBERS:
            return "darkchambers"  # ale-py ROM name has no underscore
        elif self == AtariGame.DEMON_ATTACK:
            return "demon_attack"
        elif self == AtariGame.ELEVATOR_ACTION:
            return "elevator_action"
        elif self == AtariGame.FISHING_DERBY:
            return "fishing_derby"
        elif self == AtariGame.JAMESBOND:
            return "jamesbond"
        elif self == AtariGame.KLAX:
            return "klax"
        elif self == AtariGame.MONTEZUMA_REVENGE:
            return "montezuma_revenge"
        elif self == AtariGame.ROBOTANK:
            return "robotank"
        elif self == AtariGame.TUTANKHAM:
            return "tutankham"
        elif self == AtariGame.DEFENDER:
            return "defender"
        elif self == AtariGame.DOUBLE_DUNK:
            return "double_dunk"
        elif self == AtariGame.GOPHER:
            return "gopher"
        elif self == AtariGame.GRAVITAR:
            return "gravitar"
        elif self == AtariGame.HERO:
            return "hero"
        elif self == AtariGame.ICE_HOCKEY:
            return "ice_hockey"
        elif self == AtariGame.KANGAROO:
            return "kangaroo"
        elif self == AtariGame.KRULL:
            return "krull"
        elif self == AtariGame.KUNG_FU_MASTER:
            return "kung_fu_master"
        elif self == AtariGame.NAME_THIS_GAME:
            return "name_this_game"
        elif self == AtariGame.PHOENIX:
            return "phoenix"
        elif self == AtariGame.PITFALL:
            return "pitfall"
        elif self == AtariGame.POOYAN:
            return "pooyan"
        elif self == AtariGame.PRIVATE_EYE:
            return "private_eye"
        elif self == AtariGame.RIVERRAID:
            return "riverraid"
        elif self == AtariGame.ROAD_RUNNER:
            return "road_runner"
        elif self == AtariGame.SKIING:
            return "skiing"
        elif self == AtariGame.SOLARIS:
            return "solaris"
        elif self == AtariGame.STAR_GUNNER:
            return "star_gunner"
        elif self == AtariGame.SURROUND:
            return "surround"
        elif self == AtariGame.TENNIS:
            return "tennis"
        elif self == AtariGame.TIME_PILOT:
            return "time_pilot"
        elif self == AtariGame.UP_N_DOWN:
            return "up_n_down"
        elif self == AtariGame.VENTURE:
            return "venture"
        elif self == AtariGame.VIDEO_PINBALL:
            return "video_pinball"
        elif self == AtariGame.WIZARD_OF_WOR:
            return "wizard_of_wor"
        elif self == AtariGame.YARS_REVENGE:
            return "yars_revenge"
        elif self == AtariGame.ZAXXON:
            return "zaxxon"
        elif self == AtariGame.ADVENTURE:
            return "adventure"
        elif self == AtariGame.AIR_RAID:
            return "air_raid"
        elif self == AtariGame.ATLANTIS2:
            return "atlantis2"
        elif self == AtariGame.BACKGAMMON:
            return "backgammon"
        elif self == AtariGame.BASIC_MATH:
            return "basic_math"
        elif self == AtariGame.BLACKJACK:
            return "blackjack"
        elif self == AtariGame.CARNIVAL:
            return "carnival"
        elif self == AtariGame.CASINO:
            return "casino"
        elif self == AtariGame.CROSSBOW:
            return "crossbow"
        elif self == AtariGame.DONKEY_KONG:
            return "donkey_kong"
        elif self == AtariGame.EARTHWORLD:
            return "earthworld"
        elif self == AtariGame.ENTOMBED:
            return "entombed"
        elif self == AtariGame.ET:
            return "et"
        elif self == AtariGame.FLAG_CAPTURE:
            return "flag_capture"
        elif self == AtariGame.FROGGER:
            return "frogger"
        elif self == AtariGame.GALAXIAN:
            return "galaxian"
        elif self == AtariGame.HANGMAN:
            return "hangman"
        elif self == AtariGame.HAUNTED_HOUSE:
            return "haunted_house"
        elif self == AtariGame.HUMAN_CANNONBALL:
            return "human_cannonball"
        elif self == AtariGame.JOURNEY_ESCAPE:
            return "journey_escape"
        elif self == AtariGame.KABOOM:
            return "kaboom"
        elif self == AtariGame.KEYSTONE_KAPERS:
            return "keystone_kapers"
        elif self == AtariGame.KING_KONG:
            return "king_kong"
        elif self == AtariGame.KOOLAID:
            return "koolaid"
        elif self == AtariGame.LASER_GATES:
            return "laser_gates"
        elif self == AtariGame.LOST_LUGGAGE:
            return "lost_luggage"
        elif self == AtariGame.MARIO_BROS:
            return "mario_bros"
        elif self == AtariGame.MINIATURE_GOLF:
            return "miniature_golf"
        elif self == AtariGame.MR_DO:
            return "mr_do"
        elif self == AtariGame.OTHELLO:
            return "othello"
        elif self == AtariGame.PACMAN:
            return "pacman"
        elif self == AtariGame.SIR_LANCELOT:
            return "sir_lancelot"
        elif self == AtariGame.SPACE_WAR:
            return "space_war"
        elif self == AtariGame.SUPERMAN:
            return "superman"
        elif self == AtariGame.TETRIS:
            return "tetris"
        elif self == AtariGame.TIC_TAC_TOE_3D:
            return "tic_tac_toe_3d"
        elif self == AtariGame.TRONDEAD:
            return "trondead"
        elif self == AtariGame.TURMOIL:
            return "turmoil"
        elif self == AtariGame.VIDEO_CHECKERS:
            return "video_checkers"
        elif self == AtariGame.VIDEO_CHESS:
            return "video_chess"
        elif self == AtariGame.VIDEO_CUBE:
            return "video_cube"
        elif self == AtariGame.WORD_ZAPPER:
            return "word_zapper"
        return "unknown"

    def rom_file(self) -> String:
        """ROM path relative to the repo root (ale-py ROM naming)."""
        return "roms/" + self.name() + ".bin"

    def mapper(self) -> UInt8:
        """Cartridge mapper override (ROM_* id), or ROM_AUTO for size-based.

        Size detection cannot distinguish F8 / E0 / FE / F8SC at 8K (or
        F6 / F6SC at 16K). Values baked from running ALE's
        Cartridge::autodetectType content signatures over the ROM set; only
        games whose mapper differs from the size default appear here.
        """
        if (
            self == AtariGame.JAMESBOND
            or self == AtariGame.MONTEZUMA_REVENGE
            or self == AtariGame.TUTANKHAM
        ):
            return ROM_E0
        elif self == AtariGame.ROBOTANK:
            return ROM_FE
        elif self == AtariGame.ELEVATOR_ACTION:
            return ROM_F8SC
        elif self == AtariGame.DARK_CHAMBERS or self == AtariGame.KLAX:
            return ROM_F6SC
        return ROM_AUTO

    def action_mask(self) -> UInt32:
        """18-bit mask of the game's minimal action set (ALE isMinimal)."""
        if self == AtariGame.PONG or self == AtariGame.SPACE_INVADERS:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif self == AtariGame.BREAKOUT or self == AtariGame.KABOOM:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
            )
        elif self == AtariGame.MS_PACMAN:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPRIGHT)
                | _bit(ACTION_UPLEFT)
                | _bit(ACTION_DOWNRIGHT)
                | _bit(ACTION_DOWNLEFT)
            )
        elif (
            self == AtariGame.QBERT
            or self == AtariGame.BASIC_MATH
            or self == AtariGame.KING_KONG
        ):
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
            )
        elif self == AtariGame.ASTEROIDS:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPRIGHT)
                | _bit(ACTION_UPLEFT)
                | _bit(ACTION_UPFIRE)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
                | _bit(ACTION_DOWNFIRE)
                | _bit(ACTION_UPRIGHTFIRE)
                | _bit(ACTION_UPLEFTFIRE)
            )
        elif self == AtariGame.FREEWAY:
            return _bit(ACTION_NOOP) | _bit(ACTION_UP) | _bit(ACTION_DOWN)
        elif self == AtariGame.ENDURO:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_DOWNRIGHT)
                | _bit(ACTION_DOWNLEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif self == AtariGame.AMIDAR:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPFIRE)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
                | _bit(ACTION_DOWNFIRE)
            )
        elif self == AtariGame.ATLANTIS or self == AtariGame.ATLANTIS2:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif self == AtariGame.ASSAULT:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif (
            self == AtariGame.ASTERIX
            or self == AtariGame.CRAZY_CLIMBER
            or self == AtariGame.KOOLAID
            or self == AtariGame.LOST_LUGGAGE
        ):
            # 4 directions + 4 diagonals, no fire.
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPRIGHT)
                | _bit(ACTION_UPLEFT)
                | _bit(ACTION_DOWNRIGHT)
                | _bit(ACTION_DOWNLEFT)
            )
        elif self == AtariGame.BEAM_RIDER:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_UPRIGHT)
                | _bit(ACTION_UPLEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif (
            self == AtariGame.BOWLING
            or self == AtariGame.POOYAN
            or self == AtariGame.UP_N_DOWN
        ):
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPFIRE)
                | _bit(ACTION_DOWNFIRE)
            )
        elif (
            self == AtariGame.DEMON_ATTACK
            or self == AtariGame.NAME_THIS_GAME
            or self == AtariGame.AIR_RAID
            or self == AtariGame.CARNIVAL
            or self == AtariGame.GALAXIAN
            or self == AtariGame.SIR_LANCELOT
        ):
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif self == AtariGame.GOPHER:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_UPFIRE)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif self == AtariGame.KUNG_FU_MASTER:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_DOWNRIGHT)
                | _bit(ACTION_DOWNLEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
                | _bit(ACTION_DOWNFIRE)
                | _bit(ACTION_UPRIGHTFIRE)
                | _bit(ACTION_UPLEFTFIRE)
                | _bit(ACTION_DOWNRIGHTFIRE)
                | _bit(ACTION_DOWNLEFTFIRE)
            )
        elif self == AtariGame.PHOENIX:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
                | _bit(ACTION_DOWNFIRE)
            )
        elif self == AtariGame.SKIING:
            return _bit(ACTION_NOOP) | _bit(ACTION_RIGHT) | _bit(ACTION_LEFT)
        elif self == AtariGame.SURROUND:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
            )
        elif (
            self == AtariGame.TIME_PILOT
            or self == AtariGame.WIZARD_OF_WOR
            or self == AtariGame.MR_DO
        ):
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPFIRE)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
                | _bit(ACTION_DOWNFIRE)
            )
        elif self == AtariGame.VIDEO_PINBALL:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPFIRE)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif self == AtariGame.BACKGAMMON:
            # No NOOP: the paddle cursor only acts on input.
            return _bit(ACTION_FIRE) | _bit(ACTION_RIGHT) | _bit(ACTION_LEFT)
        elif self == AtariGame.BLACKJACK or self == AtariGame.CASINO:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_DOWN)
            )
        elif self == AtariGame.FROGGER or self == AtariGame.PACMAN:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
            )
        elif self == AtariGame.JOURNEY_ESCAPE:
            # Full set minus FIRE and UPFIRE.
            return ALL_ACTIONS_MASK & ~(
                _bit(ACTION_FIRE) | _bit(ACTION_UPFIRE)
            )
        elif self == AtariGame.KEYSTONE_KAPERS:
            # Full set minus the four diagonal fires.
            return ALL_ACTIONS_MASK & ~(
                _bit(ACTION_UPRIGHTFIRE)
                | _bit(ACTION_UPLEFTFIRE)
                | _bit(ACTION_DOWNRIGHTFIRE)
                | _bit(ACTION_DOWNLEFTFIRE)
            )
        elif (
            self == AtariGame.OTHELLO
            or self == AtariGame.TIC_TAC_TOE_3D
            or self == AtariGame.VIDEO_CHESS
        ):
            # Cursor games: fire + 4 directions + 4 diagonals.
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPRIGHT)
                | _bit(ACTION_UPLEFT)
                | _bit(ACTION_DOWNRIGHT)
                | _bit(ACTION_DOWNLEFT)
            )
        elif self == AtariGame.TETRIS:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
            )
        elif self == AtariGame.TURMOIL:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPRIGHT)
                | _bit(ACTION_UPLEFT)
                | _bit(ACTION_DOWNRIGHT)
                | _bit(ACTION_DOWNLEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif self == AtariGame.VIDEO_CHECKERS:
            # No NOOP: cursor moves diagonally only.
            return (
                _bit(ACTION_FIRE)
                | _bit(ACTION_UPRIGHT)
                | _bit(ACTION_UPLEFT)
                | _bit(ACTION_DOWNRIGHT)
                | _bit(ACTION_DOWNLEFT)
            )
        elif self == AtariGame.TUTANKHAM:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPFIRE)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        # Full action set: Seaquest, Frostbite, Boxing, Alien, BankHeist,
        # BattleZone, Berzerk, Centipede, ChopperCommand, DarkChambers,
        # ElevatorAction, FishingDerby, Jamesbond, Klax, MontezumaRevenge,
        # Robotank.
        return ALL_ACTIONS_MASK

    def starting_actions(self) -> Tuple[UInt8, Int, UInt8, Int]:
        """(action1, frames1, action2, frames2) to inject right after reset,
        before the agent acts — port of ALE's per-game getStartingActions().
        Some games need an input to leave the title screen (FIRE), DarkChambers
        ignores all input during its ~8 s boot animation, and SirLancelot
        needs console RESET followed by LEFT."""
        if (
            self == AtariGame.ASTERIX
            or self == AtariGame.ENDURO
            or self == AtariGame.GOPHER
            or self == AtariGame.UP_N_DOWN
            or self == AtariGame.YARS_REVENGE
            or self == AtariGame.AIR_RAID
            or self == AtariGame.JOURNEY_ESCAPE
            or self == AtariGame.KABOOM
            or self == AtariGame.LOST_LUGGAGE
            or self == AtariGame.MR_DO
            or self == AtariGame.TURMOIL
        ):
            return (ACTION_FIRE, 1, ACTION_NOOP, 0)
        elif self == AtariGame.BEAM_RIDER:
            return (ACTION_RIGHT, 1, ACTION_NOOP, 0)
        elif self == AtariGame.DARK_CHAMBERS:
            return (ACTION_NOOP, 486, ACTION_NOOP, 0)
        elif (
            self == AtariGame.ELEVATOR_ACTION or self == AtariGame.GRAVITAR
        ):
            return (ACTION_FIRE, 16, ACTION_NOOP, 0)
        elif self == AtariGame.DOUBLE_DUNK:
            return (ACTION_UPFIRE, 1, ACTION_NOOP, 0)
        elif self == AtariGame.PITFALL or self == AtariGame.PRIVATE_EYE:
            return (ACTION_UP, 1, ACTION_NOOP, 0)
        elif self == AtariGame.SKIING:
            return (ACTION_DOWN, 16, ACTION_NOOP, 0)
        elif self == AtariGame.ENTOMBED:
            return (ACTION_FIRE, 1, ACTION_NOOP, 5)
        elif (
            self == AtariGame.KEYSTONE_KAPERS
            or self == AtariGame.LASER_GATES
        ):
            return (ACTION_RESET, 1, ACTION_NOOP, 0)
        elif self == AtariGame.SIR_LANCELOT:
            return (ACTION_RESET, 1, ACTION_LEFT, 1)
        return (ACTION_NOOP, 0, ACTION_NOOP, 0)

    def fire_until(self) -> Int:
        """ALE RAM address that must become nonzero before the agent acts,
        achieved by mashing FIRE (2 frames on / 28 off), or -1 for none.
        Mario Bros sits on its title screen (lives byte $87 == 0, so every
        step would read terminal) and its FIRE polling window is timing-
        sensitive — a single press misses it; mashing until the lives byte
        latches is robust."""
        if self == AtariGame.MARIO_BROS:
            return 0x87
        return -1

    def swap_ports(self) -> Bool:
        """Player 1 uses the RIGHT joystick port (Stella Console.SwapPorts
        property — Wizard of Wor is the only such game in the ALE set)."""
        return self == AtariGame.WIZARD_OF_WOR

    def select_until(self) -> Tuple[Int, Int, Int]:
        """(ALE RAM address, desired value, press frames) for console-SELECT
        game-mode selection at reset, or (-1, 0, 0) for none. Port of ALE's
        DEFAULT-mode setMode path (default = FIRST of getAvailableModes):
        e.g. Surround boots in 2-player mode (RAM $F9 == 0) and ALE presses
        SELECT until $F9 == 1 (single player vs computer), then soft-resets.
        Games that already boot in the default mode skip the loop entirely.
        The press duration matters: ET only registers ~25-frame presses."""
        if self == AtariGame.SURROUND:
            return (0xF9, 1, 2)
        elif self == AtariGame.AIR_RAID:
            return (0xAA, 1, 10)  # modes 1-8, panel opens on first press
        elif self == AtariGame.BACKGAMMON:
            return (0xDC, 3, 1)
        elif self == AtariGame.BASIC_MATH:
            return (0xC5, 5, 2)  # modes {5,6,7,8}
        elif self == AtariGame.CASINO:
            return (0xD4, 0, 2)
        elif self == AtariGame.CROSSBOW:
            return (0x8D, 1, 2)  # byte = mode + 1
        elif self == AtariGame.ET:
            return (0xEA, 1, 25)  # byte = mode + 1; long press needed
        elif self == AtariGame.FLAG_CAPTURE:
            return (0xD6, 8, 2)  # modes {8,9,10} (solo)
        elif self == AtariGame.FROGGER:
            return (0xDD, 1, 2)  # byte = 1 + 2*mode (odd = 1 player)
        elif self == AtariGame.GALAXIAN:
            return (0xB3, 1, 2)  # modes 1-9
        elif self == AtariGame.HANGMAN:
            return (0xEE, 0, 2)
        elif self == AtariGame.HAUNTED_HOUSE:
            return (0xCC, 0, 2)
        # NOTE: no entry for HUMAN_CANNONBALL or PACMAN — their ALE mode
        # bytes ($B6 / $CC) double as in-game score bytes, which read 0
        # after console RESET; a select loop here would scramble the mode.
        # Both carts boot in the default mode already.
        elif self == AtariGame.KING_KONG:
            return (0xEC, 0, 2)  # byte = 2*mode (even = 1 player)
        elif self == AtariGame.LOST_LUGGAGE:
            return (0x94, 1, 2)  # byte = 1 + 3*mode
        elif self == AtariGame.MARIO_BROS or self == AtariGame.MR_DO:
            return (0x80, 0, 5)
        elif self == AtariGame.OTHELLO:
            return (0xDE, 1, 2)  # byte = mode + 1
        elif self == AtariGame.SPACE_WAR:
            return (0xA7, 6, 2)  # modes {6..17}
        elif self == AtariGame.TIC_TAC_TOE_3D:
            return (0x88, 0, 2)
        elif self == AtariGame.TURMOIL:
            return (0xEA, 0, 2)
        elif self == AtariGame.VIDEO_CHECKERS:
            return (0xF6, 1, 1)  # modes 1-9 (+11-19 reverse)
        elif self == AtariGame.VIDEO_CHESS:
            return (0xEA, 0, 1)
        elif self == AtariGame.WORD_ZAPPER:
            return (0xDB, 0, 2)
        return (-1, 0, 0)

    def num_actions(self) -> Int:
        """Size of the minimal action set."""
        var mask = self.action_mask()
        var n = 0
        for i in range(18):
            if (mask & (UInt32(1) << UInt32(i))) != 0:
                n += 1
        return n

    def action(self, action_idx: Int) -> UInt8:
        """Map [0, num_actions) to the idx-th ALE action (ascending id order,
        like ALE's getMinimalActionSet)."""
        var mask = self.action_mask()
        var seen = 0
        for i in range(18):
            if (mask & (UInt32(1) << UInt32(i))) != 0:
                if seen == action_idx:
                    return UInt8(i)
                seen += 1
        return ACTION_NOOP


@fieldwise_init
struct GameSignals(Copyable, ImplicitlyCopyable, Movable):
    """Per-step RL signals extracted from RAM."""

    var score: Int
    var reward: Int
    var lives: Int
    var terminal: Bool


def game_signals(
    game: AtariGame,
    mut state: AtariState,
    prev_score: Int,
) -> GameSignals:
    """Extract (score, reward, lives, terminal) — port of ALE's per-game step().

    `prev_score` is the score after the previous step (0 at episode start);
    reward = score delta with each game's quirks (wrap correction, terminal
    garbage suppression, clamping) applied exactly as in ALE.

    Takes the full AtariState because some games need more than the 128-byte
    RIOT RAM: Klax reads its score from Superchip RAM (state.sc_ram), and
    games with cross-step latches in ALE (ChopperCommand's started flag,
    DarkChambers' last level) persist them in state.game_aux (mut). BeamRider
    reads state.lives (the previous step's lives, set by the caller after
    each call) to filter the blinking lives counter.
    """
    if game == AtariGame.PONG:
        # ALE Pong.cpp: raw bytes, not BCD.
        var cpu = _rb(state.ram, 13)
        var player = _rb(state.ram, 14)
        var score = player - cpu
        return GameSignals(
            score, score - prev_score, 0, cpu == 21 or player == 21
        )

    elif game == AtariGame.BREAKOUT:
        var x = _rb(state.ram, 77)
        var y = _rb(state.ram, 76)
        var score = (x & 0x0F) + 10 * ((x & 0xF0) >> 4) + 100 * (y & 0x0F)
        var lives = _rb(state.ram, 57)
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.SPACE_INVADERS:
        var score = get_decimal_score_2(state.ram, 0xE8, 0xE6)
        var reward = score - prev_score
        if reward < 0:
            # Score wrapped (10000 is the maximum).
            reward = (10000 - prev_score) + score
        var lives = _rb(state.ram, 0xC9)
        var terminal = (_rb(state.ram, 0x98) & 0x80) != 0 or lives == 0
        return GameSignals(score, reward, lives, terminal)

    elif game == AtariGame.MS_PACMAN:
        var score = get_decimal_score_3(state.ram, 0xF8, 0xF9, 0xFA)
        var lives_byte = _rb(state.ram, 0xFB) & 0xF
        var terminal = lives_byte == 0 and _rb(state.ram, 0xA7) == 0x53
        return GameSignals(
            score, score - prev_score, (lives_byte & 0x7) + 1, terminal
        )

    elif game == AtariGame.SEAQUEST:
        var score = get_decimal_score_3(state.ram, 0xBA, 0xB9, 0xB8)
        return GameSignals(
            score,
            score - prev_score,
            _rb(state.ram, 0xBB) + 1,
            _rb(state.ram, 0xA3) != 0,
        )

    elif game == AtariGame.QBERT:
        # Lives byte counts down 2,1,0,0xFF,0xFE (signed); 0xFE = death.
        var b = _rb(state.ram, 0x88)
        var sb = b - 256 if b >= 128 else b
        var lives = sb + 2
        if lives < 0:
            lives = 0
        var terminal = b == 0xFE
        if terminal:
            # ALE: suppress the garbage score on the reset frame.
            return GameSignals(prev_score, 0, lives, True)
        var score = get_decimal_score_3(state.ram, 0xDB, 0xDA, 0xD9)
        return GameSignals(score, score - prev_score, lives, False)

    elif game == AtariGame.ASTEROIDS:
        var score = get_decimal_score_2(state.ram, 0xBE, 0xBD) * 10
        var reward = score - prev_score
        if reward < 0:
            reward += 100000  # score wrap
        var lives = _rb(state.ram, 0xBC) >> 4
        return GameSignals(score, reward, lives, lives == 0)

    elif game == AtariGame.FROSTBITE:
        var score = get_decimal_score_3(state.ram, 0xCA, 0xC9, 0xC8)
        var lives_byte = _rb(state.ram, 0xCC) & 0xF
        var terminal = lives_byte == 0 and (_rb(state.ram, 0xF1) & 0x80) != 0
        return GameSignals(
            score, score - prev_score, lives_byte + 1, terminal
        )

    elif game == AtariGame.FREEWAY:
        var score = get_decimal_score(state.ram, 103)
        var reward = score - prev_score
        if reward < 0:
            reward = 0
        if reward > 1:
            reward = 1
        return GameSignals(score, reward, 0, _rb(state.ram, 22) == 1)

    elif game == AtariGame.BOXING:
        var my_score = get_decimal_score(state.ram, 0x92)
        var oppt_score = get_decimal_score(state.ram, 0x93)
        # 0xC0 = KO sentinel.
        if _rb(state.ram, 0x92) == 0xC0:
            my_score = 100
        if _rb(state.ram, 0x93) == 0xC0:
            oppt_score = 100
        var score = my_score - oppt_score
        var terminal: Bool
        if my_score == 100 or oppt_score == 100:
            terminal = True
        else:
            var minutes = _rb(state.ram, 0x90) >> 4
            var seconds = (_rb(state.ram, 0x91) & 0xF) + (_rb(state.ram, 0x91) >> 4) * 10
            terminal = minutes == 0 and seconds == 0
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.ENDURO:
        var score = 0
        var level = _rb(state.ram, 0xAD)
        if level != 0:
            var cars_passed = get_decimal_score_2(state.ram, 0xAB, 0xAC)
            if level == 1:
                cars_passed = 200 - cars_passed
            else:
                cars_passed = 300 - cars_passed
            if level >= 2:
                # First level has 200 cars; 300 for every level after.
                score = 200 + (level - 2) * 300
            score += cars_passed
        return GameSignals(
            score, score - prev_score, 0, _rb(state.ram, 0xAF) == 0xFF
        )

    elif game == AtariGame.AMIDAR:
        var score = get_decimal_score_3(state.ram, 0xD9, 0xDA, 0xDB)
        var lives_byte = _rb(state.ram, 0xD6)
        return GameSignals(
            score, score - prev_score, lives_byte & 0xF, lives_byte == 0x80
        )

    elif game == AtariGame.ATLANTIS:
        var score = get_decimal_score_3(state.ram, 0xA2, 0xA3, 0xA1) * 100
        var lives = _rb(state.ram, 0xF1)
        if lives == 0xFF:
            # ALE: garbage gets written to 0xA1 on the terminal frame —
            # freeze the score and zero the reward.
            return GameSignals(prev_score, 0, lives, True)
        return GameSignals(score, score - prev_score, lives, False)

    elif game == AtariGame.ALIEN:
        # Digits stored one per byte: 0x80 means blank (0), else byte >> 3.
        var score = 0
        var mult = 1
        for addr in [0x8B, 0x89, 0x87, 0x85, 0x83]:
            var b = _rb(state.ram, addr)
            score += (0 if b == 0x80 else b >> 3) * mult
            mult *= 10
        score *= 10
        var lives = _rb(state.ram, 0xC0) & 0xF
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.ASSAULT:
        var score = get_decimal_score_3(state.ram, 0x82, 0x81, 0x80)
        var lives = _rb(state.ram, 0xE5)
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.ASTERIX:
        var score = get_decimal_score_3(state.ram, 0xE0, 0xDF, 0xDE)
        var lives = _rb(state.ram, 0xD3) & 0xF
        # Cannot wait for lives==0: the player can restart on the very last
        # frame (lives==1, death_counter==1) by holding fire.
        var death_counter = _rb(state.ram, 0xC7)
        var terminal = death_counter == 0x01 and lives == 1
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.BANK_HEIST:
        var score = get_decimal_score_3(state.ram, 0xDA, 0xD9, 0xD8)
        var death_timer = _rb(state.ram, 0xCE)
        var lives = _rb(state.ram, 0xD5)
        var terminal = death_timer == 0x01 and lives == 0
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.BATTLE_ZONE:
        # Score digits use 10 as a blank sentinel.
        var first_val = _rb(state.ram, 0x9D)
        var first_right = first_val & 15
        var first_left = (first_val - first_right) >> 4
        if first_left == 10:
            first_left = 0
        var second_val = _rb(state.ram, 0x9E)
        var second_right = second_val & 15
        var second_left = (second_val - second_right) >> 4
        if second_right == 10:
            second_right = 0
        if second_left == 10:
            second_left = 0
        var score = (
            first_left + 10 * second_right + 100 * second_left
        ) * 1000
        var lives = _rb(state.ram, 0xBA) & 0xF
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.BEAM_RIDER:
        var score = get_decimal_score_3(state.ram, 9, 10, 11)
        # The lives counter blinks during the death animation; only commit
        # a one-life decrease once the animation flag (0x8C) is set.
        var prev_lives = Int(state.lives)
        var new_lives = _rb(state.ram, 0x85) + 1
        var lives = new_lives
        if new_lives == prev_lives - 1 and _rb(state.ram, 0x8C) != 0x01:
            lives = prev_lives
        var terminal = _rb(state.ram, 5) == 255
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.BERZERK:
        var score = get_decimal_score_3(state.ram, 95, 94, 93)
        var lives_byte = _rb(state.ram, 0xDA)
        if lives_byte == 0xFF:
            return GameSignals(score, score - prev_score, 0, True)
        return GameSignals(score, score - prev_score, lives_byte + 1, False)

    elif game == AtariGame.BOWLING:
        var score = get_decimal_score_2(state.ram, 0xA1, 0xA6)
        return GameSignals(
            score, score - prev_score, 0, _rb(state.ram, 0xA4) > 0x10
        )

    elif game == AtariGame.CENTIPEDE:
        var score = get_decimal_score_3(state.ram, 118, 117, 116)
        var reward = score - prev_score
        # ALE HACK: the score sometimes resets before termination.
        if reward < 0:
            reward = 0
        var lives = ((_rb(state.ram, 0xED) >> 4) & 0x7) + 1
        var terminal = (_rb(state.ram, 0xA6) & 0x40) != 0
        return GameSignals(score, reward, lives, terminal)

    elif game == AtariGame.CHOPPER_COMMAND:
        var score = get_decimal_score_2(state.ram, 0xEE, 0xEC) * 100
        var lives = _rb(state.ram, 0xE4) & 0xF
        # 0xC2 bit 0 is 1 once gameplay has started (chopper faces right);
        # latch it so mode-select screens (always facing left) don't read
        # as terminal. ALE keeps m_is_started for the same reason.
        state.game_aux |= Int64(_rb(state.ram, 0xC2) & 0x1)
        var terminal = state.game_aux != 0 and lives == 0
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.CRAZY_CLIMBER:
        # Digits stored one per byte (not BCD).
        var score = (
            _rb(state.ram, 0x82)
            + 10 * _rb(state.ram, 0x83)
            + 100 * _rb(state.ram, 0x84)
            + 1000 * _rb(state.ram, 0x85)
        ) * 100
        var reward = score - prev_score
        if reward < 0:
            reward = 0
        var lives = _rb(state.ram, 0xAA)
        return GameSignals(score, reward, lives, lives == 0)

    elif game == AtariGame.DARK_CHAMBERS:
        # game_aux holds the last seen level: levels only go up; a drop
        # means the game restarted (terminal). Score wrap is also terminal.
        var new_level = _rb(state.ram, 0xD5)
        if new_level < Int(state.game_aux):
            return GameSignals(prev_score, 0, 0, True)
        state.game_aux = Int64(new_level)
        var score = get_decimal_score_2(state.ram, 0xCC, 0xCF) * 10
        if score < prev_score:
            # Exceeded the maximum score.
            return GameSignals(prev_score, 0, 0, True)
        # Low 5 bits are health; the top 3 are item flags.
        var health = _rb(state.ram, 0xCA) & 0x1F
        return GameSignals(score, score - prev_score, health, health == 0)

    elif game == AtariGame.DEMON_ATTACK:
        var score = get_decimal_score_3(state.ram, 0x85, 0x83, 0x81)
        # ALE MGB: the score RAM is not initialized to 0 on boot.
        if (
            _rb(state.ram, 0x81) == 0xAB
            and _rb(state.ram, 0x83) == 0xCD
            and _rb(state.ram, 0x85) == 0xEA
        ):
            score = 0
        var lives_displayed = _rb(state.ram, 0xF2)
        var display_flag = _rb(state.ram, 0xF1)
        var terminal = lives_displayed == 0 and display_flag == 0xBD
        return GameSignals(
            score, score - prev_score, lives_displayed + 1, terminal
        )

    elif game == AtariGame.ELEVATOR_ACTION:
        var score = get_decimal_score_3(state.ram, 0x89, 0x88, 0x87)
        var lives = _rb(state.ram, 0x83)
        # 0x81 == 0 only on the start screen, where lives reads 0 too.
        var terminal = lives == 0 and _rb(state.ram, 0x81) != 0x00
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.FISHING_DERBY:
        var my_score = get_decimal_score(state.ram, 0xBD)
        var oppt_score = get_decimal_score(state.ram, 0xBE)
        var score = my_score - oppt_score
        # Either side reaching 99 (0x99 BCD) ends the game.
        var terminal = (
            _rb(state.ram, 0xBD) == 0x99 or _rb(state.ram, 0xBE) == 0x99
        )
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.JAMESBOND:
        var score = get_decimal_score_3(state.ram, 0xDC, 0xDD, 0xDE)
        var lives_byte = _rb(state.ram, 0x86) & 0xF
        # 0x8C reads 0x68 on death; the system loops back to the start
        # state after a while (where fire starts a new game).
        var terminal = lives_byte == 0 and _rb(state.ram, 0x8C) == 0x68
        return GameSignals(
            score, score - prev_score, lives_byte + 1, terminal
        )

    elif game == AtariGame.KLAX:
        # Score and level live in Superchip RAM (read port $F080-$F0FF →
        # sc_ram[addr - 0x80]); ALE reads them via the full memory map.
        var b_lo = Int(state.sc_ram[0x34])  # $F0B4
        var b_mid = Int(state.sc_ram[0x35])  # $F0B5
        var b_hi = Int(state.sc_ram[0x36])  # $F0B6
        var score = (
            10 * ((b_lo >> 4) & 0xF)
            + (b_lo & 0xF)
            + 1000 * ((b_mid >> 4) & 0xF)
            + 100 * (b_mid & 0xF)
            + 100000 * ((b_hi >> 4) & 0xF)
            + 10000 * (b_hi & 0xF)
        )
        var misses = Int(state.sc_ram[0x6E])  # $F0EE
        var max_misses = Int(state.sc_ram[0x69])  # $F0E9
        var level = Int(state.sc_ram[0x1D])  # $F09D
        var game_active = _rb(state.ram, 0xA8) == 4
        # The 25 bottom blocks live at 0xB3..0xCB; types 0/2/6/10/14 are
        # empty or level-end bonus fillers, not real blocks.
        var num_blocks = 0
        for i in range(25):
            var t = _rb(state.ram, 0xB3 + i)
            if t != 0 and t != 2 and t != 6 and t != 10 and t != 14:
                num_blocks += 1
        var terminal = (
            (max_misses > 0 and misses == max_misses)
            or (game_active and num_blocks == 25)
            or level == 0x99
        )
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.MONTEZUMA_REVENGE:
        var score = get_decimal_score_3(state.ram, 0x95, 0x94, 0x93)
        var new_lives = _rb(state.ram, 0xBA)
        var terminal = new_lives == 0 and _rb(state.ram, 0xFE) == 0x60
        return GameSignals(
            score, score - prev_score, (new_lives & 0x7) + 1, terminal
        )

    elif game == AtariGame.ROBOTANK:
        # Raw counters, not BCD: a squadron is 12 tanks.
        var score = _rb(state.ram, 0xB6) * 12 + _rb(state.ram, 0xB5)
        var lives = _rb(state.ram, 0xA8)
        var terminal = lives == 0 and _rb(state.ram, 0xB4) == 0xFF
        return GameSignals(
            score, score - prev_score, (lives & 0xF) + 1, terminal
        )

    elif game == AtariGame.TUTANKHAM:
        var score = get_decimal_score_2(state.ram, 0x9C, 0x9A)
        var lives_byte = _rb(state.ram, 0x9E)
        # 0x81 is 0x84 when the game is freshly loaded but not yet reset.
        var terminal = lives_byte == 0 and _rb(state.ram, 0x81) != 0x84
        return GameSignals(
            score, score - prev_score, lives_byte & 0x3, terminal
        )

    elif game == AtariGame.DEFENDER:
        # Six digits one per byte at 0x9C-0xA1; 0xA = blank zero.
        var score = 0
        var mult = 1
        for digit in range(6):
            var v = _rb(state.ram, 0x9C + digit) & 0xF
            if v == 0xA:
                v = 0
            score += v * mult
            mult *= 10
        var lives = _rb(state.ram, 0xC2)
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.DOUBLE_DUNK:
        var my_score = get_decimal_score(state.ram, 0xF6)
        var oppt_score = get_decimal_score(state.ram, 0xF7)
        var score = my_score - oppt_score
        var terminal = (my_score >= 24 or oppt_score >= 24) and _rb(
            state.ram, 0xFE
        ) == 0xE7
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.GOPHER:
        var score = get_decimal_score_3(state.ram, 0xB2, 0xB1, 0xB0)
        # Lives = popcount of the 3 carrot bits.
        var carrots = _rb(state.ram, 0xB4) & 0x7
        var lives = (carrots & 1) + ((carrots >> 1) & 1) + (
            (carrots >> 2) & 1
        )
        return GameSignals(score, score - prev_score, lives, carrots == 0)

    elif game == AtariGame.GRAVITAR:
        var score = get_decimal_score_3(state.ram, 9, 8, 7)
        var screen_byte = _rb(state.ram, 0x81)
        # 6 lives on the starting screen, else read from RAM.
        var lives = 6 if screen_byte == 0x0 else _rb(state.ram, 0x84) + 1
        return GameSignals(
            score, score - prev_score, lives, screen_byte == 0x01
        )

    elif game == AtariGame.HERO:
        var score = get_decimal_score_3(state.ram, 0xB9, 0xB8, 0xB7)
        var lives = _rb(state.ram, 0xB3)
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.ICE_HOCKEY:
        var my_score = get_decimal_score(state.ram, 0x8A)
        var oppt_score = get_decimal_score(state.ram, 0x8B)
        var score = my_score - oppt_score
        var reward = score - prev_score
        if reward > 1:
            reward = 1  # ALE clamps to +1
        # Game ends when the clock runs out.
        var terminal = (
            _rb(state.ram, 0x87) == 0 and _rb(state.ram, 0x86) == 0
        )
        return GameSignals(score, reward, 0, terminal)

    elif game == AtariGame.KANGAROO:
        var score = get_decimal_score_2(state.ram, 0xA8, 0xA7) * 100
        var lives_byte = _rb(state.ram, 0xAD)
        return GameSignals(
            score,
            score - prev_score,
            (lives_byte & 0x7) + 1,
            lives_byte == 0xFF,
        )

    elif game == AtariGame.KRULL:
        var score = get_decimal_score_3(state.ram, 0x9E, 0x9D, 0x9C)
        var lives = _rb(state.ram, 0x9F)
        var terminal = (
            lives == 0
            and _rb(state.ram, 0xA2) == 0x03
            and _rb(state.ram, 0x80) == 0x80
        )
        return GameSignals(
            score, score - prev_score, (lives & 0x7) + 1, terminal
        )

    elif game == AtariGame.KUNG_FU_MASTER:
        var score = get_decimal_score_3(state.ram, 0x9A, 0x99, 0x98)
        var lives_byte = _rb(state.ram, 0x9D)
        return GameSignals(
            score,
            score - prev_score,
            (lives_byte & 0x7) + 1,
            lives_byte == 0xFF,
        )

    elif game == AtariGame.NAME_THIS_GAME:
        var score = get_decimal_score_3(state.ram, 0xC6, 0xC5, 0xC4)
        var lives = _rb(state.ram, 0xC7) & 0x7
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.PHOENIX:
        var score = get_decimal_score_2(state.ram, 0xC8, 0xC9) * 10
        score += _rb(state.ram, 0xC7) >> 4
        score *= 10
        var terminal = _rb(state.ram, 0xCC) == 0x80
        var lives = _rb(state.ram, 0xCB) & 0x7
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.PITFALL:
        # Score starts at 2000 and can DECREASE (falls/time penalties);
        # reset_game syncs state.score so the first step doesn't see +2000.
        var score = get_decimal_score_3(state.ram, 0xD7, 0xD6, 0xD5)
        var lives_byte = _rb(state.ram, 0x80) >> 4
        # 0x9E nonzero while the player is uncontrollable (logo screen).
        var terminal = lives_byte == 0 and _rb(state.ram, 0x9E) != 0
        var lives: Int
        if lives_byte == 0xA:
            lives = 3
        elif lives_byte == 0x8:
            lives = 2
        else:
            lives = 1
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.POOYAN:
        var score = get_decimal_score_3(state.ram, 0x8A, 0x89, 0x88)
        var lives_byte = _rb(state.ram, 0x96)
        var terminal = lives_byte == 0x0 and _rb(state.ram, 0x98) == 0x05
        return GameSignals(
            score, score - prev_score, (lives_byte & 0x7) + 1, terminal
        )

    elif game == AtariGame.PRIVATE_EYE:
        var score = get_decimal_score_3(state.ram, 0xCA, 0xC9, 0xC8)
        # Copyright timer: 0x00 while running, 0x01 at game start.
        var t = _rb(state.ram, 0xC2)
        var terminal = t != 0x00 and t != 0x01
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.RIVERRAID:
        # Digits stored as value*8 (0,8,...,72); anything else reads as 0.
        var score = 0
        var mult = 1
        for addr in [87, 85, 83, 81, 79, 77]:
            var v = _rb(state.ram, addr)
            var d = v >> 3
            if (v & 7) != 0 or d > 9:
                d = 0
            score += d * mult
            mult *= 10
        # Terminal = lives byte going 0x59 (last life) -> 0x58 (empty).
        # game_aux carries the previous step's lives byte (ALE m_lives_byte).
        var lives_byte = _rb(state.ram, 0xC0)
        var terminal = lives_byte == 0x58 and Int(state.game_aux) == 0x59
        state.game_aux = Int64(lives_byte)
        var lives: Int
        if lives_byte == 0x58:
            lives = 4  # beginning of episode
        elif lives_byte == 0x59:
            lives = 1
        else:
            lives = lives_byte // 8 + 1
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.ROAD_RUNNER:
        # Four digits one per byte at 0xC9-0xCC; 0xA = blank zero.
        var score = 0
        var mult = 1
        for digit in range(4):
            var v = _rb(state.ram, 0xC9 + digit) & 0xF
            if v == 0xA:
                v = 0
            score += v * mult
            mult *= 10
        score *= 100
        var lives_byte = _rb(state.ram, 0xC4) & 0x7
        var terminal = lives_byte == 0 and (
            _rb(state.ram, 0xB9) != 0 or _rb(state.ram, 0xBD) != 0
        )
        return GameSignals(
            score, score - prev_score, lives_byte + 1, terminal
        )

    elif game == AtariGame.SKIING:
        # Score = elapsed time (counts UP); reward = NEGATIVE time delta.
        var centiseconds = get_decimal_score_2(state.ram, 0xEA, 0xE9)
        var score = _rb(state.ram, 0xE8) * 6000 + centiseconds
        var terminal = _rb(state.ram, 0x91) == 0xFF
        return GameSignals(score, prev_score - score, 0, terminal)

    elif game == AtariGame.SOLARIS:
        # Five digits displayed but six tracked.
        var score = get_decimal_score_3(state.ram, 0xDC, 0xDD, 0xDE) * 10
        var lives_byte = _rb(state.ram, 0xD9)
        return GameSignals(
            score, score - prev_score, lives_byte & 0xF, lives_byte == 0
        )

    elif game == AtariGame.STAR_GUNNER:
        # Four digits one per byte; 10 = blank zero.
        var score = 0
        var mult = 1
        for addr in [0x83, 0x84, 0x85, 0x86]:
            var v = _rb(state.ram, addr) & 0x0F
            if v == 10:
                v = 0
            score += v * mult
            mult *= 10
        score *= 100
        var lives_byte = _rb(state.ram, 0x87)
        # game_aux latches "game started" (lives byte reaches 5) so the
        # title screen's garbage lives byte reads as the initial 5.
        if lives_byte == 0x05:
            state.game_aux = 1
        var lives = (lives_byte & 0xF) if state.game_aux != 0 else 5
        return GameSignals(
            score, score - prev_score, lives, lives_byte == 0
        )

    elif game == AtariGame.SURROUND:
        var their_score = get_decimal_score(state.ram, 0xF6)
        var my_score = get_decimal_score(state.ram, 0xF7)
        var score = my_score - their_score
        var terminal = their_score == 10 or my_score == 10
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.TENNIS:
        # Reward: set-point delta if it changed this step, else game-score
        # delta. game_aux packs the previous deltas as two signed bytes
        # (low = score, high = points); 0 decodes to (0, 0) = reset state.
        var my_score = _rb(state.ram, 0xC5)
        var oppt_score = _rb(state.ram, 0xC6)
        var my_points = _rb(state.ram, 0xC7)
        var oppt_points = _rb(state.ram, 0xC8)
        var delta_score = my_score - oppt_score
        var delta_points = my_points - oppt_points
        var aux = Int(state.game_aux)
        var prev_ds = aux & 0xFF
        if prev_ds >= 128:
            prev_ds -= 256
        var prev_dp = (aux >> 8) & 0xFF
        if prev_dp >= 128:
            prev_dp -= 256
        var reward = 0
        if prev_dp != delta_points:
            reward = delta_points - prev_dp
        elif prev_ds != delta_score:
            reward = delta_score - prev_ds
        state.game_aux = Int64(
            ((delta_points & 0xFF) << 8) | (delta_score & 0xFF)
        )
        var terminal = (
            (my_points >= 6 and delta_points >= 2)
            or (oppt_points >= 6 and -delta_points >= 2)
            or my_points == 7
            or oppt_points == 7
        )
        return GameSignals(
            prev_score + reward, reward, 0, terminal
        )

    elif game == AtariGame.TIME_PILOT:
        var score = get_decimal_score_2(state.ram, 0x8D, 0x8F) * 100
        var terminal = _rb(state.ram, 0xA0) != 0
        # Only trust the lives byte while actually flying (screen 2);
        # game_aux carries the last good value across other screens.
        if (_rb(state.ram, 0x80) & 0xF) == 2:
            state.game_aux = Int64((_rb(state.ram, 0x8B) & 0x7) + 1)
        return GameSignals(
            score, score - prev_score, Int(state.game_aux), terminal
        )

    elif game == AtariGame.UP_N_DOWN:
        var score = get_decimal_score_3(state.ram, 0x82, 0x81, 0x80)
        var lives_byte = _rb(state.ram, 0x86) & 0xF
        var terminal = _rb(state.ram, 0x94) > 0x40 and lives_byte == 0
        return GameSignals(
            score, score - prev_score, lives_byte + 1, terminal
        )

    elif game == AtariGame.VENTURE:
        var score = get_decimal_score_2(state.ram, 0xC8, 0xC7) * 100
        var lives_byte = _rb(state.ram, 0xC6)
        var terminal = (
            lives_byte == 0
            and _rb(state.ram, 0xCD) == 0xFF
            and (_rb(state.ram, 0xBF) & 0x80) != 0
        )
        return GameSignals(
            score, score - prev_score, (lives_byte & 0x7) + 1, terminal
        )

    elif game == AtariGame.VIDEO_PINBALL:
        var score = get_decimal_score_3(state.ram, 0xB0, 0xB2, 0xB4)
        var terminal = (_rb(state.ram, 0xAF) & 0x1) != 0
        # Lives display as ball number; extra-ball flag adds one.
        var lives = (
            4 + (_rb(state.ram, 0xA8) & 0x1) - (_rb(state.ram, 0x99) & 0x7)
        )
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.WIZARD_OF_WOR:
        var score = get_decimal_score_2(state.ram, 0x86, 0x88)
        if score >= 8000:
            score -= 8000  # ALE MGB: score does not go beyond 999
        score *= 100
        var new_lives = _rb(state.ram, 0x8D) & 15
        var terminal = new_lives == 0 and _rb(state.ram, 0xF4) == 0xF8
        # Lives drop when entering the play field; only commit the value
        # while waiting (0xD7 bit 0 clear). game_aux carries it across.
        if (_rb(state.ram, 0xD7) & 0x1) == 0:
            state.game_aux = Int64(new_lives)
        return GameSignals(
            score, score - prev_score, Int(state.game_aux), terminal
        )

    elif game == AtariGame.YARS_REVENGE:
        var score = get_decimal_score_3(state.ram, 0xE2, 0xE1, 0xE0)
        var lives = _rb(state.ram, 0x9E) >> 4
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.ZAXXON:
        var score = get_decimal_score_2(state.ram, 0xE9, 0xE8) * 100
        # Lives read 0 before console RESET is pushed; our reset holds it.
        var lives = _rb(state.ram, 0xEA) & 0x7
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.ADVENTURE:
        # +1 for bringing the chalice to the yellow castle; terminal on win
        # or on being eaten by a dragon.
        var won = _rb(state.ram, 0xB9) == 0x12
        var eaten = _rb(state.ram, 0xE0) == 2
        var reward = 1 if won else 0
        return GameSignals(prev_score + reward, reward, 0, won or eaten)

    elif game == AtariGame.AIR_RAID:
        var score = get_decimal_score_3(state.ram, 0xAA, 0xA9, 0xA8)
        return GameSignals(
            score, score - prev_score, 0, _rb(state.ram, 0xA7) == 0xFF
        )

    elif game == AtariGame.ATLANTIS2:
        var lives = _rb(state.ram, 0xF1)
        if lives == 0xFF:
            # Ignore the garbage terminal score.
            return GameSignals(prev_score, 0, lives, True)
        var score = get_decimal_score_3(state.ram, 0xA1, 0xA2, 0xA3)
        return GameSignals(score, score - prev_score, lives, False)

    elif game == AtariGame.BACKGAMMON:
        # Player pieces are negative counts, computer pieces positive. The
        # computer scratches this RAM during its turn — only act on a board
        # where both sides total 15 pieces.
        var player_out = -_sb(state.ram, 0x80)
        var computer_out = _sb(state.ram, 0x8E)
        var player_in = 0
        var computer_in = 0
        for addr in range(0x81, 0x8E):
            var p = _sb(state.ram, addr)
            if p > 0:
                computer_in += p
            elif p < 0:
                player_in += -p
        for addr in range(0x8F, 0x9C):
            var p = _sb(state.ram, addr)
            if p > 0:
                computer_in += p
            elif p < 0:
                player_in += -p
        var valid = (
            computer_in + computer_out == 15
            and player_in + player_out == 15
        )
        if valid and player_out == 15:
            return GameSignals(prev_score + 1, 1, 0, True)
        elif valid and computer_out == 15:
            return GameSignals(prev_score - 1, -1, 0, True)
        return GameSignals(prev_score, 0, 0, False)

    elif game == AtariGame.BASIC_MATH:
        var score = get_decimal_score(state.ram, 0x84)
        # 10 rounds; 0x86 != 0 only on the final score screen (waiting for
        # round 10's point before terminating).
        var terminal = get_decimal_score(state.ram, 0x86) != 0
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.BLACKJACK:
        # Chip count; '0bbb' display means bust.
        var bust = (
            _rb(state.ram, 0x86) == 0x0B and _rb(state.ram, 0x89) == 0xBB
        )
        var score = 0 if bust else get_decimal_score_2(
            state.ram, 0x89, 0x86
        )
        var terminal = bust or score >= 1000
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.CARNIVAL:
        var score = get_decimal_score_2(state.ram, 0xAE, 0xAD) * 10
        return GameSignals(
            score, score - prev_score, 0, _rb(state.ram, 0x83) < 1
        )

    elif game == AtariGame.CASINO:
        var score = get_decimal_score_2(state.ram, 0x95, 0x8C)
        var mode = _rb(state.ram, 0xD4)
        if mode == 3:
            # Poker Solitaire: ends once all 25 cards placed and awarded.
            var finished = _rb(state.ram, 0x9E) == 0xAA
            return GameSignals(
                score, score - prev_score, 0, score > 0 and finished
            )
        # Blackjack / Stud Poker: bust or break-the-bank (input disabled).
        var input_disabled = (_rb(state.ram, 0xD3) & 0x80) != 0
        var reward = 0 if input_disabled else score - prev_score
        var bet = get_decimal_score(state.ram, 0x9E)
        var terminal = score == 0 or (bet > 0 and input_disabled)
        return GameSignals(score, reward, 0, terminal)

    elif game == AtariGame.CROSSBOW:
        var score = get_decimal_score_3(state.ram, 0x8D, 0x8C, 0x8B)
        # 0xE7: 0x80 front end, 0x81 level select, 0x00 in-game, 0x82 over.
        return GameSignals(
            score, score - prev_score, 0, _rb(state.ram, 0xE7) == 0x82
        )

    elif game == AtariGame.DONKEY_KONG:
        var score = get_decimal_score_2(state.ram, 0x88, 0x87) * 100
        var lives = _rb(state.ram, 0xA3)
        var terminal = (
            lives == 0
            and _rb(state.ram, 0x8F) == 0x03
            and _rb(state.ram, 0x8B) == 0x1F
        )
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.EARTHWORLD:
        # Clue number as score; finding the 10th clue wins.
        var score = get_decimal_score(state.ram, 0xA7)
        return GameSignals(score, score - prev_score, 0, score == 10)

    elif game == AtariGame.ENTOMBED:
        # Score is plain hex, not BCD.
        var score = _rb(state.ram, 0xE3)
        var lives = _rb(state.ram, 0xC7) & 0x03
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.ET:
        var score = get_decimal_score_3(state.ram, 0xE1, 0xE0, 0xDF)
        # The lives counter wraps to 0xFF when none remain.
        var lives = (_rb(state.ram, 0xE5) + 1) & 0xFF
        var terminal = lives == 0 and _rb(state.ram, 0x80) == 8
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.FLAG_CAPTURE:
        var score = get_decimal_score(state.ram, 0xEA)
        # 75-second timer at 0xEB.
        var terminal = get_decimal_score(state.ram, 0xEB) == 0
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.FROGGER:
        var score = get_decimal_score_2(state.ram, 0xCE, 0xCC)
        var lives_byte = _rb(state.ram, 0xD0)
        return GameSignals(
            score, score - prev_score, lives_byte, lives_byte == 0xFF
        )

    elif game == AtariGame.GALAXIAN:
        var score = get_decimal_score_3(state.ram, 0xAE, 0xAD, 0xAC)
        var reward = score - prev_score
        if reward < 0:
            # ALE treats any drop as a 1,000,000 wrap; in our trajectories
            # the score RAM also clears transiently on game-over/restart,
            # which would leak a bogus ~+1M reward — only wrap when the
            # previous score was actually near the maximum.
            if prev_score > 990000:
                reward = (1000000 - prev_score) + score
            else:
                reward = 0
        var terminal = (_rb(state.ram, 0xBF) & 0x80) != 0
        var lives = 0 if terminal else _rb(state.ram, 0xB9) + 1
        return GameSignals(score, reward, lives, terminal)

    elif game == AtariGame.HANGMAN:
        # +1 per word guessed, -1 per word failed; the timer wrapping
        # 0xFF -> 0x00 ends the game. game_aux packs (playerScore,
        # computerScore, prev timer, prev-prev timer).
        var aux = Int(state.game_aux)
        var prev_player = aux & 0x7F
        var prev_computer = (aux >> 7) & 0x7F
        var timer_prev = (aux >> 14) & 0xFF
        var player = get_decimal_score(state.ram, 0xEB)
        var computer = get_decimal_score(state.ram, 0xEC)
        var reward = (player - prev_player) - (computer - prev_computer)
        var timer_now = _rb(state.ram, 0xF1)
        var timed_out = timer_now == 0 and timer_prev == 255
        state.game_aux = Int64(
            (timer_now << 14) | (computer << 7) | player
        )
        return GameSignals(
            prev_score + reward, reward, 0, reward != 0 or timed_out
        )

    elif game == AtariGame.HAUNTED_HOUSE:
        # -1 per match used (match counter wraps 99->0 but any change means
        # one was used); +100 for escaping with the urn. game_aux = last
        # match count.
        var matches = get_decimal_score(state.ram, 0x82)
        var reward = 0
        if matches != Int(state.game_aux):
            reward -= 1
            state.game_aux = Int64(matches)
        var lives = _rb(state.ram, 0x96)
        var escaped = _rb(state.ram, 0x99) == 0x44
        if escaped:
            reward += 100
        return GameSignals(
            prev_score + reward, reward, lives, lives == 0 or escaped
        )

    elif game == AtariGame.HUMAN_CANNONBALL:
        var score = get_decimal_score(state.ram, 0xB6)
        var misses = get_decimal_score(state.ram, 0xB7)
        return GameSignals(
            score, score - prev_score, 0, score == 7 or misses == 7
        )

    elif game == AtariGame.JOURNEY_ESCAPE:
        var score = get_decimal_score_3(state.ram, 0x92, 0x91, 0x90)
        var reward = score - prev_score
        if reward == 50000:
            reward = 0  # ALE HACK: ignore the starting cash
        var terminal = (
            _rb(state.ram, 0x95) == 0 and _rb(state.ram, 0x96) == 0
        )
        return GameSignals(score, reward, 0, terminal)

    elif game == AtariGame.KABOOM:
        var score = get_decimal_score_3(state.ram, 0xA5, 0xA4, 0xA3)
        var lives = _rb(state.ram, 0xA1)
        var terminal = lives == 0x0 or score == 999999
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.KEYSTONE_KAPERS:
        var score = get_decimal_score_2(state.ram, 0x9C, 0x9B)
        var lives = _rb(state.ram, 0x96)
        var terminal = lives == 0 and _rb(state.ram, 0x88) == 0x00
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.KING_KONG:
        var score = get_decimal_score_2(state.ram, 0x83, 0x82)
        var lives = _rb(state.ram, 0xEE)
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.KOOLAID:
        var score = get_decimal_score_2(state.ram, 0x81, 0x80) * 100
        return GameSignals(
            score, score - prev_score, 0, _rb(state.ram, 0xD1) == 0x80
        )

    elif game == AtariGame.LASER_GATES:
        var score = get_decimal_score_3(state.ram, 0x82, 0x81, 0x80)
        return GameSignals(
            score, score - prev_score, 0, _rb(state.ram, 0x83) == 0x00
        )

    elif game == AtariGame.LOST_LUGGAGE:
        var score = get_decimal_score_3(state.ram, 0x96, 0x95, 0x94)
        var lives = _rb(state.ram, 0xCA)
        var terminal = (
            lives == 0
            and _rb(state.ram, 0xC8) == 0x0A
            and _rb(state.ram, 0xA5) == 0x00
            and _rb(state.ram, 0xA9) == 0x00
        )
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.MARIO_BROS:
        var score = get_decimal_score_2(state.ram, 0x8A, 0x89) * 100
        var lives = _rb(state.ram, 0x87)
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.MINIATURE_GOLF:
        # Reward = (par - hits) per completed level; the left status shows
        # cumulative hits (wraps at 99) during play and the level number in
        # the lobby, where the right status shows the level par. game_aux
        # packs (levelNumber, levelPar, leftStatus, hits, hitsAtStartOfLevel).
        var aux = Int(state.game_aux)
        var level_number = aux & 0xF
        var level_par = (aux >> 4) & 0x7F
        var left_status_prev = (aux >> 11) & 0x7F
        var hits = (aux >> 18) & 0xFFFF
        var hits_at_start = (aux >> 34) & 0xFFFF
        var left_status = get_decimal_score(state.ram, 0x87)
        var right_status = get_decimal_score(state.ram, 0x88)
        var new_level = get_decimal_score(state.ram, 0xAF)
        var reward = 0
        var terminal = False
        if new_level != level_number:
            # Level just completed.
            var total_hits = left_status_prev + hits
            var level_hits = total_hits - hits_at_start
            if level_hits > 0:
                reward = level_par - level_hits
            if new_level == 0:
                terminal = True
            level_number = new_level
            hits = 0
            hits_at_start = left_status_prev
        if right_status != 0:
            # Lobby mode: right status displays the level par.
            level_par = right_status
        else:
            if left_status < left_status_prev:
                hits += left_status_prev  # left status wrapped at 99
            left_status_prev = left_status
        state.game_aux = Int64(
            (hits_at_start << 34)
            | (hits << 18)
            | (left_status_prev << 11)
            | (level_par << 4)
            | level_number
        )
        return GameSignals(prev_score + reward, reward, 0, terminal)

    elif game == AtariGame.MR_DO:
        var score = get_decimal_score_2(state.ram, 0x82, 0x83) * 10
        var lives = _rb(state.ram, 0xDB)
        return GameSignals(
            score, score - prev_score, lives, _rb(state.ram, 0xDA) == 0x40
        )

    elif game == AtariGame.OTHELLO:
        var score = get_decimal_score(state.ram, 0xCE) - get_decimal_score(
            state.ram, 0xD0
        )
        # Turn byte 0xC0 is 0 (only persistently) once the game is over;
        # game_aux counts consecutive no-input steps.
        if _rb(state.ram, 0xC0) == 0:
            state.game_aux += 1
        else:
            state.game_aux = 0
        return GameSignals(
            score, score - prev_score, 0, Int(state.game_aux) > 50
        )

    elif game == AtariGame.PACMAN:
        var score = get_decimal_score_3(state.ram, 0xCC, 0xCE, 0xD0)
        var lives = _rb(state.ram, 0x98) + 1
        var terminal = lives == 1 and _rb(state.ram, 0xE4) == 0x3F
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.SIR_LANCELOT:
        var score = get_decimal_score_3(state.ram, 0xA0, 0x9F, 0x9E)
        var lives = _rb(state.ram, 0xA9)
        var terminal = lives == 0 and _rb(state.ram, 0xA7) == 0xA0
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.SPACE_WAR:
        var score = get_decimal_score(state.ram, 0xA7)
        # First to 10 points, or the 10-minute timer (0x80) wrapping to 0.
        var terminal = score == 10 or _rb(state.ram, 0x80) == 0
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.SUPERMAN:
        # Reward on completion only, proportional to speed: max time minus
        # elapsed in-game time. Ends when entering the Daily Bugle as Clark
        # Kent.
        var seconds = get_decimal_score(state.ram, 0xE2)
        var minutes = get_decimal_score(state.ram, 0xE3)
        var room = _rb(state.ram, 0x80) + (_rb(state.ram, 0x81) << 8)
        var is_clark = (_rb(state.ram, 0x9F) & 0x40) != 0
        var terminal = is_clark and room == 0xF2AC
        var reward = 0
        if terminal:
            reward = (99 * 60 + 59) - (minutes * 60 + seconds)
        return GameSignals(prev_score + reward, reward, 0, terminal)

    elif game == AtariGame.TETRIS:
        var score = get_decimal_score_2(state.ram, 0x71, 0x72)
        var reward = score - prev_score
        if reward < 0:
            reward = 0
        state.game_aux = 1  # started latch (ALE m_started)
        var terminal = (
            state.game_aux != 0 and (_rb(state.ram, 0x73) & 0x80) != 0
        )
        return GameSignals(score, reward, 0, terminal)

    elif game == AtariGame.TIC_TAC_TOE_3D:
        # Win/loss: return address 0xF310 on the stack-top bytes; winner in
        # 0xE1 (0x08 = X = us). Draw: the game pauses with a full grid.
        if (
            _rb(state.ram, 0xFF) == 0xF3 and _rb(state.ram, 0xFE) == 0x10
        ):
            var reward = 1 if _rb(state.ram, 0xE1) == 0x08 else -1
            return GameSignals(prev_score + reward, reward, 0, True)
        var full = True
        for i in range(0x9A, 0xDA):
            if _rb(state.ram, i) == 0:
                full = False
                break
        return GameSignals(prev_score, 0, 0, full)

    elif game == AtariGame.TRONDEAD:
        var score = get_decimal_score_3(state.ram, 0xBF, 0xBE, 0xBD)
        var hits = _rb(state.ram, 0xC8)
        return GameSignals(
            score, score - prev_score, 5 - hits, hits == 5
        )

    elif game == AtariGame.TURMOIL:
        var score = (
            get_decimal_score_2(state.ram, 0x89, 0x8A) + _rb(state.ram, 0xD3)
        ) * 10
        var lives = _rb(state.ram, 0xB9)
        var terminal = lives == 0 and _rb(state.ram, 0xC5) == 0x01
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.VIDEO_CHECKERS:
        # Count both sides' pieces across the four board quadrants; a side
        # reaching zero ends the game (default mode = normal checkers, we
        # play black: black gone = -1, white gone = +1).
        var num_black = 0
        var num_white = 0
        var quad_starts = [0x80, 0x89, 0x92, 0x9B]
        for q in range(4):
            for off in range(8):
                var v = _rb(state.ram, quad_starts[q] + off)
                if v == 0x10 or v == 0x20:
                    num_black += 1
                elif v == 0x90 or v == 0xA0:
                    num_white += 1
        if num_black == 0:
            return GameSignals(prev_score - 1, -1, 0, True)
        elif num_white == 0:
            return GameSignals(prev_score + 1, 1, 0, True)
        return GameSignals(prev_score, 0, 0, False)

    elif game == AtariGame.VIDEO_CHESS:
        # Only score during white's turn (0xE1 == 0x82): the Atari AI
        # simulates moves while searching, scrambling RAM.
        if _rb(state.ram, 0xE1) == 0x82:
            var checkmate = _rb(state.ram, 0xEE)
            if checkmate == 0x00:  # checkmate black
                return GameSignals(prev_score + 1, 1, 0, True)
            elif checkmate == 0x01:  # checkmate white
                return GameSignals(prev_score - 1, -1, 0, True)
        return GameSignals(prev_score, 0, 0, False)

    elif game == AtariGame.VIDEO_CUBE:
        # Reward = newly completed faces (6 faces x 9 blocks from 0xA0);
        # -1 and terminal when the game timer wraps 0xFF -> 0x00. game_aux
        # packs (prev timer, prev-prev timer, face count).
        var complete = 0
        for face in range(6):
            var base = 0xA0 + face * 9
            var first = _rb(state.ram, base)
            var all_match = True
            for off in range(1, 9):
                if _rb(state.ram, base + off) != first:
                    all_match = False
                    break
            if all_match:
                complete += 1
        var aux = Int(state.game_aux)
        var timer_prev = aux & 0xFF
        var prev_faces = (aux >> 16) & 0xF
        var timer_now = _rb(state.ram, 0xDB)
        var timed_out = timer_now == 0 and timer_prev == 255
        state.game_aux = Int64((complete << 16) | timer_now)
        var reward = -1 if timed_out else complete - prev_faces
        return GameSignals(
            prev_score + reward, reward, 0, timed_out or complete == 6
        )

    elif game == AtariGame.WORD_ZAPPER:
        # Game state only valid while the wall clock runs. Score = rounds
        # completed (byte counts down 2..0, wrapping to 0xFF after round 3 —
        # read as signed). game_aux latches "round timer was seen running"
        # so the title screen (timer 0, wall clock ticking) doesn't read as
        # an immediate terminal every step.
        var wall_clock = get_decimal_score(state.ram, 0xCF)
        if wall_clock > 0:
            var score = 2 - _sb(state.ram, 0xDC)
            var time_remaining = get_decimal_score(state.ram, 0xDE)
            if time_remaining > 0:
                state.game_aux = 1
            var terminal = score == 3 or (
                time_remaining == 0 and state.game_aux != 0
            )
            return GameSignals(score, score - prev_score, 0, terminal)
        return GameSignals(prev_score, 0, 0, False)

    return GameSignals(0, 0, 0, False)
