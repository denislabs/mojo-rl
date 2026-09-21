# +--------------------------------------------------------------------------+ #
# | STS3215 control table
# +--------------------------------------------------------------------------+ #
"""Register addresses and widths for the STS/SMS series (the SO-101's servo).

Transcribed from `lerobot/motors/feetech/tables.py`
(`STS_SMS_SERIES_CONTROL_TABLE`), which is the table lerobot itself drives
these arms with. Each entry is `(address, width_in_bytes)` folded into two
`comptime`s, because Mojo has no compile-time tuple indexing worth the
indirection here.

⚠ **Three registers are SIGN-MAGNITUDE, not two's complement**, with the
direction bit at the width's top bit minus one. `Homing_Offset` (bit 11) is
the one that bites immediately — the follower's real offsets include -430 and
-485, which read as 3666 and 3611 under two's complement. `Present_Position`
and `Goal_Position` carry a sign bit at 15: harmless while a joint sits inside
0..4095, and NOT harmless once `Present_Position = Actual − Homing_Offset`
goes negative. `SIGN_BIT_*` below is what the bus consults.
"""

# ── EEPROM ────────────────────────────────────────────────────────────────
comptime STS_FIRMWARE_MAJOR = 0
comptime STS_FIRMWARE_MINOR = 1
comptime STS_MODEL_NUMBER = 3
comptime STS_MODEL_NUMBER_SIZE = 2
comptime STS_ID = 5
comptime STS_BAUD_RATE = 6
comptime STS_RETURN_DELAY_TIME = 7
comptime STS_RESPONSE_STATUS_LEVEL = 8
comptime STS_MIN_POSITION_LIMIT = 9
comptime STS_MAX_POSITION_LIMIT = 11
comptime STS_MAX_TEMPERATURE_LIMIT = 13
comptime STS_MAX_VOLTAGE_LIMIT = 14
comptime STS_MIN_VOLTAGE_LIMIT = 15
comptime STS_MAX_TORQUE_LIMIT = 16
comptime STS_PHASE = 18
comptime STS_UNLOADING_CONDITION = 19
comptime STS_LED_ALARM_CONDITION = 20
comptime STS_P_COEFFICIENT = 21
comptime STS_D_COEFFICIENT = 22
comptime STS_I_COEFFICIENT = 23
comptime STS_MINIMUM_STARTUP_FORCE = 24
comptime STS_PROTECTION_CURRENT = 28
comptime STS_ANGULAR_RESOLUTION = 30
comptime STS_HOMING_OFFSET = 31
comptime STS_OPERATING_MODE = 33
comptime STS_PROTECTIVE_TORQUE = 34
comptime STS_PROTECTION_TIME = 35
comptime STS_OVERLOAD_TORQUE = 36

# ── SRAM ──────────────────────────────────────────────────────────────────
comptime STS_TORQUE_ENABLE = 40
comptime STS_ACCELERATION = 41
comptime STS_GOAL_POSITION = 42
comptime STS_GOAL_TIME = 44
comptime STS_GOAL_VELOCITY = 46
comptime STS_TORQUE_LIMIT = 48
comptime STS_LOCK = 55
comptime STS_PRESENT_POSITION = 56
comptime STS_PRESENT_VELOCITY = 58
comptime STS_PRESENT_LOAD = 60
comptime STS_PRESENT_VOLTAGE = 62
comptime STS_PRESENT_TEMPERATURE = 63
comptime STS_STATUS = 65
comptime STS_MOVING = 66
comptime STS_PRESENT_CURRENT = 69

# ── widths ────────────────────────────────────────────────────────────────
comptime SIZE_1 = 1
comptime SIZE_2 = 2

# ── sign-magnitude direction bits (0 = plain unsigned) ────────────────────
comptime SIGN_BIT_HOMING_OFFSET = 11
comptime SIGN_BIT_POSITION = 15
comptime SIGN_BIT_VELOCITY = 15
comptime SIGN_BIT_LOAD = 10

# ── model facts ───────────────────────────────────────────────────────────
comptime STS_RESOLUTION = 4096
"""Encoder counts per turn. `TICKS_PER_RADIAN = (4096 - 1) / 2pi` — note the
MINUS ONE: lerobot's `normalization.py` maps the *inclusive* range 0..4095
onto a full turn, and an off-by-one here is a ~0.09 degree systematic error
that shows up as a constant sim-to-real offset."""

comptime STS_MODEL_STS3215 = 777
comptime STS_DEFAULT_BAUD = 1000000
comptime STS_PROTOCOL = 0

# ── Operating_Mode values ─────────────────────────────────────────────────
comptime MODE_POSITION = 0
comptime MODE_VELOCITY = 1
comptime MODE_PWM = 2
comptime MODE_STEP = 3

comptime TORQUE_DISABLED = 0
comptime TORQUE_ENABLED = 1


def sign_bit_for(addr: Int) -> Int:
    """Direction-bit index for a register, or 0 if it is plain unsigned.

    One function rather than a check at each call site: the reason
    `Homing_Offset` was ever decoded wrongly anywhere is that the rule lived
    inline in one place and not the other.
    """
    if addr == STS_HOMING_OFFSET:
        return SIGN_BIT_HOMING_OFFSET
    if addr == STS_PRESENT_POSITION or addr == STS_GOAL_POSITION:
        return SIGN_BIT_POSITION
    if addr == STS_PRESENT_VELOCITY or addr == STS_GOAL_VELOCITY:
        return SIGN_BIT_VELOCITY
    if addr == STS_PRESENT_LOAD:
        return SIGN_BIT_LOAD
    return 0
