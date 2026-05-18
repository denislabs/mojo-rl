"""PongBuffer — re-export shim.

The struct has been renamed and moved to
``mojo_rl.envs.arcade_games.pong.offline_buffer`` as ``PongOfflineBuffer``.
This module remains for source compatibility during the strangler
migration: existing imports of the form

    from mojo_rl.experimental.lewm.pong_buffer import (
        PongBuffer, PONG_FRAME_BYTES, PONG_NUM_ACTIONS, ...
    )

continue to resolve through this shim. New code should import the
canonical names from ``mojo_rl.envs.arcade_games.pong.offline_buffer``.

See ``docs/PLANNERS_PACKAGE.md`` Phase 1.
"""

from mojo_rl.envs.arcade_games.pong.offline_buffer import (
    PongOfflineBuffer as PongBuffer,
    PONG_OBS_C,
    PONG_OBS_H,
    PONG_OBS_W,
    PONG_NUM_ACTIONS,
    PONG_FRAME_BYTES,
    PONG_OBS_DIM,
    PONG_BUFFER_MAGIC,
    PONG_BUFFER_VERSION,
    PONG_BUFFER_HEADER_BYTES,
)
