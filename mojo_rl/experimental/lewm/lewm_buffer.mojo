"""LeWMBuffer — re-export shim.

The trait has been renamed and promoted to ``mojo_rl.core.offline_buffer``
as ``OfflineBuffer`` (it has no LeWM-specific semantics — any pixel-obs
offline buffer can conform). This module remains for source compatibility
during the strangler migration: existing imports of the form

    from mojo_rl.experimental.lewm.lewm_buffer import LeWMBuffer

continue to resolve through this shim. New code should import
``OfflineBuffer`` from ``mojo_rl.core`` directly.

See ``docs/PLANNERS_PACKAGE.md`` Phase 1.
"""

from mojo_rl.core.offline_buffer import OfflineBuffer as LeWMBuffer
