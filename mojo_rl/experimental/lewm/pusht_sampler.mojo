"""LewmPushTSampler — re-export shim.

The struct has been renamed and moved to
``mojo_rl.envs.pusht.offline_sampler`` as ``PushTOfflineSampler``. This
module remains for source compatibility during the strangler migration:
existing imports of the form

    from mojo_rl.experimental.lewm.pusht_sampler import LewmPushTSampler

continue to resolve through this shim. New code should import
``PushTOfflineSampler`` from ``mojo_rl.envs.pusht.offline_sampler``
directly.

See ``docs/PLANNERS_PACKAGE.md`` Phase 1.
"""

from mojo_rl.envs.pusht.offline_sampler import (
    PushTOfflineSampler as LewmPushTSampler,
)
