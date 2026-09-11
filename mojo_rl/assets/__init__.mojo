# +--------------------------------------------------------------------------+ #
# | Asset packs — env meshes and textures, out of the git index
# +--------------------------------------------------------------------------+ #
"""Declare, fetch and materialise versioned asset packs.

    from mojo_rl.assets import Pack, load_packs, pull_pack

`docs/PROJECT_LAYER_PLAN.md` §9. ⚠ THE INFRASTRUCTURE EXISTS BEFORE THE
MIGRATION, deliberately: nothing has been moved out of git yet, and that is
what makes moving it a reversible decision rather than a commitment.
"""

from .pack import (
    PROVIDER_HF,
    PROVIDER_HTTPS,
    PROVIDER_NOEIRA,
    Pack,
    PackFile,
    load_packs,
    parse_packs,
)
from .resolve import (
    cache_root,
    materialise,
    pack_status,
    pull_pack,
    resolve_url,
)
