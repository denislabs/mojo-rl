"""Polyak slow-value sync for a Module (storage-native).

`slowvalue ← (1-rate)·slowvalue + rate·value`. src and dst share the SAME
module type → the storage `Module.polyak_from` recurses params in identical
order and applies `p_dst = tau·p_src + (1-tau)·p_dst` per param (tau=rate),
which is exactly the legacy mix (no name matching, no index-keyed collect).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module


def polyak_module[
    target: StaticString, V: Module
](
    mut src: V,
    mut dst: V,
    rate: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Soft-update `dst` toward `src`: `dst = (1-rate)·dst + rate·src`."""
    dst.polyak_from[target](src, rate, ctx)
