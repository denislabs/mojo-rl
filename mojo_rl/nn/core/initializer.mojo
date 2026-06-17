"""Initializer trait — host-side weight + bias fill for parameterized layers.

Trait method signatures take an `UnsafePointer<Scalar[DT]>` rather than a
List, because GPU `make[INIT]` calls them on a HostBuffer's pointer (which
isn't List-backed). Leaf modules pass their CPU `List`'s `unsafe_ptr()` or
the host-mirror's `unsafe_ptr()` uniformly.

`init_bias` has no default; concrete inits pick (usually zero, sometimes a
small positive constant).
"""

from ..constants import DT


trait Initializer:
    @staticmethod
    def init_weight[
        buf_origin: Origin[mut=True]
    ](
        buf: UnsafePointer[Scalar[DT], buf_origin],
        n_elems: Int,
        fan_in: Int,
        fan_out: Int,
    ):
        ...

    @staticmethod
    def init_bias[
        buf_origin: Origin[mut=True]
    ](
        buf: UnsafePointer[Scalar[DT], buf_origin],
        n_elems: Int,
    ):
        ...
