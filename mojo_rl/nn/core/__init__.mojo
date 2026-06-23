"""Shared, framework-agnostic core surface.

The legacy stateless-LayoutTensor framework (Module / Param / Tensor /
combinators / optimizers / loss / checkpoint / autodiff walkers) was removed in
the legacy-`nn` sunset (see `docs/STORAGE_NN_LEGACY_REMOVAL_SCOPE.md`); its
replacement lives under `mojo_rl/nn/storage/`. What remains here is the shared
infra consumed by the storage tree and the migrated agents: the `Op` traits the
activation functors conform to, logging/serialization helpers, the device-buffer
`TargetStorage`, and the `mptr` pointer-erasure chokepoint.

Note: target-tag constants and most symbols are imported from their submodules
directly at the use site; only the stable shared surface is re-exported here."""

from .element_op import ElementOp
from .binary_element_op import BinaryElementOp
from .reduce_op import ReduceOp
from .saveable import Saveable
from .save_scalar import SaveScalar, SaveI, SaveBool
from .metric import Metric, LogScalar
from .log_bundle import log_bundle
from .target_storage import TargetStorage
from .ptr import mptr
