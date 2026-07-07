"""Per-field tensor containers for the physics3d slab→fields migration
(P1 of docs/PHYSICS3D_TENSOR_MIGRATION_SCOPE.md).

`DataFields` — batched per-field simulation state (replaces the flat
`[BATCH, STATE_SIZE]` state slab). `ModelFields` — static model config as
one packed tensor per record family (replaces the flat model slab).
Both carry transitional `load_from_slab` / `store_to_slab` bridges for the
coexistence period; the legacy flat path is untouched and still the one
running until pipelines are ported (P2+).
"""

from .data_fields import DataFields
from .model_fields import ModelFields
from .dynamics_scratch import DynamicsScratch
from .contact_scratch import ContactScratch
from .rk4_scratch import Rk4Scratch
