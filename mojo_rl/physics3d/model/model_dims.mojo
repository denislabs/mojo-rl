"""`ModelDims[MD]` — a `DimsLike` derived from a model def.

This is the derivation that makes phase 1c worth doing: a shipped model names
its dimensions ONCE, here, and every container spelling downstream becomes
`Data[DTYPE, D, BATCH]` instead of seven positional `Int`s.

⚠ IT LIVES IN `model/`, NOT `fields/`, and that is structural. It needs
`ModelDefLike` from this package, and `model` already imports `fields`; the
trait and the raw provider therefore sit in `fields/dims.mojo` and this
derivation sits here, so the edge stays `model -> fields` and `fields` stays a
leaf. Phase 2.0 (`af667e5c`) deleted a 22.5k-line cycle from this graph; the
next step should not put one back.

⚠⚠ `nmesh_verts` IS A PARAMETER BECAUSE IT IS NOT A MODEL-DEF MEMBER.

Every other dimension here is read off `MD`. `NMESH_VERTS` is not on
`ModelDefLike` at all — `Phyics3dEnv` reads it from the CONFIG
(`comptime NMESH_VERTS: Int = Self.CONFIG.NMESH_VERTS`, `phyics3d_env.mojo`),
because whether a model's meshes are COLLIDABLE is an env decision, not a
property of the MJCF: the same model can run with mesh collision on or off.
Defaulting it to 0 here matches the config's own default (only configs with
collidable meshes override it).

⇒ a caller that has a config MUST pass it: `ModelDims[MODEL_DEF, CONFIG.NMESH_VERTS]`.
Omitting it silently yields a model whose mesh geoms carry no geometry, which
is a failure this tree has already shipped once and gated green
(`feedback_a_silent_asset_cap_leaves_geoms_with_no_geometry`).

NAME MAPPING — the two sides disagree, on purpose, and the map is here:

    DimsLike.NEQUALITY  <-  MD.MAX_EQUALITY
    DimsLike.NTENDON    <-  MD.MAX_TENDON
    DimsLike.NTEN       <-  MD.NTEN_F

`fields.Model` and `fields.SpecFields` already spell these as NEQUALITY /
NTENDON / NTEN while the model def calls them MAX_* / *_F, so the rename is
pre-existing and this file is where it stops being retyped at every call site.
"""

from ..fields.dims import DimsLike
from .model_def import ModelDefLike


struct ModelDims[MD: ModelDefLike, nmesh_verts: Int = 0](DimsLike):
    """Every `fields` container's shape, read off one model def."""

    comptime NQ = Self.MD.NQ
    comptime NV = Self.MD.NV
    comptime NBODY = Self.MD.NBODY
    comptime NJOINT = Self.MD.NJOINT
    comptime NGEOM = Self.MD.NGEOM
    comptime NSITE = Self.MD.NSITE
    comptime MAX_CONTACTS = Self.MD.MAX_CONTACTS
    comptime NEQUALITY = Self.MD.MAX_EQUALITY
    comptime NTENDON = Self.MD.MAX_TENDON
    comptime NEXCLUDE = Self.MD.NEXCLUDE
    comptime NMESH_VERTS = Self.nmesh_verts
    comptime NPAIR = Self.MD.NPAIR
    comptime NACT = Self.MD.NACT
    comptime NTEN = Self.MD.NTEN_F
    comptime NKEY = Self.MD.NKEY
