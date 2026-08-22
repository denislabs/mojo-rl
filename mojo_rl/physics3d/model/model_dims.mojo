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

from ..fields.dims import DimsLike, Dims
from .model_def import ModelDefLike


# ⚠⚠ AN ALIAS, NOT A STRUCT — AND THAT IS THE WHOLE POINT.
#
# As a `struct ModelDims[...](DimsLike)` this was a DISTINCT TYPE from an
# equivalent `Dims[nq=..., nv=...]`, even with every value equal. That would
# have forced a flag day: containers converted with local `Dims[...]` adapters
# (the only way to convert one at a time — see `Rk4Scratch`) would all have to
# flip to `ModelDims` in ONE commit the moment the env started supplying `D`,
# which is exactly the "two incompatible calling conventions and no working
# engine" failure §6 names as phase 2's #1 risk.
#
# As a parameterized alias it EXPANDS to a `Dims[...]`, so
# `ModelDims[DogModel]` and the hand-spelled `Dims[nq=80, nv=79, ...]` are
# the same type and interoperate freely. Verified, not assumed
# (`scratchpad/probe_alias.mojo`): the compiler prints the alias-derived type
# as `Dims[Int(11), Int(9)]` and assigns a value across the two spellings.
#
# ⇒ containers can convert one at a time with adapters, and each adapter
# disappears silently when its owner gains a real `D`. No flag day.

comptime ModelDims[
    MD: ModelDefLike, nmesh_verts: Int = 0, nhfield_data: Int = 0
] = Dims[
    nq = MD.NQ,
    nv = MD.NV,
    nbody = MD.NBODY,
    njoint = MD.NJOINT,
    ngeom = MD.NGEOM,
    nsite = MD.NSITE,
    max_contacts = MD.MAX_CONTACTS,
    nequality = MD.MAX_EQUALITY,
    ntendon = MD.MAX_TENDON,
    nexclude = MD.NEXCLUDE,
    nmesh_verts=nmesh_verts,
    nhfield_data=nhfield_data,
    npair = MD.NPAIR,
    nact = MD.NACT,
    nten = MD.NTEN_F,
    nkey = MD.NKEY,
]
