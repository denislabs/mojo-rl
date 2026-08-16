"""`DimsLike` — the single provider that replaces the containers' loose `Int`s.

WHAT THIS IS FOR (phase 1c, docs §5.1/§11.1)

Today every container spells its dimensions out one `Int` at a time:

    Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH]
    Model[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON, NSITE,
          NEXCLUDE, NMESH_VERTS, NPAIR]

That is **924 spellings across the tree** and 2 247 dimension parameters, of
which 1 457 are in `mojo_rl/envs` reward/observation code doing nothing but
naming a type. Collapsing them to `Data[DTYPE, D, BATCH]` turns those 1 457
into ~292 — the single largest mechanical reduction in phase 2, and all of it
type-spelling with no arithmetic.

⚠⚠ THE REAL MOTIVE IS NOT BREVITY — IT IS THE POSITIONAL HAZARD. Every one of
those parameters is an `Int`, so inserting a dimension mid-list silently
shifts every positional instantiation with nothing to compile-error on.
`fields.Model` carries a standing comment about exactly this (`NMESH_VERTS`
would take `NPAIR`'s value and mesh collision would switch itself off tree-
wide), and its comment says the failure "has happened here before". One
named provider makes that class of bug unspellable.

WHY THE TRAIT IS *COMPTIME-ONLY* IN 1c

§5.1 designs `DimsLike` with runtime accessors (`fn nv(self) -> Int`) so a
dynamic leg can exist. **Those are deliberately absent here.** Phase 1c moves
the containers onto a named provider and changes NO code generation: every
member below is a `comptime Int`, so the static leg is bit-identical by
construction and 1c is gatable bit-exact. The runtime accessors arrive in 2b
with the dynamic leg, which is where they first have a caller.

Validated already: §10.6 read the arm64 asm and found `StaticDims.nv()`
byte-identical to a comptime parameter; §10.8 measured the trait free on Metal
(1.0005-1.0036x). The design question is closed on both backends — what is
open is the sweep, not the mechanism.

⚠ `BATCH` IS NOT A DIMENSION AND IS NOT HERE. It is the env batch size,
orthogonal to the model, and it stays a separate container parameter. Likewise
`ContactScratch`'s `JE_WS`, which is a computed workspace size owned by
`solver/je_budget.mojo`.

⚠ THE DERIVATION FROM A MODEL DEF IS NOT IN THIS FILE. It needs
`ModelDefLike`, which lives in `model/`, and `model` already imports `fields`
— putting it here would close a `fields -> model -> fields` cycle. Phase 2.0
(`af667e5c`) existed to delete exactly that shape from this package graph, so
`ModelDims[MD]` lives in `model/model_dims.mojo` instead. `fields` stays a
leaf.
"""


trait DimsLike:
    """Every model dimension the `fields` containers are shaped by.

    ⚠ ONE PROVIDER CARRIES ALL OF THEM even though no single container uses
    all — `Rk4Scratch` wants NQ/NV, `Model` wants ten others. A per-container
    trait would mean seven traits and seven derivations, and the call sites
    would have to pick the right one; carrying the union means every container
    takes the same `D` and the model's dimensions are named once per env.
    Unused members cost nothing: they are comptime `Int`s.
    """

    comptime NQ: Int
    comptime NV: Int
    comptime NBODY: Int
    comptime NJOINT: Int
    comptime NGEOM: Int
    comptime NSITE: Int
    comptime MAX_CONTACTS: Int
    comptime NEQUALITY: Int
    comptime NTENDON: Int
    comptime NEXCLUDE: Int
    comptime NMESH_VERTS: Int
    comptime NPAIR: Int
    comptime NACT: Int
    comptime NTEN: Int
    comptime NKEY: Int


struct Dims[
    nq: Int = 0,
    nv: Int = 0,
    nbody: Int = 0,
    njoint: Int = 0,
    ngeom: Int = 0,
    nsite: Int = 0,
    max_contacts: Int = 0,
    nequality: Int = 0,
    ntendon: Int = 0,
    nexclude: Int = 0,
    nmesh_verts: Int = 0,
    npair: Int = 0,
    nact: Int = 0,
    nten: Int = 0,
    nkey: Int = 0,
](DimsLike):
    """Dimensions spelled out directly — for models with no `ModelDefLike`.

    ⚠ USE `ModelDims[MD]` FOR ANY SHIPPED MODEL. This provider exists for the
    hand-built synthetic models in the test tree (e.g. `test_fk_fields`'s
    2-body hinge with NSITE=2, whose records are written straight into the
    per-field tensors with no model def anywhere).

    ⚠⚠ AND FOR THAT REASON, SPELL IT WITH KEYWORDS: `Dims[nq=3, nv=3]`, never
    `Dims[3, 3]`. Fifteen defaulted `Int`s in a row is the positional hazard
    this type was introduced to kill, and it would be absurd to reintroduce it
    inside the fix. Lowercase parameters + uppercase members is the same shape
    `ModelDefFromXML` uses to satisfy `ModelDefLike`.
    """

    comptime NQ = Self.nq
    comptime NV = Self.nv
    comptime NBODY = Self.nbody
    comptime NJOINT = Self.njoint
    comptime NGEOM = Self.ngeom
    comptime NSITE = Self.nsite
    comptime MAX_CONTACTS = Self.max_contacts
    comptime NEQUALITY = Self.nequality
    comptime NTENDON = Self.ntendon
    comptime NEXCLUDE = Self.nexclude
    comptime NMESH_VERTS = Self.nmesh_verts
    comptime NPAIR = Self.npair
    comptime NACT = Self.nact
    comptime NTEN = Self.nten
    comptime NKEY = Self.nkey
