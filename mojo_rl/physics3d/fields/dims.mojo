"""`DimsLike` — the single provider that replaces the containers' loose `Int`s.

WHAT THIS IS FOR (phase 1c, docs §5.1/§11.1)

Today every container spells its dimensions out one `Int` at a time:

    Data <DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH>
    Model <DTYPE, NV, NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON, NSITE,
           NEXCLUDE, NMESH_VERTS, NPAIR>

⚠ THE ANGLE BRACKETS ABOVE ARE DELIBERATE. Written in real syntax, this
"before" picture is indistinguishable from a call site: the repoint tool
matches by BRACKET STRUCTURE and cannot tell code from prose, so it rewrote
this example on the `Model` pass and again on the `Data` pass. It also
MANGLED the equivalent example in `parser/full_parser.mojo`, mapping
`nv=pm.NQ` / `nbody=pm.NV`, because that one listed arguments in an order
that never matched the real signature. Documentation is part of a scripted
edit's blast radius.

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

THE RUNTIME ACCESSORS ARRIVED IN 2b.0

Through phase 1c and 2a this trait was comptime-ONLY: every member was a
`comptime Int`, which is what made those phases gatable bit-exact. §5.1's
runtime accessors were held back until they had a caller, and 2b.0 is where
they got one — see `DimsLike`'s own docstring for the three families the
trait now carries, and `tests/physics3d/test_dyn_dims_ldl.mojo` for the
kernel that runs on both.

The static leg is unchanged by their arrival, and that is checked rather than
asserted: §10.6 read the arm64 asm and found the static accessor
byte-identical to a comptime parameter, and 2b.0 re-read it against THIS
trait (`scratchpad/p2b/asm_static.mojo` — 67 instructions each, the loop
bound the immediate `#27` in both, with a dynamic third arm at 174 as the
negative control). §10.8 measured the trait free on Metal (1.0005-1.0036x).

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


trait DimsLike(Copyable, Movable, ImplicitlyCopyable, Deinitable):
    """Every model dimension the `fields` containers are shaped by.

    ⚠ `ImplicitlyCopyable & Deinitable` ARE THERE SO A CONTAINER CAN STORE ONE
    (`var dims: D` in `Data`/`Model`, phase 3a). A struct field must be
    `Deinitable`, and `self.dims = dims` must copy without a `.copy()` the
    call sites would have to spell. Both providers are plain value types, so
    conformance is synthesized and nothing about the static leg changes.

    ⚠ ONE PROVIDER CARRIES ALL OF THEM even though no single container uses
    all — `Rk4Scratch` wants NQ/NV, `Model` wants ten others. A per-container
    trait would mean seven traits and seven derivations, and the call sites
    would have to pick the right one; carrying the union means every container
    takes the same `D` and the model's dimensions are named once per env.
    Unused members cost nothing: they are comptime `Int`s.

    THREE FAMILIES, AND THE DIFFERENCE IS THE WHOLE POINT OF 2b
    ==========================================================

    * `NQ`, `NV`, … — the **comptime exact** dimension. Every shipped call
      site reads these today. On a DYNAMIC provider they do not exist as
      values and are POISONED (see `DynDims`), so a site left unconverted
      fails loudly instead of silently sizing itself to zero.
    * `CAP_NQ`, `CAP_NV`, … — the **scratch cap**, i.e. which CONTAINER a
      `Scratch[T, CAP]` picks. On a static provider cap == exact, so its
      allocations stay byte-identical stack arrays; on a dynamic provider it
      is **0**, which selects the heap. It is no longer a bound the model may
      not exceed — see `DynDims` on why the caps were removed.

      ⚠ THIS FAMILY POISONS TO 0 WHILE `NV` POISONS TO -1, and the difference
      is load-bearing in both directions. Caps get MULTIPLIED (`ME * V_CAP`,
      `NV * NV`, `3 * NBODY`) and only 0 survives a product as 0; -1 would
      give `cap[(-1) * (-1)] == cap[1]`, selecting the STATIC leg with a
      one-element array. Meanwhile `NV` must stay negative so an UNCONVERTED
      site dies where it stands. Do not merge the two families.
    * `get_nq()`, `get_nv()`, … — the **runtime** dimension: loop bounds and
      strides. §10.6 read the arm64 asm and found the static implementation
      byte-identical to a comptime parameter, and §10.8 measured the trait
      free on Metal (1.0005-1.0036x).

    ⚠ THE ACCESSORS ARE `get_nv()` AND NOT `nv()` BECAUSE OF A NAME
    COLLISION, not a style preference. `Dims` spells its parameters in
    lowercase (`Dims[nq=3, nv=3]`) and its members in uppercase, because a
    struct parameter does NOT satisfy a `comptime` trait member — the
    explicit `comptime NQ = Self.nq` lines below are load-bearing. That
    leaves `nv` taken by the parameter, and a method may not share a name
    with one ("invalid redefinition of 'nv' … cannot overload with this
    non-function definition"). The prefix earns its keep anyway: `D.NV` and
    `dims.get_nv()` are the two sides of this migration, and they should not
    look alike at a glance.

    ⚠ `CAP_`, NOT §5.1's `MAX_`, because `MAX_CONTACTS` is itself a
    dimension and `MAX_MAX_CONTACTS` is not a name anyone should have to
    read.
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

    comptime CAP_NQ: Int
    comptime CAP_NV: Int
    comptime CAP_NBODY: Int
    comptime CAP_NJOINT: Int
    comptime CAP_NGEOM: Int
    comptime CAP_NSITE: Int
    comptime CAP_MAX_CONTACTS: Int
    comptime CAP_NEQUALITY: Int
    comptime CAP_NTENDON: Int
    comptime CAP_NEXCLUDE: Int
    comptime CAP_NMESH_VERTS: Int
    comptime CAP_NPAIR: Int
    comptime CAP_NACT: Int
    comptime CAP_NTEN: Int
    comptime CAP_NKEY: Int

    @staticmethod
    def comptime_value() raises -> Self:
        """The provider as a VALUE, when the caller has only the TYPE.

        ⚠ THIS IS `AsStatic[D]()`'s JOB MOVED ONTO THE TRAIT, and it exists
        for exactly one caller: a container's NULLARY constructor. `Data[…, D,
        …]()` has to fill `var dims: D` and has nothing to fill it from, and
        `D()` is not spellable because the trait cannot require a default
        constructor (see `AsStatic`).

        ⚠⚠ IT RAISES ON A DYNAMIC PROVIDER, AND THAT IS THE POINT. `DynDims`
        has no comptime value — an all-zero default is the silent failure
        `DIM_POISON` exists to prevent. So a `Data[…, DynDims, …]()` built
        without dimensions fails AT CONSTRUCTION with a message naming the
        fix, instead of allocating nothing and dying four calls away. Build
        those with the explicit `Data(dims)` constructor.

        The static leg never reaches the raise: `Dims` is stateless, so this
        is `Self()` and folds away.
        """
        ...

    def get_nq(self) -> Int:
        ...

    def get_nv(self) -> Int:
        ...

    def get_nbody(self) -> Int:
        ...

    def get_njoint(self) -> Int:
        ...

    def get_ngeom(self) -> Int:
        ...

    def get_nsite(self) -> Int:
        ...

    def get_max_contacts(self) -> Int:
        ...

    def get_nequality(self) -> Int:
        ...

    def get_ntendon(self) -> Int:
        ...

    def get_nexclude(self) -> Int:
        ...

    def get_nmesh_verts(self) -> Int:
        ...

    def get_npair(self) -> Int:
        ...

    def get_nact(self) -> Int:
        ...

    def get_nten(self) -> Int:
        ...

    def get_nkey(self) -> Int:
        ...


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

    # Cap == exact. Every `InlineArray[T, D.CAP_NV]` on the static leg is
    # therefore the allocation that ships today, to the byte.
    comptime CAP_NQ = Self.nq
    comptime CAP_NV = Self.nv
    comptime CAP_NBODY = Self.nbody
    comptime CAP_NJOINT = Self.njoint
    comptime CAP_NGEOM = Self.ngeom
    comptime CAP_NSITE = Self.nsite
    comptime CAP_MAX_CONTACTS = Self.max_contacts
    comptime CAP_NEQUALITY = Self.nequality
    comptime CAP_NTENDON = Self.ntendon
    comptime CAP_NEXCLUDE = Self.nexclude
    comptime CAP_NMESH_VERTS = Self.nmesh_verts
    comptime CAP_NPAIR = Self.npair
    comptime CAP_NACT = Self.nact
    comptime CAP_NTEN = Self.nten
    comptime CAP_NKEY = Self.nkey

    def __init__(out self):
        """Stateless: every dimension is a parameter. Exists only so a
        provider can be PASSED to a kernel — `_ldl_solve_env(env, dims, ...)`
        — which is what lets one body serve both legs."""
        pass

    @staticmethod
    def comptime_value() raises -> Self:
        """Stateless, so the value is free. See `DimsLike.comptime_value`."""
        return Self()

    @always_inline
    def get_nq(self) -> Int:
        return Self.nq

    @always_inline
    def get_nv(self) -> Int:
        return Self.nv

    @always_inline
    def get_nbody(self) -> Int:
        return Self.nbody

    @always_inline
    def get_njoint(self) -> Int:
        return Self.njoint

    @always_inline
    def get_ngeom(self) -> Int:
        return Self.ngeom

    @always_inline
    def get_nsite(self) -> Int:
        return Self.nsite

    @always_inline
    def get_max_contacts(self) -> Int:
        return Self.max_contacts

    @always_inline
    def get_nequality(self) -> Int:
        return Self.nequality

    @always_inline
    def get_ntendon(self) -> Int:
        return Self.ntendon

    @always_inline
    def get_nexclude(self) -> Int:
        return Self.nexclude

    @always_inline
    def get_nmesh_verts(self) -> Int:
        return Self.nmesh_verts

    @always_inline
    def get_npair(self) -> Int:
        return Self.npair

    @always_inline
    def get_nact(self) -> Int:
        return Self.nact

    @always_inline
    def get_nten(self) -> Int:
        return Self.nten

    @always_inline
    def get_nkey(self) -> Int:
        return Self.nkey


comptime AsStatic[D: DimsLike] = Dims[
    nq=D.NQ,
    nv=D.NV,
    nbody=D.NBODY,
    njoint=D.NJOINT,
    ngeom=D.NGEOM,
    nsite=D.NSITE,
    max_contacts=D.MAX_CONTACTS,
    nequality=D.NEQUALITY,
    ntendon=D.NTENDON,
    nexclude=D.NEXCLUDE,
    nmesh_verts=D.NMESH_VERTS,
    npair=D.NPAIR,
    nact=D.NACT,
    nten=D.NTEN,
    nkey=D.NKEY,
]
"""A VALUE of the provider `D`, for a caller that only has it as a parameter.

The converted kernels take `dims: D` as an argument, but a dispatcher holds
`D` as a comptime TYPE and has no value to pass. `D()` will not do — the
trait cannot require a default constructor, because `DynDims` has nothing to
default to (all-zero dimensions are precisely the silent failure `DIM_POISON`
exists to prevent). So the static leg re-spells its provider instead.

`Dims` is nominal and `ModelDims[MD]` expands to a `Dims[...]`, so
`AsStatic[D]` IS `D` for every provider that has comptime dimensions — no
conversion, no second type. And for one that does NOT, it expands to
`Dims[nq=-1, ...]`, which is the loud failure a dispatcher still on the
static leg should get if a dynamic provider reaches it.

⚠ ENUMERATED FROM THE TRAIT, NOT FROM WHAT ANY CALLER SPELLS. Phase 2a lost
seventeen tests to a union built from the members that happened to appear at
a site; `nact`/`nten`/`nkey` are named by no `Data` or `Model` spelling and
defaulted silently to 0.
"""


comptime DIM_POISON: Int = -1
"""What a DYNAMIC provider answers when asked for a COMPTIME dimension.

⚠ IT IS NOT ZERO, AND THAT IS THE ENTIRE DESIGN. A missing dimension that
defaults to 0 type-checks, allocates an empty tensor and fails four calls
later as "index 0 is out of bounds, valid range is 0 to -1" — this tree lost
seventeen tests to exactly that in phase 2a, all with the same message and
none pointing at a changed line. A negative dimension cannot be allocated and
cannot be a loop bound, so a site that reads `D.NV` off a dynamic provider —
i.e. a site the sweep has not converted yet — dies AT the unconverted site.
"""


struct DynDims(DimsLike):
    """Dimensions carried as RUNTIME state. NO BOUND ON MODEL SIZE.

    This is the provider that makes one compiled body serve many models: every
    loop bound and stride comes from the fields below.

    ## Why there are no longer any caps (§10.5 decision 2, resolved)

    This type used to take fifteen `cap_*` parameters and check them at
    construction, because §4.2 planned to keep the per-call scratch on the
    STACK with a fixed cap. §10.7 built that and refuted it: a fixed-cap
    `InlineArray` under a RUNTIME bound is 1.13-1.18x *worse* than the heap
    `List` it was meant to beat, because the cost is the runtime bound, not
    the cap size. `Scratch` therefore sends this leg to the heap (`CAP == 0`),
    and a cap that sizes nothing is a promise with nothing behind it.

    ⇒ **A binary is no longer built for a maximum model.** Any MJCF loads,
    however large, and there is no `raise` at construction to get wrong.

    ⚠⚠ ITS COMPTIME MEMBERS ARE POISON, NOT VALUES. See `DIM_POISON`. Note
    the two families poison DIFFERENTLY and both directions matter: `NV` is
    -1 so an unconverted site dies AT the site, while `CAP_NV` is 0 so that
    products of caps (`ME * V_CAP`, `NV * NV`) stay 0 and keep selecting the
    heap. With -1, `cap[D.NV * D.NV]` would be `cap[1]` — the static leg with
    a one-element array, silently overrun.
    """

    var _nq: Int
    var _nv: Int
    var _nbody: Int
    var _njoint: Int
    var _ngeom: Int
    var _nsite: Int
    var _max_contacts: Int
    var _nequality: Int
    var _ntendon: Int
    var _nexclude: Int
    var _nmesh_verts: Int
    var _npair: Int
    var _nact: Int
    var _nten: Int
    var _nkey: Int

    comptime NQ = DIM_POISON
    comptime NV = DIM_POISON
    comptime NBODY = DIM_POISON
    comptime NJOINT = DIM_POISON
    comptime NGEOM = DIM_POISON
    comptime NSITE = DIM_POISON
    comptime MAX_CONTACTS = DIM_POISON
    comptime NEQUALITY = DIM_POISON
    comptime NTENDON = DIM_POISON
    comptime NEXCLUDE = DIM_POISON
    comptime NMESH_VERTS = DIM_POISON
    comptime NPAIR = DIM_POISON
    comptime NACT = DIM_POISON
    comptime NTEN = DIM_POISON
    comptime NKEY = DIM_POISON

    comptime CAP_NQ = 0
    comptime CAP_NV = 0
    comptime CAP_NBODY = 0
    comptime CAP_NJOINT = 0
    comptime CAP_NGEOM = 0
    comptime CAP_NSITE = 0
    comptime CAP_MAX_CONTACTS = 0
    comptime CAP_NEQUALITY = 0
    comptime CAP_NTENDON = 0
    comptime CAP_NEXCLUDE = 0
    comptime CAP_NMESH_VERTS = 0
    comptime CAP_NPAIR = 0
    comptime CAP_NACT = 0
    comptime CAP_NTEN = 0
    comptime CAP_NKEY = 0

    def __init__(
        out self,
        *,
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
    ):
        """Keyword-only, for the reason `Dims`'s docstring gives: fifteen
        `Int`s in a row is the positional hazard this type exists to kill."""
        self._nq = nq
        self._nv = nv
        self._nbody = nbody
        self._njoint = njoint
        self._ngeom = ngeom
        self._nsite = nsite
        self._max_contacts = max_contacts
        self._nequality = nequality
        self._ntendon = ntendon
        self._nexclude = nexclude
        self._nmesh_verts = nmesh_verts
        self._npair = npair
        self._nact = nact
        self._nten = nten
        self._nkey = nkey

    @staticmethod
    def comptime_value() raises -> Self:
        """⚠ ALWAYS RAISES. There is no comptime value here to return — that
        is what makes this the dynamic provider. See
        `DimsLike.comptime_value`; the raise is the loud half of the same
        design that makes `DIM_POISON` negative."""
        raise Error(
            "DynDims has no comptime value: a physics3d container on the"
            " dynamic leg must be constructed WITH its dimensions, e.g."
            " Data[DTYPE, DynDims, BATCH](dims), not Data[...]()."
        )

    @always_inline
    def get_nq(self) -> Int:
        return self._nq

    @always_inline
    def get_nv(self) -> Int:
        return self._nv

    @always_inline
    def get_nbody(self) -> Int:
        return self._nbody

    @always_inline
    def get_njoint(self) -> Int:
        return self._njoint

    @always_inline
    def get_ngeom(self) -> Int:
        return self._ngeom

    @always_inline
    def get_nsite(self) -> Int:
        return self._nsite

    @always_inline
    def get_max_contacts(self) -> Int:
        return self._max_contacts

    @always_inline
    def get_nequality(self) -> Int:
        return self._nequality

    @always_inline
    def get_ntendon(self) -> Int:
        return self._ntendon

    @always_inline
    def get_nexclude(self) -> Int:
        return self._nexclude

    @always_inline
    def get_nmesh_verts(self) -> Int:
        return self._nmesh_verts

    @always_inline
    def get_npair(self) -> Int:
        return self._npair

    @always_inline
    def get_nact(self) -> Int:
        return self._nact

    @always_inline
    def get_nten(self) -> Int:
        return self._nten

    @always_inline
    def get_nkey(self) -> Int:
        return self._nkey


