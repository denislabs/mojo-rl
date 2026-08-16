"""`DimsLike` / `Dims` / `ModelDims` — phase 1c.1's gate.

⚠⚠ THE OBVIOUS TEST HERE IS VACUOUS AND IS NOT WHAT THIS FILE DOES.
Asserting `ModelDims[MD].NV == MD.NV` restates `model_dims.mojo` back to
itself: the file says so, so the test passes, and a mapping typo like
`comptime NJOINT = MD.NGEOM` sails through. "0 mismatches" and "nothing
tested" are the same number.

So the dimensions are checked against the **generated `*_dims.mojo`**, which
comes from `mujoco.MjModel.from_xml_path` (`tools/gen_model_dims.py`) and is
therefore an oracle INDEPENDENT of both the provider and our own parser.

⚠ AND THE MODEL IS CHOSEN SO A SWAP IS DETECTABLE. dm_walker cannot do this
job: its counts are nbody=8, ngeom=8, njoint=nq=nv=9, so swapping NBODY with
NGEOM — or any two of NJOINT/NQ/NV — passes every equality. dog's fifteen
counts are pairwise distinct (62/74/80/79/128/38/12/30), which is the property
that makes the assertions bite. `_distinct` below GATES that property rather
than trusting it, so if a future dog edit collides two counts this file says
so instead of quietly going vacuous.

Run: pixi run mojo run -I . tests/physics3d/test_dims_provider.mojo
"""

from std.testing import assert_true

from mojo_rl.physics3d.fields.dims import DimsLike, Dims
from mojo_rl.physics3d.model.model_dims import ModelDims

from mojo_rl.envs.dm_control.dog.dog_xml import DMDogStandWalkModel
from mojo_rl.envs.dm_control.dog.dog_dims import DM_DOG_STAND_WALK_DIMS


comptime G = DM_DOG_STAND_WALK_DIMS          # <- MuJoCo, via the generator
comptime D = ModelDims[DMDogStandWalkModel]  # <- the thing under test


struct Tally(Copyable, Movable):
    var checks: Int
    var bad: Int

    def __init__(out self):
        self.checks = 0
        self.bad = 0


def _eq(mut t: Tally, what: String, got: Int, want: Int) raises:
    t.checks += 1
    if got != want:
        t.bad += 1
        print("  FAIL", what, ": got", got, " want", want)


def _distinct(mut t: Tally, a_name: String, a: Int, b_name: String, b: Int):
    """A swap between two EQUAL counts is undetectable — say so loudly."""
    t.checks += 1
    if a == b:
        t.bad += 1
        print(
            "  VACUITY", a_name, "==", b_name, "(", a, ") — a swap between"
            " these two would pass every assertion above; pick another model",
        )


def main() raises:
    var t = Tally()

    print("=== ModelDims[dog] vs the MuJoCo-generated dims ===")
    _eq(t, "NQ", D.NQ, G.NQ)
    _eq(t, "NV", D.NV, G.NV)
    _eq(t, "NBODY", D.NBODY, G.NBODY)
    _eq(t, "NJOINT", D.NJOINT, G.NJOINT)
    _eq(t, "NGEOM", D.NGEOM, G.NGEOM)
    _eq(t, "NSITE", D.NSITE, G.NSITE)
    _eq(t, "NACT", D.NACT, G.NACT)
    _eq(t, "NEXCLUDE", D.NEXCLUDE, G.NEXCLUDE)
    _eq(t, "NPAIR", D.NPAIR, G.NPAIR)

    print()
    print("=== the vacuity guard: are those counts actually distinct? ===")
    _distinct(t, "NBODY", D.NBODY, "NGEOM", D.NGEOM)
    _distinct(t, "NJOINT", D.NJOINT, "NQ", D.NQ)
    _distinct(t, "NQ", D.NQ, "NV", D.NV)
    _distinct(t, "NBODY", D.NBODY, "NJOINT", D.NJOINT)
    _distinct(t, "NSITE", D.NSITE, "NACT", D.NACT)
    _distinct(t, "NACT", D.NACT, "NEXCLUDE", D.NEXCLUDE)

    print()
    print("=== the renamed three — no generated counterpart, so vs MD ===")
    # ⚠ WEAKER BY NECESSITY, and stated rather than hidden: MAX_EQUALITY /
    # MAX_TENDON / NTEN_F are env-and-parser concepts the generator does not
    # emit, so these three restate `model_dims.mojo`. They are here to pin the
    # RENAME (NEQUALITY<-MAX_EQUALITY, NTENDON<-MAX_TENDON, NTEN<-NTEN_F),
    # which is the part a reader gets wrong, not the value.
    _eq(t, "NEQUALITY <- MAX_EQUALITY", D.NEQUALITY,
        DMDogStandWalkModel.MAX_EQUALITY)
    _eq(t, "NTENDON <- MAX_TENDON", D.NTENDON, DMDogStandWalkModel.MAX_TENDON)
    _eq(t, "NTEN <- NTEN_F", D.NTEN, DMDogStandWalkModel.NTEN_F)
    _eq(t, "NKEY", D.NKEY, DMDogStandWalkModel.NKEY)

    print()
    print("=== NMESH_VERTS is a PARAMETER, not a model-def member ===")
    # It comes from the CONFIG (mesh collision is an env decision), so the
    # default must be 0 and an override must actually reach the member. A
    # provider that ignored its parameter would still pass the default row.
    _eq(t, "default", ModelDims[DMDogStandWalkModel].NMESH_VERTS, 0)
    _eq(t, "override", ModelDims[DMDogStandWalkModel, 4096].NMESH_VERTS, 4096)

    print()
    print("=== the raw provider maps each keyword to its own member ===")
    # Negative control for a copy-paste inside `Dims`: every value differs, so
    # any member reading a neighbour's parameter fails.
    comptime R = Dims[
        nq=1, nv=2, nbody=3, njoint=4, ngeom=5, nsite=6, max_contacts=7,
        nequality=8, ntendon=9, nexclude=10, nmesh_verts=11, npair=12,
        nact=13, nten=14, nkey=15,
    ]
    _eq(t, "raw NQ", R.NQ, 1)
    _eq(t, "raw NV", R.NV, 2)
    _eq(t, "raw NBODY", R.NBODY, 3)
    _eq(t, "raw NJOINT", R.NJOINT, 4)
    _eq(t, "raw NGEOM", R.NGEOM, 5)
    _eq(t, "raw NSITE", R.NSITE, 6)
    _eq(t, "raw MAX_CONTACTS", R.MAX_CONTACTS, 7)
    _eq(t, "raw NEQUALITY", R.NEQUALITY, 8)
    _eq(t, "raw NTENDON", R.NTENDON, 9)
    _eq(t, "raw NEXCLUDE", R.NEXCLUDE, 10)
    _eq(t, "raw NMESH_VERTS", R.NMESH_VERTS, 11)
    _eq(t, "raw NPAIR", R.NPAIR, 12)
    _eq(t, "raw NACT", R.NACT, 13)
    _eq(t, "raw NTEN", R.NTEN, 14)
    _eq(t, "raw NKEY", R.NKEY, 15)

    print()
    print("checks:", t.checks, " failures:", t.bad)
    assert_true(t.bad == 0, String(t.bad) + " dims-provider check(s) failed")
    print()
    print("PASS")
