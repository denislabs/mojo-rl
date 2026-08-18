"""Load a model from an MJCF path chosen at RUN TIME — phase 3c-c.

## What this is the last piece of

Every shipped model reaches the engine through `ModelDefFromXML[xml_path=…]`,
where the path is a COMPTIME parameter. That makes the model's *identity*
part of the type: one binary, N baked models. §15.3 called breaking that "the
actual unlock — one binary, any file", and it was the highest value per unit
of work left, because the parse LOGIC has run at run time since phase 1b
(`FlatModelDef` is `List`-backed and non-generic).

What stood between the parser and this was never the parser:

* the CONTAINERS allocated from `comptime Self.NQ` — fixed in 3b (`43167d98`)
* the DISPATCHERS built comptime `Layout`s — fixed in 3a
* `comptime if D.NSITE > 0:` skipped work silently — fixed in 3c-b
* the BUILDER read `D.NBODY` in 50 places, and its inertia helper took the
  dimensions as comptime PARAMETERS — fixed in 3c-a

So this file is small, and that is the point. It is three steps:

    var fmd = parse_model_runtime("path/to/model.xml")
    var dims = dims_from_flat(fmd, max_contacts=20)
    var m = Model[DType.float64, DynDims](dims)
    build_model_runtime[DType.float64](fmd, dims, m)

and then `Data[DTYPE, DynDims, BATCH](dims)` plus any scratch the caller
needs. `dims` is returned separately rather than bundled with the `Model`
because every container takes it and a tuple would tie two independent
lifetimes together for no benefit.

## ⚠ WHAT THIS DOES **NOT** GIVE YOU

* **The GPU path.** Kernels stay comptime by decision 3, and a runtime
  provider captured by one reads 0 — the output is silently zeroed. A
  runtime-loaded model is CPU-only until its kernels are rebuilt.
* ~~Actuators and keyframes~~ — DONE. `SpecFields` carries a provider and
  `build_spec_fields` reads it, so `spec_fields_runtime` below produces the
  actuation records, reference pose, keyframes and joint-limit table too. A
  runtime-loaded model can now be DRIVEN.
* **SAP broadphase.** `detect_contacts_auto` selects the algorithm on a
  comptime `D.NGEOM >= SAP_THRESHOLD`, which is false for `DIM_POISON`, so a
  runtime model always takes the O(N^2) path. Correct, slower on a big model,
  and deliberate — see the comment there.
* **A mesh vertex budget it can compute for you.** See `dims_from_flat`.
"""

from .flat_model import FlatModelDef
from .full_parser import parse_xml_full
from .fields_build import (
    build_model_fields_from_flat,
    apply_auto_spring_damper,
    build_spec_fields,
)
from ..fields import Data, Model, DynamicsScratch, SpecFields, DynDims
from ..dynamics.invweight import compute_invweight0


def parse_model_runtime(
    xml_path: String, asset_base_dir: String = ""
) raises -> FlatModelDef:
    """Read and parse an MJCF file whose path is known only at run time.

    `asset_base_dir` is what relative asset paths resolve against; MuJoCo's
    rule is "the directory of the model file", so the default derives it from
    `xml_path`. Pass "" explicitly to resolve against the process CWD.
    """
    var f = open(xml_path, "r")
    var text = f.read()
    f.close()
    var base = asset_base_dir
    if not base:
        var cut = xml_path.rfind("/")
        # ⚠ `[byte=...]`, and `String(...)` around it. Mojo strings are
        # UTF-8, so plain `s[:n]` is rejected outright; the byte form is
        # what a path separator search wants. The slice is also a
        # `StringSpan` borrowing `xml_path`, hence the copy.
        base = String(xml_path[byte=0:cut]) if cut > 0 else String("")
    return parse_xml_full(text, base)


def dims_from_flat(
    fmd: FlatModelDef,
    max_contacts: Int = 50,
    nmesh_verts: Int = 0,
) raises -> DynDims:
    """The model's dimensions, read off the parse.

    ⚠ `nq`/`nv` ARE SUMMED FROM THE JOINTS, NOT RE-DERIVED FROM THEIR TYPES.
    `JointData` already carries its own `nq`/`nv` (7/6 free, 4/3 ball, 1/1
    otherwise), written by the same parse that produced the record. Counting
    joint types here would be a second implementation of the same rule, and
    this tree has been bitten repeatedly by two spellings of one quantity
    drifting apart.

    ⚠ `nbody` IS `len(bodies) + 1`. `FlatModelDef` does not store the
    worldbody; `Model` indexes it as body 0. Its own docstring says so, and
    `init_fields`' dimension check spells the same `+ 1`.

    ⚠ TWO ARGUMENTS ARE NOT IN THE FILE, and both are honest defaults rather
    than derivations:

    * `max_contacts` is a WORKSPACE choice, not a model property — the
      comptime path takes it as a parameter too (`max_contacts=20` on
      walker2d). Too small silently drops contacts, so pass the model's own
      number when porting one.
    * `nmesh_verts` is a hull VERTEX BUDGET. Computing it needs the meshes
      loaded, which `build_model_fields_from_flat` does itself and after this
      point — so it cannot be derived here without loading them twice. 0 is
      right for a mesh-free model; for a mesh model pass the budget and let
      the builder's capacity check raise if it is short. It raises with the
      required number, so the failure tells you the answer.
    """
    var nq = 0
    var nv = 0
    for j in fmd.joints:
        nq += j.nq
        nv += j.nv
    return DynDims(
        nq=nq,
        nv=nv,
        nbody=len(fmd.bodies) + 1,
        njoint=len(fmd.joints),
        ngeom=len(fmd.geoms),
        nsite=len(fmd.sites),
        max_contacts=max_contacts,
        nequality=len(fmd.equalities),
        ntendon=len(fmd.tendons),
        nexclude=len(fmd.excludes),
        nmesh_verts=nmesh_verts,
        npair=len(fmd.pairs),
        nact=len(fmd.actuators),
        nten=len(fmd.tendons),
        nkey=fmd.nkey,
    )


def build_model_runtime[
    DTYPE: DType
](
    fmd: FlatModelDef,
    dims: DynDims,
    mut m: Model[DTYPE, DynDims],
) raises:
    """Fill `m` completely — the CPU half of `ModelDefFromXML.init_fields`.

    ⚠ THREE STEPS, NOT ONE, AND THE ORDER IS THE REASON THIS FUNCTION EXISTS.
    `build_model_fields_from_flat` alone leaves a model that LOOKS complete
    and is not: `stat.meaninertia` is still 0 and every
    `<joint springdamper>` still holds the XML's stiffness instead of the
    derived one. The first version of `test_runtime_model_load` compared a
    model built that way against the comptime one and found exactly one
    differing element out of 1207 — `meta[26]`, MEANINERTIA — which is what
    sent me looking.

      1. records from the parse
      2. `compute_invweight0` — writes body/dof invweight0 AND meaninertia.
         It seeds its own reference pose from the joint records, so unlike
         `init_fields` it needs no `SpecFields`/`reset_data` first.
      3. `apply_auto_spring_damper` — READS `dof_invweight0`, so it must come
         last. Shared with `init_fields`, not copied.
    """
    build_model_fields_from_flat[DTYPE](fmd, m)
    var d = Data[DTYPE, DynDims, 1](dims)
    var sc = DynamicsScratch[DTYPE, DynDims, 1](dims)
    compute_invweight0[DTYPE](d, m, sc)
    apply_auto_spring_damper[DTYPE](fmd, m)


def spec_fields_runtime[
    DTYPE: DType
](fmd: FlatModelDef, dims: DynDims) raises -> SpecFields[DTYPE, DynDims]:
    """The actuation records, reference pose, keyframes and joint limits.

    ⚠ SEPARATE FROM `build_model_runtime` BECAUSE THE TWO BUNDLES ARE
    SEPARATE, and `SpecFields`' own docstring says why: `Model` is what the
    integrator, solver and collision kernels bind; actuation is read by
    exactly one function per target. A caller that only wants to look at a
    model's geometry should not have to build its actuators.

    ⚠ `build_spec_fields` RAISES IF `nact` DISAGREES with the parse. On the
    comptime path that catches a hand-supplied `nact` parameter going stale;
    here `dims_from_flat` derives it from `len(fmd.actuators)`, so the check
    is a tautology — which is the right outcome, not a reason to remove it.
    The same guard still fires for `nten` and `nkey`.
    """
    var sf = SpecFields[DTYPE, DynDims](dims)
    build_spec_fields[DTYPE](fmd, sf)
    return sf^
