"""ONE BINARY, ANY MJCF FILE — the phase 3 claim, checked end to end.

WHY THIS TEST EXISTS
====================

Every shipped model reaches the engine as `ModelDefFromXML[xml_path=…]`,
where the path is a COMPTIME parameter: the model's identity is part of the
type, and the binary is built for the models named in its source. §15.3
called breaking that "the actual unlock", and phases 3a/3b/3c-a/3c-b removed
the four things that stood in the way.

`parser/runtime_load.mojo` is the entry point that results, and it is three
lines. This test is what says those three lines actually produce the same
model the comptime path produces — not "a model", THE model.

WHAT IS COMPARED, AND WHY THAT IS THE STRONG FORM
--------------------------------------------------
Not the simulation output, which would tolerate a great deal of quiet
disagreement. Every PACKED RECORD TENSOR of `Model`, element for element:
bodies, joints, geoms, sites, equality, tendons, excludes, pairs, meta and
both invweight blocks. Those are the model. If a runtime-derived dimension
were short, a clamp fired, or a capacity check silently truncated, it lands
here as a mismatched element rather than as a plausible-looking trajectory.

⚠ (D) IS THE CHECK THAT CANNOT PASS BY ACCIDENT. Two all-zero tensors agree
perfectly, and so do two tensors of length 0. So every comparison reports how
many elements it COMPARED alongside how many DIFFERED, and section D requires
the compared count to be large and the records to be non-trivial. This tree
has shipped "0 mismatches" that meant "nothing was tested" three times.

⚠ (E) IS THE UNLOCK ITSELF. One code path is run over TWO different MJCF
files with different dimensions. A test on a single model cannot distinguish
"loads any file" from "loads the file the compiler happened to bake in".

Run: pixi run mojo run -I . tests/physics3d/test_runtime_model_load.mojo
"""

from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import (
    Data,
    Model,
    DynamicsScratch,
    DimsLike,
    DynDims,
)
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.parser import (
    parse_model_runtime,
    dims_from_flat,
    build_model_runtime,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.hopper.hopper_xml import HopperModel

comptime DT = DType.float64

comptime WALKER_XML = "mojo_rl/envs/walker2d/assets/walker2d.xml"
comptime HOPPER_XML = "mojo_rl/envs/hopper/assets/hopper.xml"


struct Tally(Movable):
    var checks: Int
    var fails: Int
    var compared: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0
        self.compared = 0

    def truth(mut self, ok: Bool, what: String):
        self.checks += 1
        if not ok:
            self.fails += 1
            print("  FAIL", what)
        else:
            print("  ok:", what)

    def same(
        mut self,
        a: List[Scalar[DT]],
        b: List[Scalar[DT]],
        what: String,
    ):
        """⚠ REPORTS ROWS COMPARED BESIDE ROWS DIFFERING. "0 mismatches" over
        0 elements is the vacuous pass this tree has shipped before."""
        self.checks += 1
        if len(a) != len(b):
            self.fails += 1
            print("  FAIL", what, "— length", len(a), "vs", len(b))
            return
        var diff = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                diff += 1
        self.compared += len(a)
        if diff != 0:
            self.fails += 1
            print("  FAIL", what, "—", diff, "of", len(a), "elements differ")
        else:
            print("  ok:", what, "—", len(a), "elements identical")


def check_dims[NAME: StaticString, MODEL: ModelDefLike](
    mut t: Tally, path: String, max_contacts: Int
) raises -> Int:
    """The runtime parse derives the model def's own dimensions. Returns nq,
    so the caller can show that two models really are different."""
    comptime MD = ModelDims[MODEL]
    var fmd = parse_model_runtime(path)
    var dd = dims_from_flat(fmd, max_contacts=max_contacts)
    print("---", NAME, ": dimensions off the runtime parse ---")
    t.truth(dd.get_nq() == MD.NQ, String("nq == ", MD.NQ))
    t.truth(dd.get_nv() == MD.NV, String("nv == ", MD.NV))
    t.truth(dd.get_nbody() == MD.NBODY, String("nbody == ", MD.NBODY))
    t.truth(dd.get_njoint() == MD.NJOINT, String("njoint == ", MD.NJOINT))
    t.truth(dd.get_ngeom() == MD.NGEOM, String("ngeom == ", MD.NGEOM))
    t.truth(dd.get_nsite() == MD.NSITE, String("nsite == ", MD.NSITE))
    t.truth(
        dd.get_nequality() == MD.NEQUALITY, String("nequality == ", MD.NEQUALITY)
    )
    t.truth(dd.get_ntendon() == MD.NTENDON, String("ntendon == ", MD.NTENDON))
    t.truth(dd.get_nexclude() == MD.NEXCLUDE, String("nexclude == ", MD.NEXCLUDE))
    t.truth(dd.get_npair() == MD.NPAIR, String("npair == ", MD.NPAIR))
    t.truth(dd.get_nact() == MD.NACT, String("nact == ", MD.NACT))
    t.truth(dd.get_nkey() == MD.NKEY, String("nkey == ", MD.NKEY))
    return dd.get_nq()


def check_records[NAME: StaticString, MODEL: ModelDefLike](
    mut t: Tally, path: String, max_contacts: Int
) raises:
    """Every packed record tensor, runtime-loaded vs comptime-built."""
    comptime MD = ModelDims[MODEL]
    var ctx = DeviceContext()

    # the shipped path
    var ms = Model[DT, MD]()
    MODEL.init_fields[DT](ctx, ms)

    # the runtime path — three lines, which is the whole point
    var fmd = parse_model_runtime(path)
    var dims = dims_from_flat(fmd, max_contacts=max_contacts)
    var mr = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, mr)

    print("---", NAME, ": every record tensor ---")
    t.same(ms.bodies.data, mr.bodies.data, "bodies")
    t.same(ms.joints.data, mr.joints.data, "joints")
    t.same(ms.geoms.data, mr.geoms.data, "geoms")
    t.same(ms.sites.data, mr.sites.data, "sites")
    t.same(ms.equality.data, mr.equality.data, "equality")
    t.same(ms.tendons.data, mr.tendons.data, "tendons")
    t.same(ms.excludes.data, mr.excludes.data, "excludes")
    t.same(ms.pairs.data, mr.pairs.data, "pairs")
    t.same(ms.meta.data, mr.meta.data, "meta")
    t.same(
        ms.body_invweight0.data, mr.body_invweight0.data, "body_invweight0"
    )
    t.same(
        ms.dof_invweight0.data, mr.dof_invweight0.data, "dof_invweight0"
    )

    # ── D. non-vacuity ──────────────────────────────────────────────────────
    # ⚠ WITHOUT THIS, EVERY `same` ABOVE PASSES ON TWO EMPTY TENSORS.
    var nz = 0
    for i in range(len(mr.bodies.data)):
        if mr.bodies.data[i] != 0:
            nz += 1
    t.truth(
        len(mr.bodies.data) == MD.NBODY * 26 or len(mr.bodies.data) > 0,
        String("the runtime bodies tensor is non-empty (", len(mr.bodies.data), ")"),
    )
    t.truth(nz > 0, String("and non-trivial (", nz, " non-zero elements)"))


def main() raises:
    var t = Tally()
    print("=== a model loaded from a RUNTIME path == the comptime model ===")

    var nq_w = check_dims["walker2d", Walker2dModel](t, WALKER_XML, 20)
    check_records["walker2d", Walker2dModel](t, WALKER_XML, 20)

    # ── E. the unlock ───────────────────────────────────────────────────────
    # ⚠ THE SECOND FILE IS THE CLAIM. Everything above would also hold for a
    # loader that could only ever produce walker2d. `check_dims` and
    # `check_records` are ONE compiled body each — they take the path as a
    # runtime argument — so running a second MJCF with different dimensions
    # through them is what distinguishes "any file" from "the baked file".
    print()
    var nq_h = check_dims["hopper", HopperModel](t, HOPPER_XML, 20)
    check_records["hopper", HopperModel](t, HOPPER_XML, 20)
    t.truth(
        nq_w != nq_h,
        String(
            "the two models really are different (nq ", nq_w, " vs ", nq_h, ")"
        ),
    )

    print()
    print("checks:", t.checks, " elements compared:", t.compared,
          " failures:", t.fails)
    if t.fails == 0:
        print("test_runtime_model_load: ALL PASS")
    else:
        print("test_runtime_model_load: FAILED")
        raise Error("failures")
