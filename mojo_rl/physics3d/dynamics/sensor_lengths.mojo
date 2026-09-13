"""`d.ten_length` — MuJoCo's `mj_tendon`, run only when something reads it.

⚠ ITS SIBLING `d.actuator_length` IS NOT FILLED HERE, and the asymmetry is
forced. The actuator records live in `SpecFields`, which `EulerIntegrator`
does not hold — `step` is handed `Data` and `Model` only. The transmission
length is therefore written by `apply_actions_fields`, which owns those
records, already walks each actuator's triples, and runs at the same `qpos`
MuJoCo's `mj_transmission` does. See `Data.actuator_length`.

MuJoCo fills `d->ten_length` for every tendon inside every `mj_fwdPosition`.
This engine does not, and that is a deliberate scope call rather than a gap:
a spatial tendon's length is a polyline walk over its wrap geoms, and the
three places that need one — `constraints/tendon_limit`,
`constraints/equality_tendon`, `dynamics/pose_transmission` — each compute it
for the tendons THEY care about, at the point they care. Materialising the
whole array every step would add a second walk per tendon per step (700 of
them on `ms_human_700`) to serve a sensor almost nothing declares.

⚠⚠ SO THIS PASS IS GUARDED ON A CONSUMER, AND THE GUARD IS THE SENSOR TABLE.
`compute_tendon_lengths` returns immediately unless some SERVED sensor row is
a `<tendonpos>`. Everything else leaves `d.ten_length` at the NaN `Data`
filled it with — loud to a reader, free to a model that never asks. A zero
fill would have been a plausible length.

⚠ `flg_jac=False` ON THE SPATIAL CALL, WHICH IS WHY `cdof` MAY BE STALE HERE.
The length is `sum |p_{k+1} - p_k|` over world points and reaches `cdof` only
through `_contact_jacobian_row`, which that flag skips. This pass runs at
MuJoCo's `mj_tendon` point — inside the position stage, before the sensors
that read it — and the step builds `cdof` later. See
`spatial_tendon_length_jac`'s own note.

⚠ CPU, BATCH=1, like the sensor passes it feeds. `sensors/eval.mojo` takes
host `List`s and one env; when AUD-53 lands the batched leg it will want a
kernel here rather than a loop over envs.
"""

from layout import Layout, LayoutTensor

from ..fields import (
    Data, Model, DimsLike, DynamicsScratch,
    Scratch, cap, DYN1, DYN2, rl1, rl2,
)
from ..constants import SENS_TENDONPOS
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_META_IDX_NTENDON,
    MODEL_SENSOR_SIZE,
    MODEL_SITE_SIZE,
    MODEL_TENDON_SIZE,
    SENSOR_IDX_SERVED,
    SENSOR_IDX_TYPE,
    TENDON_IDX_KIND,
    TENDON_KIND_SPATIAL,
)
from .tendon import spatial_tendon_length_jac, fixed_tendon_length_jac


@always_inline
def model_reads_tendon_length[
    DTYPE: DType, D: DimsLike
](m: Model[DTYPE, D]) -> Bool:
    """Does any SERVED sensor row read `d.ten_length`?

    A loop over a table that is at most a few dozen rows on the largest model
    in the tree, run once per step on the CPU single-env leg. Caching it in a
    `MODEL_META_IDX_*` column was the alternative and buys a few nanoseconds
    at the cost of a value that can disagree with the table it summarises.
    """
    var nsensor = m.dims.get_nsensor()
    for i in range(nsensor):
        var o = i * MODEL_SENSOR_SIZE
        if Int(m.sensors.data[o + SENSOR_IDX_SERVED]) != 1:
            continue
        if Int(m.sensors.data[o + SENSOR_IDX_TYPE]) == SENS_TENDONPOS:
            return True
    return False


def compute_tendon_lengths[
    DTYPE: DType, D: DimsLike
](
    mut d: Data[DTYPE, D, 1],
    mut m: Model[DTYPE, D],
    mut sc: DynamicsScratch[DTYPE, D, 1],
) raises:
    """Fill `d.ten_length` for every tendon, if any sensor asks for it.

    Both kinds go through the shared helpers in `dynamics/tendon.mojo`, so
    the formula has one spelling — the `<tendonpos>` sensor would otherwise
    have been its fourth.
    """
    var nt = m.dims.get_ntendon()
    if nt == 0:
        return
    var nten = Int(Float64(m.meta.data[MODEL_META_IDX_NTENDON]))
    if nten <= 0:
        return
    if nten > nt:
        nten = nt
    if not model_reads_tendon_length[DTYPE, D](m):
        return

    var dm = d.dims
    var mdm = m.dims
    var nv = dm.get_nv()

    comptime V_CAP = cap[D.CAP_NV]()
    var tJ = Scratch[Scalar[DTYPE], V_CAP](nv, fill=Scalar[DTYPE](0))

    # LayoutTensor views, the same idiom every `dynamics/` dispatcher uses.
    var rl_TEN = rl2(mdm.get_ntendon(), MODEL_TENDON_SIZE)
    var rl_SITE = rl2(mdm.get_nsite(), MODEL_SITE_SIZE)
    var rl_GEOM = rl2(mdm.get_ngeom(), MODEL_GEOM_SIZE)
    var rl_BODY = rl2(mdm.get_nbody(), MODEL_BODY_SIZE)
    var rl_JOINT = rl2(mdm.get_njoint(), MODEL_JOINT_SIZE)
    var rl_MMETA = rl1(MODEL_META_SIZE)
    var rl_B3 = rl2(1, dm.get_nbody() * 3)
    var rl_B4 = rl2(1, dm.get_nbody() * 4)
    var rl_CDOF = rl2(1, dm.get_nv() * 6)
    var rl_QPOS = rl2(1, dm.get_nq())
    var tendons_v = m.tendons.lt_dyn["cpu", DYN2](rl_TEN)
    var sites_v = m.sites.lt_dyn["cpu", DYN2](rl_SITE)
    var geoms_v = m.geoms.lt_dyn["cpu", DYN2](rl_GEOM)
    var bodies_v = m.bodies.lt_dyn["cpu", DYN2](rl_BODY)
    var joints_v = m.joints.lt_dyn["cpu", DYN2](rl_JOINT)
    var mmeta_v = m.meta.lt_dyn["cpu", DYN1](rl_MMETA)
    var stcom_v = d.subtree_com.lt_dyn["cpu", DYN2](rl_B3)
    var xpos_v = d.xpos.lt_dyn["cpu", DYN2](rl_B3)
    var xquat_v = d.xquat.lt_dyn["cpu", DYN2](rl_B4)
    var qpos_v = d.qpos.lt_dyn["cpu", DYN2](rl_QPOS)
    # ⚠ STALE AT THIS POINT IN THE STEP, AND NEVER READ. `compute_cdof` runs
    # later; `flg_jac=False` below is what makes that safe. Binding it anyway
    # keeps the call shape identical to every other caller's, so a future
    # reader comparing the two sees one difference, not two.
    var cdof_v = sc.cdof.lt_dyn["cpu", DYN2](rl_CDOF)

    for t in range(nten):
        var kind = Int(Float64(m.tendons.data[t * MODEL_TENDON_SIZE
                                              + TENDON_IDX_KIND]))
        var length = Scalar[DTYPE](0)
        if kind == TENDON_KIND_SPATIAL:
            length = spatial_tendon_length_jac[DTYPE, V_CAP, 1](
                0, t, dm, tendons_v, sites_v, geoms_v, bodies_v, joints_v,
                mmeta_v, stcom_v, cdof_v, xpos_v, xquat_v, tJ,
                flg_jac=False,
            )
        else:
            length = fixed_tendon_length_jac[DTYPE, V_CAP](
                0, t, dm, tendons_v, joints_v, qpos_v, tJ
            )
        d.ten_length.data[t] = length

