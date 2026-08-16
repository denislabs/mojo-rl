"""NVIDIA localization probe: run ONLY the Ant (FREE-joint) FIELDS kernels,
with a ctx.synchronize() + print after each, so a CUDA illegal-address is
attributed to the exact kernel (the last "[n] ... ok" NOT printed).

Mirrors the fields sequence in test_crba_treewalk_fields.mojo's Ant leg but
DROPS the legacy reference kernels — so:
  * if this crashes at [n], that fields kernel has a FREE-joint bug on CUDA;
  * if it prints "ALL FIELDS KERNELS OK", the test crash was in the LEGACY
    reference kernels (which we already know miscompute on CUDA), not fields.

Run: MODULAR_DEBUG=device-sync-mode pixi run -e nvidia mojo run -I . \
        tests/physics3d/test_ant_fields_sync_probe.mojo
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import Data, Model, DynamicsScratch, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
)
from mojo_rl.physics3d.dynamics.subtree_com import (
    compute_subtree_com,
)
from mojo_rl.physics3d.dynamics.cdof import compute_cdof
from mojo_rl.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix,
)
from mojo_rl.envs.ant.ant_xml import AntModel

comptime DTYPE = DType.float32  # match the gate (Metal is fragile on float64)
comptime NQ = AntModel.NQ  # 15
comptime NV = AntModel.NV  # 14 (free joint: NQ - NV = 1)
comptime NBODY = AntModel.NBODY
comptime NJOINT = AntModel.NJOINT
comptime NGEOM = AntModel.NGEOM
comptime MC = AntModel.MAX_CONTACTS
comptime NEQ = AntModel.MAX_EQUALITY
comptime NTD = AntModel.MAX_TENDON
comptime NSITE = AntModel.NSITE
comptime NEXCL = AntModel.NEXCLUDE
comptime BATCH = 2


def main() raises:
    print("=== Ant FIELDS-only sync probe ===")
    print(
        "NQ=", NQ, " NV=", NV, " NBODY=", NBODY,
        " (free joint qpos/dof offset = NQ-NV =", NQ - NV, ")",
    )
    var ctx = DeviceContext()

    var mf = Model[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0]()
    AntModel.init_fields[DTYPE, 0](ctx, mf)

    var d = Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    # nonzero free-joint translation + torso quat + a couple joint angles
    for e in range(BATCH):
        d.qpos.data[e * NQ + 2] = Scalar[DTYPE](0.55)  # free z
        d.qpos.data[e * NQ + 3] = Scalar[DTYPE](1.0)  # quat w
        d.qpos.data[e * NQ + 8] = Scalar[DTYPE](1.0)  # a hinge
        d.qpos.data[e * NQ + 10] = Scalar[DTYPE](-1.0)  # a hinge
    d.upload_all(ctx)

    var scratch = DynamicsScratch[DTYPE, Dims[nv=NV, nbody=NBODY], BATCH]()
    scratch.upload_all(ctx)

    print("[1] forward_kinematics (serial) ...")
    forward_kinematics[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, ctx)
    ctx.synchronize()
    print("    [1] FK ok")

    print("[2] compute_subtree_com (serial) ...")
    compute_subtree_com[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, ctx)
    ctx.synchronize()
    print("    [2] subtree_com ok")

    print("[3] compute_cdof (serial) ...")
    compute_cdof[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, scratch, ctx)
    ctx.synchronize()
    print("    [3] cdof ok")

    print("[4] compute_mass_matrix PARALLEL=True (dense _mt) ...")
    compute_mass_matrix[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
        PARALLEL=True,
    ](d, mf, scratch, ctx)
    ctx.synchronize()
    print("    [4] dense mass matrix (_mt) ok")

    print("[5] compute_mass_matrix PARALLEL=True TREEWALK=True ...")
    compute_mass_matrix[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
        PARALLEL=True, TREEWALK=True,
    ](d, mf, scratch, ctx)
    ctx.synchronize()
    print("    [5] treewalk mass matrix (_mt) ok")

    # Also exercise the PARALLEL FK / cdof (the humanoid *training* config).
    print("[6] forward_kinematics PARALLEL=True (_mt) ...")
    forward_kinematics[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
        PARALLEL=True,
    ](d, mf, ctx)
    ctx.synchronize()
    print("    [6] FK _mt ok")

    print("[7] compute_cdof PARALLEL=True (_mt) ...")
    compute_cdof[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
        PARALLEL=True,
    ](d, mf, scratch, ctx)
    ctx.synchronize()
    print("    [7] cdof _mt ok")

    print("=== ALL FIELDS KERNELS OK — Ant fields path does NOT crash ===")
    print("   => the test crash was in the LEGACY reference kernels.")
