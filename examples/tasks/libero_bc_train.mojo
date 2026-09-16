"""BEHAVIOUR CLONING ON LIBERO'S DEMONSTRATIONS — the port's first policy.

    pixi run libero-bc-train
    pixi run libero-bc-train --epochs 40 --batch 512 --lr 3e-4
    pixi run libero-bc-train --demos 10 --epochs 5        # a smoke run

Reads `build/demos/<family>.lowdim.h5` (`data/libero_demos.mojo`'s store: one
episode per demonstration, `state` in OUR joint order, `action` the recorded
seven OSC_POSE words), rebuilds each frame's OBSERVATION exactly as the batched
env writes it, fits an MLP, and writes a checkpoint
`examples/tasks/libero_eval_batched.mojo` can load into `_policy_action`.

## ⚠⚠ THE OBSERVATION IS REBUILT, NOT STORED — AND THAT IS THE POINT

The store holds `state` (qpos ++ qvel), not the policy input. The input is what
`task_hooks.write_task_obs` writes on the device — qpos, qvel, one active word
per free slot, then the nine goal words measured from `robot_grip_site` — and
`write_task_obs_host` is the SAME implementation
(`tests/tasks/test_libero_task_hooks.mojo` gates the two word for word on every
LIBERO task). So each row here is: load `state` into a CPU `Data`, run FK, put
the row's task tape and mask into `meta`, and call the host writer. A policy
trained on anything else would be trained on a vector the env never produces.

⚠ FK PER ROW IS THE COST. 63 728 rows of `libero_goal` take about a minute.

## ⚠ THE SPLIT IS BY DEMONSTRATION, NOT BY ROW

Consecutive frames of one demo are nearly identical; a row-wise split puts a
frame's neighbours on both sides and reports a validation error that means
nothing. `--val-demos K` holds out the LAST K demos of every task.

## ⚠⚠ THE BASELINE IS PRINTED BESIDE THE RESULT

A validation MSE on its own is unreadable. Two references are printed with it:
the error of predicting ZERO (LIBERO's null action — what the eval harness
scores 0 with) and of predicting the TRAINING MEAN action. A policy that does
not beat both has learned nothing, and the run says so rather than reporting a
number that looks like progress.

## ⚠ WHAT THIS IS NOT

It is not a LIBERO result. A success rate comes from
`libero_eval_batched.mojo` on the frozen inits, on the box; this file only
fits actions to observations. Open-loop imitation of 50 demos per task is a
BASELINE — the thing a real policy (ACT, a diffusion head, an image policy)
has to beat, and the thing that says whether the low-dimensional observation
carries enough to imitate at all.
"""

from std.os import listdir, makedirs
from std.os.path import dirname, exists
from std.random import seed as seed_rng, random_ui64
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.checkpoint import save_params
from mojo_rl.nn.core.param import walk_params
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam

from mojo_rl.data.store import TrajectoryStore
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
)
from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family,
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, joint_qpos_addresses,
)
from mojo_rl.tasks.tape import encode_goal, TAPE_WORDS
from mojo_rl.tasks.active import active_mask
from mojo_rl.tasks.bc_policy import BcNet, BC_HID, write_bc_norm
from mojo_rl.tasks.task_hooks import write_task_obs_host
from mojo_rl.tasks.placement.libero_goal import LiberoGoalPlacement
from mojo_rl.tasks.libero_goal_xml import (
    LIBERO_GOAL_OBS_DIM, LIBERO_GOAL_MAX_CONTACTS,
)


comptime H = DType.float64
comptime FAMILY = "libero_goal"
comptime FAMILY_DIR = "mojo_rl/tasks/families/"
comptime TASK_DIR = "mojo_rl/tasks/tasks/"
comptime STORE = "build/demos/libero_goal.lowdim.h5"
comptime OUT_DIR = "build/policies"
comptime OBS = LIBERO_GOAL_OBS_DIM
comptime ACT = 7
comptime HID = BC_HID
comptime BATCH = 256
"""⚠ COMPTIME: `forward`/`vjp` take the batch as a parameter, so one build has
one batch size. `--batch` would be a second kernel instantiation, not a flag."""

comptime NET = BcNet[OBS, ACT]
"""⚠ THE SHAPE LIVES IN `tasks/bc_policy.mojo`, so the driver that RUNS this
checkpoint builds the same network from the same declaration rather than a
second spelling of it."""


def _task_names(family: String) raises -> List[String]:
    var out = List[String]()
    var want = family + "__"
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith(want) and n.endswith(".task"):
            out.append(String(n[byte = 0 : n.byte_length() - 5]))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    return out^


def _pad(s: String, n: Int) -> String:
    var out = String(s)
    if out.byte_length() > n:
        return String(out[byte = 0 : n])
    while out.byte_length() < n:
        out += " "
    return out^


def main() raises:
    var args = argv()
    var epochs = 20
    var lr = 1.0e-3
    var val_demos = 5
    var max_demos = 0
    var out_path = String(OUT_DIR) + "/libero_goal_bc.ckpt"
    var i = 1
    while i < len(args):
        var s = String(args[i])
        if s == "--epochs" and i + 1 < len(args):
            epochs = Int(String(args[i + 1]))
            i += 1
        elif s == "--lr" and i + 1 < len(args):
            lr = Float64(String(args[i + 1]))
            i += 1
        elif s == "--val-demos" and i + 1 < len(args):
            val_demos = Int(String(args[i + 1]))
            i += 1
        elif s == "--demos" and i + 1 < len(args):
            max_demos = Int(String(args[i + 1]))
            i += 1
        elif s == "--out" and i + 1 < len(args):
            out_path = String(args[i + 1])
            i += 1
        else:
            raise Error(
                "libero bc train: unknown argument '" + s + "' (--epochs N,"
                " --lr X, --val-demos K, --demos N, --out PATH)"
            )
        i += 1

    print("=" * 78)
    print("Behaviour cloning on LIBERO's demonstrations —", FAMILY)
    print("=" * 78)
    if not exists(String(STORE)):
        # ⚠ SKIPPED, LOUDLY, AND NOT A PASS — the store is gitignored.
        print("  SKIPPED: no demo store at", STORE)
        print("  Build it:  pixi run libero-demo-import --no-images")
        print("=== SKIPPED (no demonstrations — this is not a pass) ===")
        return

    # ── the scene, for the observation ────────────────────────────────────
    var f = load_family(String(FAMILY_DIR) + String(FAMILY) + ".family")
    var fmd = parse_model_runtime(scene_path(f))
    var verts = 32768
    var dims = dims_from_flat(
        fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
    )
    var m = Model[H, DynDims](dims)
    while True:
        try:
            build_model_runtime[H](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(
                fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
            )
            m = Model[H, DynDims](dims)
    var d = Data[H, DynDims, 1](dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()

    # ── the tasks: one tape and mask per task index ───────────────────────
    var names = _task_names(String(FAMILY))
    var nqs = List[Int]()
    for k in range(len(fmd.joints)):
        nqs.append(fmd.joints[k].nq)
    var jadr = joint_qpos_addresses(nqs)
    var tapes = List[List[Float64]]()
    var masks = List[Float64]()
    for ti in range(len(names)):
        var t = load_task(String(TASK_DIR) + names[ti] + ".task")
        validate_task_against_family(t, f)
        var g = bind_goal(
            parse_goal(t.goal), f, fmd.body_names, fmd.site_names,
            fmd.joint_names, jadr,
        )
        require_tier_a(g, t.name)
        tapes.append(encode_goal(g))
        masks.append(active_mask(t, f))

    # ── the store ─────────────────────────────────────────────────────────
    var store = TrajectoryStore(String(STORE))
    var n_rows_all = store.n_rows()
    var n_eps = store.n_episodes()
    print("  store :", STORE, "|", n_rows_all, "rows |", n_eps, "episodes")
    var state_col = store.load_column[H](String("state"))
    var act_col = store.load_column[DType.float32](String("action"))
    var task_col = store.load_column[DType.int32](String("task_index"))
    var state_dim = nq + nv
    if len(state_col) != n_rows_all * state_dim:
        raise Error(
            "libero bc train: state column is " + String(len(state_col))
            + " words, expected " + String(n_rows_all * state_dim)
            + " — the store was written for a different scene"
        )

    # ⚠ THE SPLIT IS BY DEMONSTRATION. Episodes run task by task in the
    # importer's order, so "the last `val_demos` of each task" is the last
    # `val_demos` episodes of each task's run of episodes.
    var ep_task = List[Int](length=n_eps, fill=0)
    var per_task_eps = List[List[Int]]()
    for _ in range(len(names)):
        per_task_eps.append(List[Int]())
    for e in range(n_eps):
        var off = Int(store.episodes.ep_offset[e])
        var ti = Int(task_col[off])
        ep_task[e] = ti
        per_task_eps[ti].append(e)
    var is_val = List[Bool](length=n_eps, fill=False)
    var use_ep = List[Bool](length=n_eps, fill=True)
    for ti in range(len(names)):
        ref eps = per_task_eps[ti]
        # ⚠ THE CAP COMES FIRST AND THE SPLIT IS TAKEN INSIDE IT. `--demos 4
        # --val-demos 2` must hold out 2 OF THOSE 4, not demos 48-49 that the
        # cap already dropped — which is how the first smoke run got an empty
        # validation set and a fit with nothing to score.
        var n_use = len(eps)
        if max_demos > 0 and max_demos < n_use:
            n_use = max_demos
        if val_demos >= n_use:
            raise Error(
                "libero bc train: --val-demos " + String(val_demos) + " of "
                + String(n_use) + " usable demos leaves no training data"
            )
        for k in range(len(eps)):
            if k >= n_use:
                use_ep[eps[k]] = False
            elif k >= n_use - val_demos:
                is_val[eps[k]] = True

    # ── the observations, rebuilt frame by frame ──────────────────────────
    var t0 = perf_counter_ns()
    var x_tr = List[Scalar[DT]]()
    var y_tr = List[Scalar[DT]]()
    var x_va = List[Scalar[DT]]()
    var y_va = List[Scalar[DT]]()
    var n_tr = 0
    var n_va = 0
    for e in range(n_eps):
        if not use_ep[e]:
            continue
        var off = Int(store.episodes.ep_offset[e])
        var ln = Int(store.episodes.ep_len[e])
        var ti = ep_task[e]
        for k in range(TAPE_WORDS):
            d.meta.data[META_IDX_TASK_PARAM_0 + k] = Scalar[H](tapes[ti][k])
        d.meta.data[META_IDX_TASK_ACTIVE] = Scalar[H](masks[ti])
        for r in range(off, off + ln):
            for j in range(nq):
                d.qpos.data[j] = state_col[r * state_dim + j]
            for j in range(nv):
                d.qvel.data[j] = state_col[r * state_dim + nq + j]
            forward_kinematics["cpu", H, DynDims, 1](d, m)
            var row = List[Scalar[H]]()
            write_task_obs_host[LiberoGoalPlacement, H, DynDims](d, row)
            if len(row) != OBS:
                raise Error(
                    "libero bc train: the host writer produced "
                    + String(len(row)) + " words, the model def says "
                    + String(OBS)
                )
            if is_val[e]:
                n_va += 1
                for j in range(OBS):
                    x_va.append(Scalar[DT](row[j]))
                for j in range(ACT):
                    y_va.append(Scalar[DT](act_col[r * ACT + j]))
            else:
                n_tr += 1
                for j in range(OBS):
                    x_tr.append(Scalar[DT](row[j]))
                for j in range(ACT):
                    y_tr.append(Scalar[DT](act_col[r * ACT + j]))
    print("  frames:", n_tr, "train /", n_va, "val (held out the last",
          val_demos, "demos of each task ) in",
          Float64(perf_counter_ns() - t0) / 1e9, "s")
    # ⚠ ANTI-VACUITY ON THE REBUILT OBSERVATION. A writer that produced a
    # constant row — a stale `Data`, an FK that never ran, a goal-word block
    # left at zero — trains a policy on nothing and reports a small MSE for it.
    # The last nine words are the GOAL block (gripper xyz, subject - gripper,
    # target - subject); they move frame to frame in a demonstration.
    var moved = 0
    var goal_nonzero = 0
    for j in range(OBS):
        if x_tr[j] != x_tr[(n_tr - 1) * OBS + j]:
            moved += 1
    for j in range(OBS - 9, OBS):
        if Float64(x_tr[j]) != 0.0:
            goal_nonzero += 1
    print("         first vs last training row:", moved, "of", OBS,
          "words differ |", goal_nonzero, "of the 9 goal words nonzero")
    if moved == 0 or goal_nonzero == 0:
        raise Error(
            "libero bc train: the rebuilt observation is constant or has an"
            " empty goal block — the frames were not written"
        )
    if n_tr == 0 or n_va == 0:
        raise Error("libero bc train: an empty split — nothing to fit or score")

    # ── normalisation, from the TRAINING rows only ────────────────────────
    # ⚠ THE VALIDATION ROWS DO NOT CONTRIBUTE. A mean that saw them is a leak,
    # small here and free to avoid.
    var mu = List[Float64](length=OBS, fill=0.0)
    var sd = List[Float64](length=OBS, fill=0.0)
    for r in range(n_tr):
        for j in range(OBS):
            mu[j] += Float64(x_tr[r * OBS + j])
    for j in range(OBS):
        mu[j] /= Float64(n_tr)
    for r in range(n_tr):
        for j in range(OBS):
            var dv = Float64(x_tr[r * OBS + j]) - mu[j]
            sd[j] += dv * dv
    for j in range(OBS):
        sd[j] = (sd[j] / Float64(n_tr)) ** 0.5
        # ⚠ A CONSTANT COLUMN KEEPS SCALE 1: dividing by its zero spread would
        # write inf into every row. `libero_goal` has several (a mask word that
        # is 1 on every task, a fixture's untouched joint).
        if sd[j] < 1.0e-6:
            sd[j] = 1.0
    for r in range(n_tr):
        for j in range(OBS):
            x_tr[r * OBS + j] = Scalar[DT](
                (Float64(x_tr[r * OBS + j]) - mu[j]) / sd[j]
            )
    for r in range(n_va):
        for j in range(OBS):
            x_va[r * OBS + j] = Scalar[DT](
                (Float64(x_va[r * OBS + j]) - mu[j]) / sd[j]
            )

    # ── the two baselines every epoch is read against ─────────────────────
    var mean_act = List[Float64](length=ACT, fill=0.0)
    for r in range(n_tr):
        for j in range(ACT):
            mean_act[j] += Float64(y_tr[r * ACT + j])
    for j in range(ACT):
        mean_act[j] /= Float64(n_tr)
    var mse_zero = 0.0
    var mse_mean = 0.0
    for r in range(n_va):
        for j in range(ACT):
            var a = Float64(y_va[r * ACT + j])
            mse_zero += a * a
            var dm = a - mean_act[j]
            mse_mean += dm * dm
    mse_zero /= Float64(n_va * ACT)
    mse_mean /= Float64(n_va * ACT)
    print("  baselines (val MSE): predict ZERO", mse_zero,
          "| predict the TRAINING MEAN", mse_mean)

    # ── the fit ───────────────────────────────────────────────────────────
    seed_rng(7)
    var ctx = DeviceContext()
    var net = NET.make["gpu", Kaiming](Optional(ctx))
    var opt = Adam(lr=Scalar[DT](lr))
    # ⚠ HOST-BACKED WITH A DEVICE CELL: `Tensor.alloc(n)` then `upload`, the
    # shape every nn driver uses — the batch is built on the host and pushed.
    var bx = Tensor.alloc(BATCH * OBS)
    var by = Tensor.alloc(BATCH * ACT)
    var pred = Tensor.alloc(BATCH * ACT)
    var gout = Tensor.alloc(BATCH * ACT)
    var gin = Tensor.alloc(BATCH * OBS)
    bx.upload(ctx)
    by.upload(ctx)
    pred.upload(ctx)
    gout.upload(ctx)
    gin.upload(ctx)
    var n_batches = n_tr // BATCH
    var best_val = 1.0e30
    print("  net   : ", OBS, "->", HID, "->", HID, "->", ACT, "| Adam lr", lr,
          "| batch", BATCH, "|", n_batches, "batches/epoch")

    for ep in range(epochs):
        var tr_loss = 0.0
        for b in range(n_batches):
            # a random contiguous-free draw: BC rows are i.i.d. only across
            # demos, so the batch is sampled row-wise with replacement
            for r in range(BATCH):
                var src = Int(random_ui64(0, UInt64(n_tr - 1)))
                for j in range(OBS):
                    bx.data[r * OBS + j] = x_tr[src * OBS + j]
                for j in range(ACT):
                    by.data[r * ACT + j] = y_tr[src * ACT + j]
            bx.upload(ctx)
            by.upload(ctx)
            net.forward["gpu", BATCH](TensorRefs[1](bx), pred, Optional(ctx))
            pred.download(ctx)
            ctx.synchronize()
            # MSE and its gradient on the host: BATCH x 7 words per step, and
            # it keeps the loss arithmetic in one readable place.
            var loss = 0.0
            for k in range(BATCH * ACT):
                var e2 = Float64(pred.data[k]) - Float64(by.data[k])
                loss += e2 * e2
                gout.data[k] = Scalar[DT](2.0 * e2 / Float64(BATCH * ACT))
            tr_loss += loss / Float64(BATCH * ACT)
            gout.upload(ctx)
            net.zero_grad["gpu"](Optional(ctx))
            net.vjp["gpu", BATCH](
                TensorRefs[1](bx), gout, TensorRefs[1](gin), Optional(ctx)
            )
            opt.begin_step()
            walk_params["gpu"](net, opt, Optional(ctx))

        # ── validation, in whole batches ──────────────────────────────────
        var va_loss = 0.0
        var va_batches = n_va // BATCH
        for b in range(va_batches):
            for r in range(BATCH):
                var src = b * BATCH + r
                for j in range(OBS):
                    bx.data[r * OBS + j] = x_va[src * OBS + j]
                for j in range(ACT):
                    by.data[r * ACT + j] = y_va[src * ACT + j]
            bx.upload(ctx)
            net.forward["gpu", BATCH](TensorRefs[1](bx), pred, Optional(ctx))
            pred.download(ctx)
            ctx.synchronize()
            for k in range(BATCH * ACT):
                var e2 = Float64(pred.data[k]) - Float64(by.data[k])
                va_loss += e2 * e2
        va_loss /= Float64(va_batches * BATCH * ACT)
        print("   epoch", _pad(String(ep), 3), " train MSE",
              tr_loss / Float64(n_batches), " val MSE", va_loss,
              " (zero", mse_zero, ", mean", mse_mean, ")")
        if va_loss < best_val:
            best_val = va_loss

    # ── the artefact ──────────────────────────────────────────────────────
    var od = dirname(out_path)
    if od != "" and not exists(od):
        makedirs(od)
    save_params["gpu"](net, out_path, Optional(ctx))
    # ⚠ THE NORMALISATION IS PART OF THE POLICY. A checkpoint without it is a
    # network fed raw metres where it was trained on standardised ones; the
    # eval driver reads this sidecar and refuses a checkpoint without it.
    write_bc_norm(out_path + ".norm", mu, sd, ACT)
    print()
    print("  wrote", out_path, "and", out_path + ".norm")

    # ── the verdict ───────────────────────────────────────────────────────
    print()
    if best_val >= mse_mean:
        print("  FAIL: val MSE", best_val, ">= the TRAINING-MEAN baseline",
              mse_mean, "— the fit learned nothing")
        raise Error("libero bc train: the fit does not beat its baseline")
    print("=== best val MSE", best_val, "against zero", mse_zero, "and mean",
          mse_mean, "===")
