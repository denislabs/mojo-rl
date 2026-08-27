# +--------------------------------------------------------------------------+ #
# | ACT on the SO-ARM101 — open-loop evaluation on a held-out episode
# +--------------------------------------------------------------------------+ #
"""Walk a held-out episode step by step, query the policy at EVERY step, combine
the overlapping chunks with temporal ensembling, and compare the resulting
action stream against what was actually recorded.

    pixi run mojo build -I . -Xlinker -ld_classic -o /tmp/act_eval \\
        examples/so101/act_so101_openloop_eval.mojo && /tmp/act_eval

This is the whole inference path — chunked query, ring buffer, exponential
weighting, denormalization back to lerobot units — exercised exactly as a
deployment would drive it, without touching hardware.

## ⚠ Open-loop is not closed-loop

Observations come from the RECORDING, not from where the policy's own actions
would have taken the arm. So this measures "does the policy predict the
demonstrated action from the demonstrated observation", which is a necessary
condition for the policy working and not a sufficient one. Compounding error —
the thing action chunking exists to fight — is invisible here by construction.
Closed-loop needs the arm.

## The baselines are the point

An error number on its own says nothing. Two references are reported beside it:

* **hold** — predict the CURRENT joint positions for every future step. This is
  the trivial policy, and on a slow demonstration it is a strong one; a model
  that cannot beat it has learned nothing useful.
* **mean** — predict the dataset's mean action. Beating only this is not
  evidence of anything.

Also reported per joint, because a model can look fine in aggregate while
ignoring the gripper — the one dimension that decides whether a grasp happens.
"""

from std.python import Python, PythonObject

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.act.config import (
    ACT_TEMPORAL_ENSEMBLE_M,
    SO101_ADIM,
    SO101_IMG_H,
    SO101_IMG_W,
    SO101_N_CAM,
    SO101_QPOS,
)
from mojo_rl.deep_agents.act.data import ACTDataset
from mojo_rl.deep_agents.act.inference import TemporalEnsemble, denormalize
from mojo_rl.deep_agents.act.trainer import ACTTrainer


# Must match act_so101_train_cpu.mojo.
comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W
comptime K = 20
comptime DIM = 64
comptime HEADS = 4
comptime FF = 256
comptime LATENT = 32
comptime N_ENC = 1
comptime N_DEC = 1
comptime BATCH = 1  # one step at a time, as a deployment would

comptime T = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC,
    BATCH,
]
comptime IMG_ELEMS = N_CAM * 3 * IMG_H * IMG_W
comptime CKPT = "/tmp/act_so101_best.ckpt"

# Joint names, in the dataset's own order (meta/info.json `action.names`).
def joint_names() -> List[String]:
    var v = List[String]()
    v.append(String("shoulder_pan"))
    v.append(String("shoulder_lift"))
    v.append(String("elbow_flex"))
    v.append(String("wrist_flex"))
    v.append(String("wrist_roll"))
    v.append(String("gripper"))
    return v^


def store_path() raises -> String:
    var os = Python.import_module("os")
    var home = String(os.path.expanduser(PythonObject("~")))
    return (
        home
        + "/.cache/mojo_rl/act_so101/"
        + "DenisLabs__record-test_20260825_094319_"
        + String(IMG_H) + "x" + String(IMG_W) + ".h5"
    )


def main() raises:
    var os = Python.import_module("os")
    var path = store_path()
    if not Bool(os.path.exists(PythonObject(path))):
        print("MISSING STORE: " + path)
        raise Error("store not found")
    if not Bool(os.path.exists(PythonObject(String(CKPT)))):
        print("MISSING CHECKPOINT: " + String(CKPT))
        print("run examples/so101/act_so101_train_cpu.mojo first")
        raise Error("checkpoint not found")

    var ds = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W](String(path), seed=7)
    # ⚠ seed 7 — the SAME seed the training example used, so the split is the
    # same and this episode is genuinely held out. A different seed here would
    # silently evaluate on training data.
    var ep = ds.val_eps[0]
    var ep_len = ds.store.episodes.length_of(ep)

    var tr = T.make()
    tr.load(String(CKPT))

    print("ACT / SO-ARM101 — open-loop evaluation")
    print("  checkpoint " + String(CKPT))
    print(
        "  episode " + String(ep) + " (held out), " + String(ep_len)
        + " steps, chunk " + String(K) + ", m = "
        + String(ACT_TEMPORAL_ENSEMBLE_M)
    )
    print("")

    var qpos = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var images = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var actions = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var valid = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)
    var chunk = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var pred_n = List[Scalar[DT]](length=ADIM, fill=Scalar[DT](0.0))
    var pred = List[Scalar[DT]](length=ADIM, fill=Scalar[DT](0.0))

    var te = TemporalEnsemble[ADIM, K](m=ACT_TEMPORAL_ENSEMBLE_M)

    var sum_abs = List[Float64](length=ADIM, fill=0.0)
    var sum_hold = List[Float64](length=ADIM, fill=0.0)
    var sum_mean = List[Float64](length=ADIM, fill=0.0)
    var n = 0
    var g0 = ds.store.episodes.start_of(ep)

    for t in range(ep_len):
        ds.fill_at[K](0, ep, t, qpos, images, actions, valid)
        tr.predict(qpos, images, actions, valid, chunk)
        te.push(t, chunk, 0)
        te.action_at(t, pred_n)
        denormalize(pred_n, 0, ds.action_mean, ds.action_std, pred, 0, ADIM)

        for j in range(ADIM):
            var truth = Float64(ds.action_raw[(g0 + t) * ADIM + j])
            sum_abs[j] += abs(Float64(pred[j]) - truth)
            # `hold`: keep the current measured joint position.
            sum_hold[j] += abs(
                Float64(ds.qpos_raw[(g0 + t) * QPOS + j]) - truth
            )
            # `mean`: the dataset's average action.
            sum_mean[j] += abs(Float64(ds.action_mean[j]) - truth)
        n += 1

        if t % 50 == 0:
            print(
                "    t=" + String(t) + "  contributors "
                + String(te.n_contributors(t))
            )

    print("")
    print("  mean |error| in lerobot units (degrees; gripper 0-100)")
    print("    joint            ACT      hold      mean")
    var names = joint_names()
    var tot_act = Float64(0.0)
    var tot_hold = Float64(0.0)
    var tot_mean = Float64(0.0)
    for j in range(ADIM):
        var a = sum_abs[j] / Float64(n)
        var h = sum_hold[j] / Float64(n)
        var m = sum_mean[j] / Float64(n)
        tot_act += a
        tot_hold += h
        tot_mean += m
        var nm = names[j]
        while nm.byte_length() < 14:
            nm += " "
        print(
            "    " + nm + "  " + String(a) + "   " + String(h) + "   "
            + String(m)
        )
    print("")
    print(
        "    ALL             " + String(tot_act / Float64(ADIM)) + "   "
        + String(tot_hold / Float64(ADIM)) + "   "
        + String(tot_mean / Float64(ADIM))
    )
    print("")
    if tot_act < tot_hold:
        print("  ACT beats `hold` — the policy is using the observation.")
    else:
        print(
            "  ⚠ ACT does NOT beat `hold`. At 4 training episodes that is a"
            " likely outcome and it means the vision tower has not learned"
            " anything usable — not that the port is wrong (the M0-M7 gates"
            " cover that). More demonstrations is the lever."
        )
