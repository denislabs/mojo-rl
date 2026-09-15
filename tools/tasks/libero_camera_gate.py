#!/usr/bin/env python
"""How far are our pixels from LIBERO's? — the L5 gate, in four columns.

    pixi run libero-camera-gate                        # dump + the numbers
    pixi run libero-camera-gate --suite libero_goal --frame 0

`docs/LIBERO_PORT_ASSESSMENT_2026_09_13.md` §6 asks for exactly this: "render
frame 0 of a demo from its `states[0]` and compare to `obs/agentview_rgb[0]`;
report PSNR and the fraction of pixels within 8/255. A number, not a
screenshot. This will not be identity (OpenGL vs a raytracer) and the point of
the gate is to know the size of the gap before a policy is trained on either
domain."

A two-column version of that would be undecidable. Our tracer differs from the
recorded frame for at least four independent reasons — a different renderer, a
different MuJoCo, a scene we composed ourselves, and our own shading — and one
PSNR cannot say which. So this prints FOUR columns and each gap between two of
them isolates one cause (`_a_third_column_makes_a_sim_to_sim_gate_decisive`):

  RECORDED   `obs/agentview_rgb[i]`, robosuite 1.4.0 + its MuJoCo, OpenGL.
  THEIRS     MuJoCo 3.12 (our runtime) rendering THEIR recorded `model_file`
             at the same state. RECORDED-vs-THEIRS is the renderer-and-version
             floor: no work on our side can beat it.
  OURS-MJ    MuJoCo 3.12 rendering OUR composed `.family` scene at the same
             state, this demo's fixture draw patched in. THEIRS-vs-OURS-MJ is
             what our SCENE costs — assets, poses, composition — with the
             renderer held fixed.
  OURS-RT    our batched raytracer on our scene, printed by the Mojo leg
             (`examples/tasks/libero_camera_gate.mojo`). OURS-MJ-vs-OURS-RT is
             what our SHADING costs with everything else held fixed.

This half writes the first three and a `.dump` the Mojo leg reads.

⚠⚠ `agentview_rgb[i]` IS THE FRAME AT `states[i + 1]`, NOT `states[i]`.
`LIBERO-master/scripts/create_dataset.py` appends `obs` from AFTER
`env.step(action)` at index j and records `states[valid_index[j]]`, which is
the state BEFORE that step. Lining the two up naively costs 6 dB — measured,
and asserted here on every run rather than trusted: `--check-alignment` renders
both and refuses if the +1 is not the better match.

⚠ THE RECORDED IMAGE IS UPSIDE DOWN RELATIVE TO `mujoco.Renderer`. robosuite
flips its camera observations; `Renderer.render()` does not. Every comparison
here flips ours, and the same assertion covers it: unflipped scores 18 dB
against 44 dB, so a silent flip could not survive the alignment check.

⚠ ONLY GEOM GROUP 1 IS RENDERED. robosuite runs with
`render_collision_mesh=False`, so its pictures contain the VISUAL meshes and
none of the collision primitives that sit inside them. Rendering MuJoCo's
default group set instead scores 10.8 dB against 43.9 — the collision boxes
are opaque and they are in front. Our tracer inherits the same rule.
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero_demo_common import (  # noqa: E402
    DEMOS,
    SCENES,
    body_names,
    build_remap,
    convert_state,
    fixture_poses,
    rewrite,
)

CAM_THEIRS = "agentview"
CAM_OURS = "arena_agentview"
GROUP_VISUAL = 1


def visual_only():
    """MuJoCo's scene options with only the VISUAL geom group on, and no sites.

    ⚠ SITES OFF HERE; `libero_sites` TURNS ON THE ONE LIBERO SHOWS. LIBERO's
    fixtures carry marker sites with an opaque rgba in the XML — `flat_stove`'s
    `burner` is `0.9 0.05 0.05 1` — and LIBERO writes `site_rgba[..][3]` every
    step (`set_visualization` in each problem class): 0 while the stove is off,
    1 once its knob is at 0.5 or more. Rendering the XML's alpha puts a red disc
    on every hob; rendering none misses the lit burner that
    `turn_on_the_stove`'s last frames show.
    """
    o = mujoco.MjvOption()
    o.geomgroup[:] = 0
    o.geomgroup[GROUP_VISUAL] = 1
    o.sitegroup[:] = 0
    return o


def libero_sites(m, d, opt):
    """LIBERO's runtime site visibility, applied to `m.site_rgba` for `d`.

    The same rule as `mojo_rl/tasks/libero_visual.mojo`: a `*burner` site is
    visible when its sibling `*button` hinge is at 0.5 or more
    (`FlatStove.turn_on`), and every other site is hidden — robosuite hides the
    robot's at construction. Enables site group 0, where the burner lives."""
    opt.sitegroup[0] = 1
    for i in range(m.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or ""
        alpha = 0.0
        if name.endswith("burner"):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name[: -len("burner")] + "button")
            if j >= 0 and d.qpos[m.jnt_qposadr[j]] >= 0.5:
                alpha = 1.0
        m.site_rgba[i][3] = alpha


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0 * 255.0 / mse)


def within(a, b, tol=8):
    return float((np.abs(a.astype(np.int32) - b.astype(np.int32)) <= tol).mean())


def render(m, d, r, opt, cam, qpos, qvel):
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    mujoco.mj_forward(m, d)
    libero_sites(m, d, opt)
    r.update_scene(d, camera=cam, scene_option=opt)
    # `Renderer` hands back row 0 = top; the recording is flipped. See header.
    return r.render()[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--frame", type=int, default=0,
                    help="index into obs/agentview_rgb (see the header's off-by-one)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", action="store_true",
                    help="also write RECORDED/THEIRS/OURS-MJ as PNGs beside the dumps")
    a = ap.parse_args()

    out_dir = a.out or os.path.join(DEMOS, "_camera", a.suite)
    os.makedirs(out_dir, exist_ok=True)

    scene = os.path.join(SCENES, a.suite + ".xml")
    mo = mujoco.MjModel.from_xml_path(scene)
    do = mujoco.MjData(mo)
    our_b = body_names(mo)
    if mujoco.mj_name2id(mo, mujoco.mjtObj.mjOBJ_CAMERA, CAM_OURS) < 0:
        sys.exit(f"{scene} has no camera {CAM_OURS!r} — is the arena regenerated?")

    files = sorted(glob.glob(os.path.join(DEMOS, a.suite, "*_demo.hdf5")))
    if not files:
        sys.exit(f"no demo files under {os.path.join(DEMOS, a.suite)}")

    opt = visual_only()
    print(f"{a.suite}: {len(files)} tasks, {a.demo}, frame {a.frame}")
    print(f"  our scene {scene}  nq {mo.nq} nv {mo.nv} ngeom {mo.ngeom}")
    print()
    print(f"  {'task':<50} {'THEIRS':>16} {'OURS-MJ':>16} {'OURS-MJ|THEIRS':>16}")
    print(f"  {'':<50} {'psnr  within8':>16} {'psnr  within8':>16} {'psnr  within8':>16}")

    written = []
    sum_theirs = []
    sum_ours = []
    checked_alignment = False
    for path in files:
        stem = os.path.basename(path)[: -len("_demo.hdf5")]
        f = h5py.File(path, "r")
        g = f["data"][a.demo]
        states = np.array(g["states"])
        rgb = np.array(g["obs"]["agentview_rgb"])
        h, w = rgb.shape[1], rgb.shape[2]
        if a.frame + 1 >= len(states):
            sys.exit(f"{stem}: frame {a.frame} needs states[{a.frame + 1}] of {len(states)}")
        recorded = rgb[a.frame]

        xml = os.path.join(out_dir, "_m.xml")
        with open(xml, "w") as fh:
            fh.write(rewrite(g.attrs["model_file"]))
        mt = mujoco.MjModel.from_xml_path(xml)
        dt = mujoco.MjData(mt)
        rt = mujoco.Renderer(mt, h, w)
        remap = build_remap(mt, mo)

        row = states[a.frame + 1]
        theirs = render(mt, dt, rt, opt, CAM_THEIRS,
                        row[1:1 + mt.nq], row[1 + mt.nq:])

        # ── the alignment assertion, once, on the first task ──────────────
        if not checked_alignment:
            prev = states[a.frame]
            off = render(mt, dt, rt, opt, CAM_THEIRS,
                         prev[1:1 + mt.nq], prev[1 + mt.nq:])
            p1, p0 = psnr(theirs, recorded), psnr(off, recorded)
            pf = psnr(theirs[::-1], recorded)
            print(f"  [alignment] states[i+1] {p1:.2f} dB   states[i] {p0:.2f} dB"
                  f"   unflipped {pf:.2f} dB")
            if not (p1 > p0 and p1 > pf):
                sys.exit(
                    "the frame alignment this gate assumes is not the best one:"
                    f" states[i+1] {p1:.2f}, states[i] {p0:.2f}, unflipped {pf:.2f}."
                    " Read the header before changing it."
                )
            checked_alignment = True

        # ── OURS-MJ: our scene, this demo's fixture draw, the same state ──
        fix = fixture_poses(mt, our_b)
        for ob, bp, bq in fix:
            b = mujoco.mj_name2id(mo, mujoco.mjtObj.mjOBJ_BODY, ob)
            mo.body_pos[b] = bp
            mo.body_quat[b] = bq
        qo, vo = convert_state(row, remap, mt, mo)
        ro = mujoco.Renderer(mo, h, w)
        ours_mj = render(mo, do, ro, opt, CAM_OURS, qo, vo)

        pt, wt = psnr(theirs, recorded), within(theirs, recorded)
        pm, wm = psnr(ours_mj, recorded), within(ours_mj, recorded)
        pc, wc = psnr(ours_mj, theirs), within(ours_mj, theirs)
        sum_theirs.append((pt, wt))
        sum_ours.append((pm, wm))
        print(f"  {stem:<50} {pt:7.2f} {wt:8.3f} {pm:7.2f} {wm:8.3f} {pc:7.2f} {wc:8.3f}")

        # ── what the Mojo leg reads ───────────────────────────────────────
        ref = os.path.join(out_dir, stem + ".rgb")
        with open(ref, "wb") as fh:
            fh.write(recorded.astype(np.uint8).tobytes())
        # ⚠ THE VISUAL-GEOM NUMBERING IS `build_visual_model`'S: the geoms of
        # the render group, in model order, from 0. `+1` leaves 0 for the
        # background, which is what our tracer writes for a miss.
        segopt = visual_only()
        segopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        ro_seg = mujoco.Renderer(mo, h, w)
        ro_seg.enable_segmentation_rendering()
        do.qpos[:] = qo
        do.qvel[:] = vo
        mujoco.mj_forward(mo, do)
        ro_seg.update_scene(do, camera=CAM_OURS, scene_option=segopt)
        segimg = ro_seg.render()[:, :, 0]
        keep = [i for i in range(mo.ngeom) if mo.geom_group[i] == GROUP_VISUAL]
        remap = {g: k + 1 for k, g in enumerate(keep)}
        segvis = np.zeros(segimg.shape, np.uint8)
        for gid, vid in remap.items():
            segvis[segimg == gid] = vid
        with open(os.path.join(out_dir, stem + ".mj.seg"), "wb") as fh:
            fh.write(segvis.tobytes())

        mjref = os.path.join(out_dir, stem + ".mj.rgb")
        with open(mjref, "wb") as fh:
            fh.write(ours_mj.astype(np.uint8).tobytes())
        lines = [f"WIDTH {w}", f"HEIGHT {h}", f"CAMERA {CAM_OURS}",
                 f"GROUP {GROUP_VISUAL}",
                 f"RECORDED {ref}", f"OURSMJ {mjref}",
                 "QPOS " + " ".join(repr(float(x)) for x in qo),
                 "QVEL " + " ".join(repr(float(x)) for x in vo)]
        for ob, bp, bq in fix:
            lines.append("FIX " + ob + " "
                         + " ".join(repr(float(x)) for x in bp) + " "
                         + " ".join(repr(float(x)) for x in bq))
        dump = os.path.join(out_dir, stem + ".dump")
        with open(dump, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        written.append((stem, dump))

        if a.png:
            import imageio.v2 as iio
            iio.imwrite(os.path.join(out_dir, stem + ".recorded.png"), recorded)
            iio.imwrite(os.path.join(out_dir, stem + ".theirs.png"), theirs)
            iio.imwrite(os.path.join(out_dir, stem + ".ours_mj.png"), ours_mj)
        os.remove(xml)
        f.close()

    def mean(rows, k):
        return sum(r[k] for r in rows) / len(rows)

    print()
    print(f"  {'MEAN':<50} {mean(sum_theirs,0):7.2f} {mean(sum_theirs,1):8.3f}"
          f" {mean(sum_ours,0):7.2f} {mean(sum_ours,1):8.3f}")
    index = os.path.join(out_dir, "index.txt")
    with open(index, "w") as fh:
        for stem, dump in written:
            fh.write(f"{a.suite}__{stem} {dump}\n")
    print()
    print(f"wrote {index}")
    print("now:  pixi run mojo run -I . examples/tasks/libero_camera_gate.mojo "
          + index)


if __name__ == "__main__":
    main()
