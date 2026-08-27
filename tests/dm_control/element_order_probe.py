"""Which suite models have joints/sites whose XML TEXT order differs from
MuJoCo's compiled (body-grouped) order?

That difference is exactly the exposure condition for the latent defect in
`full_parser`: `_find_joint_index_by_name` / `_find_site_index_by_name` count
tags in the raw text, and those lookups run AFTER `_stable_group_by_body_*`
has already permuted the arrays the returned index is used against.

Measured against dm_control's own suite XMLs (our ports are transcriptions of
these), compiled by the pixi MuJoCo runtime.
"""
import glob
import os
import re
import sys

import mujoco

SUITE = "references/dm_control-main/dm_control/suite"


def strip_comments(s):
    return re.sub(r"<!--.*?-->", "", s, flags=re.S)


def text_order(xml, *tags):
    """Names of the given elements inside <worldbody>, in raw text order.

    ⚠ ONE pass over an alternation, not one pass per tag. Scanning `<joint`
    and `<freejoint` separately and concatenating puts every free joint at the
    END of the list, which manufactures a divergence at index 0 for every
    model with a floating root — dog, humanoid, humanoid_CMU and quadruped all
    reported one on the first draft of this probe. That was the instrument,
    not the model.
    """
    wb = xml.find("<worldbody")
    if wb == -1:
        return []
    end = xml.find("</worldbody>", wb)
    body = xml[wb:end if end != -1 else len(xml)]
    alt = "|".join(tags)
    out = []
    for m in re.finditer(r"<(?:%s)(\s[^>]*)?/?>" % alt, body):
        attrs = m.group(1) or ""
        nm = re.search(r'name="([^"]*)"', attrs)
        out.append(nm.group(1) if nm else "<unnamed>")
    return out


def compiled_order(m, objtype, n):
    return [mujoco.mj_id2name(m, objtype, i) or "<unnamed>" for i in range(n)]


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    return None


rows = []
for path in sorted(glob.glob(os.path.join(SUITE, "*.xml"))):
    name = os.path.basename(path)
    raw = open(path).read()
    try:
        m = mujoco.MjModel.from_xml_path(path)
    except Exception as e:
        rows.append((name, "SKIP", str(e).splitlines()[0][:60], ""))
        continue
    xml = strip_comments(raw)

    # `<freejoint>` is sugar our parser normalizes to `<joint type="free">`,
    # so it must be scanned IN POSITION alongside `<joint`.
    jt = text_order(xml, "joint", "freejoint")
    jc = compiled_order(m, mujoco.mjtObj.mjOBJ_JOINT, m.njnt)
    st = text_order(xml, "site")
    sc = compiled_order(m, mujoco.mjtObj.mjOBJ_SITE, m.nsite)

    jdiv = first_divergence(jt, jc) if len(jt) == len(jc) else "LEN"
    sdiv = first_divergence(st, sc) if len(st) == len(sc) else "LEN"

    def fmt(d, t, c):
        if d == "LEN":
            return f"len {len(t)} vs {len(c)}"
        if d is None:
            return "same"
        return f"DIFFERS at {d[0]}: text {d[1]!r} vs mj {d[2]!r}"

    rows.append((name, f"njnt {m.njnt}", fmt(jdiv, jt, jc),
                 f"nsite {m.nsite} " + fmt(sdiv, st, sc)))

w = max(len(r[0]) for r in rows)
for r in rows:
    print(f"{r[0]:<{w}}  {r[1]:<10}  joints: {r[2]:<52}  sites: {r[3]}")
