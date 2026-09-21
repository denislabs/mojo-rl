#!/usr/bin/env bash
# Re-score ONLINE FB checkpoints at several late rungs and print the mean.
#
# Why this exists: a single checkpoint's zero-shot ratio swung 5x inside one
# run (§18.9.3 — stand 0.62–2.91 over rungs 250 k steps apart) while every
# training metric stayed flat. §18.5's first standing rule is "the mean over
# several late rungs"; the online ladder (A3 runs 1–3, A3.5, cpr_online) was
# read at ONE rung each, so its ranking is not established. This re-reads it
# from the checkpoints already on disk, at no training cost.
#
# Usage (from the project root):
#     bash examples/fb/fb_rescore.sh                       # a3c a35, 5 rungs
#     TAGS="a3c a35 cpr_online" bash examples/fb/fb_rescore.sh
#     CKPT_DIR=fb_sweep TAGS=cpr_online RUNGS=final bash examples/fb/fb_rescore.sh
#
# Knobs: TAGS, RUNGS, CKPT_DIR, OUT (csv), EVAL.
# ~4 min per rung on Apple (CPU scorer; no -e nvidia needed).

set -u

TAGS="${TAGS:-a3c a35}"
RUNGS="${RUNGS:-4000000 4250000 4500000 4750000 final}"
CKPT_DIR="${CKPT_DIR:-.}"
OUT="${OUT:-fb_rescore_results.csv}"
EVAL="${EVAL:-examples/fb/fb_eval_walker_online.mojo}"

[ -f "$OUT" ] || echo "arm,rung,task,pi_z,random,ratio" > "$OUT"

echo "======================================================================"
echo "re-score: arms [$TAGS]  rungs [$RUNGS]  from $CKPT_DIR  -> $OUT"
echo "======================================================================"

for tag in $TAGS; do
    echo "== $tag"
    for rung in $RUNGS; do
        ck="${CKPT_DIR}/fb_online_walker_${tag}.ckpt.${rung}"
        if [ ! -f "$ck" ]; then
            # ⚠ Reported, never silent: a missing rung is a smaller mean than
            # the summary claims.
            echo "   -- rung $rung ABSENT ($ck) — SKIPPED"
            continue
        fi
        log="fb_rescore_${tag}.${rung}.eval.log"
        echo "   -- eval $ck"
        pixi run mojo run -I . "$EVAL" "$ck" > "$log" 2>&1 || {
            echo "   !! eval FAILED — see $log" >&2
            continue
        }
        python3 - "$tag" "$rung" "$log" "$OUT" <<'PARSE'
import re, sys
tag, rung, log, out = sys.argv[1:5]
pat = re.compile(r"^\s*(stand|walk|run)\s*:\s*pi_z\s+(\S+)\s+random\s+(\S+)\s+ratio\s+(\S+)")
rows = [m.groups() for m in map(pat.match, open(log, encoding="utf-8", errors="replace")) if m]
if not rows:
    sys.stderr.write("   !! parsed 0 task rows from %s — eval output format changed?\n" % log)
    sys.exit(0)
with open(out, "a", encoding="utf-8") as f:
    for task, pz, rnd, ratio in rows:
        f.write("%s,%s,%s,%s,%s,%s\n" % (tag, rung, task, pz, rnd, ratio))
print("      " + "   ".join("%s %.2f" % (t, float(r)) for t, _, _, r in rows))
PARSE
    done
done

python3 - "$OUT" <<'SUM'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    by[r["arm"]][r["task"]].append((r["rung"], float(r["ratio"])))
print()
print("=" * 70)
print("mean over rungs  (per-rung values printed so the SPREAD is visible)")
print("=" * 70)
print("%-14s %-6s %6s %6s %6s   %s" % ("arm", "task", "mean", "min", "max", "rungs"))
for arm in by:
    for task in ("stand", "walk", "run"):
        v = by[arm].get(task)
        if not v:
            continue
        x = [r for _, r in v]
        print("%-14s %-6s %6.2f %6.2f %6.2f   %s"
              % (arm, task, sum(x) / len(x), min(x), max(x),
                 " ".join("%.2f" % r for r in x)))
print()
print("bars: offline 24-D pair, 3 rungs  1.39 / 2.06 / 1.51"
      "   ·  cpr_online, 5 rungs  1.74 / 1.87 / 1.28")
print("a ONE-rung number is not comparable to any of these (§18.9.3).")
SUM
