#!/usr/bin/env bash
#
# FB hyperparameter sweep — the one §14 item 5 asks for, with the target values
# §16.3 supplies.
#
# Until 2026-08-18 this was a blind search: `bc_weight`, `d` and `ortho_weight`
# were first guesses and nothing said what they SHOULD be. Reading BFM-Zero
# (arXiv 2511.04131) and its released `fb_cpr/configs.py` turned two of them
# into a comparison against published values:
#
#     ortho_coef   they ship 100      we run 1.0     <- factor of 100
#     lr_B         they ship 1e-5     we run 3e-4    <- one lr for all four
#     obs norm     BatchNorm1d        we run raw qpos|qvel
#
# ⚠⚠ **EVERY ARM VARIES ONE THING.** The temptation is to run the three
# reference values together and see a big number; that arm cannot tell you
# which change produced it, and if two changes act in opposite directions it
# reports "no effect" for both. Combine only AFTER the singles are read, by
# adding an arm here and re-running — the script skips arms it has already
# finished, so that costs one arm, not the sweep.
#
# ⚠⚠ **EACH ARM IS SCORED OVER SEVERAL LATE RUNGS, NOT ITS LAST CHECKPOINT.**
# §14's second standing caution: across 11 rungs of one 1.22 M run the same
# task ranged 0.94 to 2.41. Scoring on the final checkpoint alone reports the
# best (or worst) of N and calls it the arm.
#
# Usage:
#     bash examples/fb/fb_sweep.sh                 # all arms, 300 k steps each
#     STEPS=100000 bash examples/fb/fb_sweep.sh    # a quick shakeout
#     ARMS="base ortho100" bash examples/fb/fb_sweep.sh    # a subset
#     FORCE=1 bash examples/fb/fb_sweep.sh         # re-run finished arms
#
# Budget: an arm is ~17 min of training at 300 k steps (3.37 ms/step) plus
# ~3 CPU evals. Six arms is an afternoon, which is what §14 item 5 costs out.

set -euo pipefail
# ⚠ The French locale prints decimals with a comma and every numeric compare in
# the summary silently becomes a string compare. This bit once already.
export LC_ALL=C

cd "$(dirname "$0")/../.."

STEPS=${STEPS:-300000}
PIXI_ENV=${PIXI_ENV:-nvidia}
# ⚠ EXISTENCE CHECK ONLY — this does NOT select the store. Both
# `fb_train_gpu.mojo` and `fb_eval_walker.mojo` read their store from a
# COMPTIME constant, deliberately: a `--store` flag on one of them would let an
# arm train on one dataset and infer z from another, and the ratios would still
# print. Change the store by editing both constants, together.
STORE=${STORE:-fb_walker_all_sac.h5}
OUT=${OUT:-fb_sweep_results.csv}
# Late region only. At CKPT_EVERY=50000 a 300 k run writes 50k..250k plus
# `.final`; the first rungs are still moving and averaging them in would
# penalise every arm equally but noisily.
RUNGS=${RUNGS:-"200000 250000 final"}
FORCE=${FORCE:-0}

# tag : flags. `base` MUST stay first — the summary reports every other arm as
# a delta against it, and an arm compared against nothing is not a measurement.
declare -a ARM_TAGS=(base       ortho100      lrb1e5          obsnorm          bc0p3      bc3p0)
declare -a ARM_FLAG=(""         "--ortho 100" "--lr-b 1e-5"   "--obs-norm 1"   "--bc 0.3" "--bc 3.0")

ARMS=${ARMS:-"${ARM_TAGS[*]}"}

if [ ! -f "$STORE" ]; then
    echo "FATAL: store '$STORE' not found. Run examples/fb/collect_walker_all.mojo first." >&2
    exit 1
fi

echo "sweep: steps=$STEPS  env=$PIXI_ENV  store=$STORE  rungs='$RUNGS'"
echo "       arms: $ARMS"
echo ""

if [ ! -f "$OUT" ]; then
    echo "arm,flags,rung,task,pi_z,random,ratio" > "$OUT"
fi

skipped_rungs=0

for i in "${!ARM_TAGS[@]}"; do
    tag="${ARM_TAGS[$i]}"
    flags="${ARM_FLAG[$i]}"
    case " $ARMS " in *" $tag "*) ;; *) continue ;; esac

    ckpt="fb_walker_${tag}.ckpt"
    if [ -f "${ckpt}.final" ] && [ "$FORCE" != "1" ]; then
        echo "=== arm '$tag' — already trained (${ckpt}.final exists), skipping training"
    else
        echo "=== arm '$tag' — training ${STEPS} steps  ${flags}"
        # shellcheck disable=SC2086
        pixi run -e "$PIXI_ENV" mojo run -I . examples/fb/fb_train_gpu.mojo \
            --tag "$tag" --steps "$STEPS" $flags \
            2>&1 | tee "fb_sweep_${tag}.train.log" | grep -E "^\[|^   step (0|[0-9]*000) " || true
    fi

    for rung in $RUNGS; do
        ck="${ckpt}.${rung}"
        if [ ! -f "$ck" ]; then
            # ⚠ Reported, never silent. A rung that does not exist because
            # STEPS was lowered below it is a smaller sweep than the header
            # claims, and the summary must not read as full coverage.
            echo "    -- rung $rung absent for '$tag' (STEPS=$STEPS) — SKIPPED"
            skipped_rungs=$((skipped_rungs + 1))
            continue
        fi
        echo "    -- eval $ck"
        pixi run mojo run -I . examples/fb/fb_eval_walker.mojo "$ck" \
            > "fb_sweep_${tag}.${rung}.eval.log" 2>&1 || {
                echo "    !! eval FAILED for $ck — see fb_sweep_${tag}.${rung}.eval.log" >&2
                continue
            }
        python3 - "$tag" "$flags" "$rung" "fb_sweep_${tag}.${rung}.eval.log" "$OUT" <<'PARSE'
import re, sys
tag, flags, rung, log, out = sys.argv[1:6]
pat = re.compile(r"^\s*(stand|walk|run)\s*:\s*pi_z\s+(\S+)\s+random\s+(\S+)\s+ratio\s+(\S+)")
rows = []
for line in open(log, encoding="utf-8", errors="replace"):
    m = pat.match(line)
    if m:
        rows.append(m.groups())
if not rows:
    sys.stderr.write("    !! parsed 0 task rows from %s — eval output format changed?\n" % log)
    sys.exit(0)
with open(out, "a", encoding="utf-8") as f:
    for task, pz, rnd, ratio in rows:
        f.write("%s,%s,%s,%s,%s,%s,%s\n" % (tag, flags.replace(",", " "), rung, task, pz, rnd, ratio))
print("       " + "  ".join("%s %s" % (t, r) for t, _, _, r in rows))
PARSE
    done
    echo ""
done

echo "=== summary (mean ratio over the late rungs, delta vs 'base') ==="
python3 - "$OUT" "$skipped_rungs" <<'SUMMARY'
import csv, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
skipped = int(sys.argv[2])

agg = defaultdict(list)
for r in rows:
    try:
        agg[(r["arm"], r["task"])].append(float(r["ratio"]))
    except ValueError:
        pass

arms, tasks = [], []
for (a, t) in agg:
    if a not in arms:
        arms.append(a)
    if t not in tasks:
        tasks.append(t)
tasks = [t for t in ("stand", "walk", "run") if t in tasks]
if "base" in arms:
    arms = ["base"] + [a for a in arms if a != "base"]

def mean(v):
    return sum(v) / len(v) if v else float("nan")

w = max([len(a) for a in arms] + [8])
print("")
print("  %-*s  %s" % (w, "arm", "  ".join("%-16s" % t for t in tasks)))
base = {t: mean(agg.get(("base", t), [])) for t in tasks}
for a in arms:
    cells = []
    for t in tasks:
        vals = agg.get((a, t), [])
        m = mean(vals)
        if a == "base" or base.get(t) != base.get(t):   # nan-safe
            cells.append("%-16s" % ("%.3f (n=%d)" % (m, len(vals))))
        else:
            cells.append("%-16s" % ("%.3f (%+.3f)" % (m, m - base[t])))
    print("  %-*s  %s" % (w, a, "  ".join(cells)))
print("")
print("  n = rungs averaged. A delta smaller than the spread ACROSS rungs of")
print("  one arm is not a result — open fb_sweep_results.csv and look at the")
print("  per-rung values before believing any row here.")
if skipped:
    print("")
    print("  ⚠ %d rung(s) were absent and skipped. This table covers LESS" % skipped)
    print("    than the configured rung set; see the '-- rung ... SKIPPED' lines.")
SUMMARY

echo ""
echo "per-rung rows -> $OUT"
echo "training logs -> fb_sweep_<arm>.train.log ; eval logs -> fb_sweep_<arm>.<rung>.eval.log"
