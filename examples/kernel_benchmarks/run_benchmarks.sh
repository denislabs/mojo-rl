#!/usr/bin/env bash
# Run all kernel benchmark builds and report compilation time for each group.
#
# Usage (from mojo-rl/ directory):
#   bash examples/kernel_benchmarks/run_benchmarks.sh
#   bash examples/kernel_benchmarks/run_benchmarks.sh nvidia   # NVIDIA GPU
#
# Each group compiles in isolation so the times are independent.
# Mojo caches compiled artifacts; add -c to clear cache first if needed.

set -euo pipefail
cd "$(dirname "$0")/../.."  # cd to mojo-rl/

ENV="${1:-apple}"
BUILD_FLAGS="-o /dev/null"

GROUPS=(
    "A: Simple data/gradient kernels     (build_za, extract_obs/act/rew, consistency_grad, bce_grad, policy_grad)"
    "B: Distributional RL kernels        (two_hot_grad, q_decode, td_targets, decode_and_min)"
    "C: Gradient clipping + soft-update  (gradient_norm ×5 sizes, gradient_reduce_apply ×5, soft_update)"
    "D: Linear layer kernels             (tiled matmul: 6 unique shapes × fwd_no_cache + fwd + bwd)"
    "E: NormedLinear + SimNorm + Sigmoid (LayerNorm + Mish + SimNorm + Sigmoid fwd/bwd)"
    "F: Env collection kernels           (random_actions, sample_actions, accumulate_rewards, etc.)"
)

FILES=(
    "examples/kernel_benchmarks/bench_a_tdmpc2_simple.mojo"
    "examples/kernel_benchmarks/bench_b_tdmpc2_distributional.mojo"
    "examples/kernel_benchmarks/bench_c_gradient_clip.mojo"
    "examples/kernel_benchmarks/bench_d_linear_kernels.mojo"
    "examples/kernel_benchmarks/bench_e_normed_linear_kernels.mojo"
    "examples/kernel_benchmarks/bench_f_env_collection.mojo"
)

echo "================================================================"
echo "TDMPC2 Kernel Compilation Benchmark  (env: $ENV)"
echo "================================================================"
echo ""

RESULTS=()

for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    DESC="${GROUPS[$i]}"
    LABEL="Group ${DESC:0:1}"

    printf "Building %-10s %s\n" "$LABEL" "$FILE"
    printf "  %s\n" "$DESC"

    START=$(date +%s%3N)
    if pixi run -e "$ENV" mojo build "$FILE" $BUILD_FLAGS 2>&1 | tail -3; then
        END=$(date +%s%3N)
        ELAPSED=$(( (END - START) ))
        RESULTS+=("$(printf "  %-10s %6d ms   %s" "$LABEL" "$ELAPSED" "${DESC:3}")")
        printf "  -> %d ms\n\n" "$ELAPSED"
    else
        RESULTS+=("  $LABEL  FAILED")
        echo "  -> FAILED"
        echo ""
    fi
done

echo "================================================================"
echo "SUMMARY (wall-clock compilation time)"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    echo "$r"
done
echo ""
echo "Sorted by compile time (slowest first):"
printf '%s\n' "${RESULTS[@]}" | sort -t'.' -k2 -rn
