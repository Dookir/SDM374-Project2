#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
    echo "Usage: $0 DATASET CHECKPOINT OUTPUT_DIR LEHD_SECONDS_PER_INSTANCE" >&2
    exit 2
fi

cd "$(dirname "$0")"
dataset=$1
checkpoint=$2
output=$3
reference_seconds=$4
python_bin=${PROJECT2_PYTHON:-/opt/conda/bin/python}
lkh=${PROJECT2_LKH:-/nfs1/code/vendor/LKH3/LKH}
export CONDA_PKGS_DIRS=/nfs1/conda/pkgs
export PIP_CACHE_DIR=/nfs1/.cache/pip
export TMPDIR=/nfs1/.tmp
export MPLCONFIGDIR=/nfs1/.cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$output" "$TMPDIR" "$MPLCONFIGDIR"
common=(--dataset "$dataset" --lkh "$lkh" --scratch "$TMPDIR")
if [[ -n ${PROJECT2_BASELINE:-} ]]; then
    common+=(--baseline-jsonl "$PROJECT2_BASELINE")
fi
if [[ -n ${PROJECT2_LIMIT:-} ]]; then
    common+=(--limit "$PROJECT2_LIMIT")
fi
if [[ ${PROJECT2_TSPLIB:-0} == 1 ]]; then
    common+=(--test_in_tsplib true)
fi

for multiplier in 1 4 16; do
    seconds=$(awk -v base="$reference_seconds" -v multiplier="$multiplier" 'BEGIN { print base * multiplier }')
    for mode in native static dynamic; do
        model=()
        if [[ "$mode" != native ]]; then model=(--checkpoint "$checkpoint"); fi
        "$python_bin" -u test_tsp.py "${common[@]}" "${model[@]}" \
            --mode "$mode" --seconds "$seconds" --output "$output/${mode}-${multiplier}x"
    done
done

for variant in unguided random neural lkh; do
    mode=$variant
    init=both
    if [[ "$variant" == neural || "$variant" == lkh ]]; then
        mode=dynamic
        init=$variant
    fi
    "$python_bin" -u test_tsp.py "${common[@]}" --checkpoint "$checkpoint" --mode "$mode" \
        --initialization "$init" --seconds "$reference_seconds" --output "$output/ablation-$variant"
done

if [[ -n ${PROJECT2_EAX:-} ]]; then
    "$python_bin" -u test_tsp.py "${common[@]}" --mode eax --eax "$PROJECT2_EAX" \
        --seconds "$reference_seconds" --output "$output/eax-1x"
fi
if [[ -n ${PROJECT2_NEUROLKH:-} ]]; then
    "$python_bin" -u test_tsp.py "${common[@]}" --mode neurolkh --neurolkh "$PROJECT2_NEUROLKH" \
        --seconds "$reference_seconds" --output "$output/neurolkh-1x"
fi

"$python_bin" -m neural_tsp.quality_curve --summary \
    "$output"/native-*x/summary.json "$output"/static-*x/summary.json "$output"/dynamic-*x/summary.json \
    --output "$output/quality-curve.png"
