#!/bin/bash
# Decode a prompt list with one model using the report's settings (decode H, voice 4200, seed 1234, cached).
#   bash decode.sh <model_dir> <prompts.jsonl> <out_dir> [selfplan|noplan]
#   env: PYTHON, SPK (4200 | row), SEED, ROWS=a:b, REPAIR=0, FISH_PYTHON (+ FISH_SPEECH_REPO, FISH_CODEC_PTH) for wavs

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ $# -ge 3 ] || { sed -n '2,/^set -e/p' "$0" | grep '^#'; exit 1; }
MODEL=$(cd "$1" && pwd)
PS=$(readlink -f "$2")
mkdir -p "$3"; OUT=$(cd "$3" && pwd)
CELL=${4:-selfplan}
PY=${PYTHON:-python}
SPK=${SPK:-4200}; SEED=${SEED:-1234}; REPAIR=${REPAIR:-1}
export MVC_NUM_SPEECH_TOKENS=2048 MVC_NUM_CODEC_LEVELS=10 MVC_ENABLE_PLAN_TOKENS=1
[ -f "$MODEL/model.safetensors" ] || { echo "no model.safetensors in $MODEL"; exit 1; }

FLAGS=$("$PY" -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["generator_flags"]))' "$MODEL/config.json")

DEC="--temp_semantic 0.7 --acoustic_temps 0.80 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 \
     --rep_penalty_levels 1.2 1.6 1.2 1.2 1.2 1.2 1.2 1.2 1.2 1.2 \
     --ras_tau 0.1 --ras_win 10 --top_p 0.8 --rep_win 16 --stop_threshold 0.02"
if [ "$REPAIR" = "1" ]; then
    DEC="$DEC --loop_break_flipk 5.0 --loop_break_win 32 --loop_break_temp 0.6 --loop_break_rep 2.0 \
         --loop_break_stop_after 24 --silence_guard_codes $HERE/data/silence_codes_fish_s2.json \
         --silence_guard_budget 10 --silence_guard_stop_after 10 --silence_guard_retries 3 \
         --silence_guard_retry_temp 0.6 --silence_guard_stop_noplan"
fi
DEC="$DEC --cached"

case "$CELL" in
  selfplan) MODE="--mode self --cot_mode plan --plan_window --plan_dur_lo 0.8 --plan_dur_hi 1.5 \
                  --plan_speech_ratio 0.660 --plan_injection" ;;
  noplan)   MODE="--mode nocot --duration_model $HERE/data/duration_model.json --dur_lo 0.9 --dur_hi 1.5" ;;
  *) echo "cell must be selfplan or noplan"; exit 1 ;;
esac
if [ -n "${ROWS:-}" ]; then
    SEL="--rows $(seq -s ' ' "${ROWS%%:*}" $(( ${ROWS##*:} - 1 )))"
else
    SEL="--limit $(grep -c . "$PS")"
fi
if [ "$SPK" = "row" ]; then SPKARG=""; else SPKARG="--force_speaker_id $SPK"; fi

cd "$HERE"
"$PY" scripts/gen_fish_cot_samples.py --val "$PS" $SEL --dtype fp32 $MODE \
    --checkpoint "$MODEL/model.safetensors" $SPKARG --seed "$SEED" --max_think 0 \
    $FLAGS $DEC --out "$OUT/$CELL.jsonl"

if [ -n "${FISH_PYTHON:-}" ]; then
    "$FISH_PYTHON" scripts/decode_fish_codes.py --codes "$OUT/$CELL.jsonl" --outdir "$OUT/$CELL"
fi
