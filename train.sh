#!/bin/bash
# Train one model with the report's recipe (4 GPUs, effective batch 32).
#   STAGE=2 ARM=pure TIER=full INIT=<stage1p base>/model.safetensors DATA=data/stage2_v2_train.jsonl bash train.sh
#   STAGE 1p | 2, ARM pure | hybrid | transformer, TIER full (C3) | tags+plan (C2), optional VAL OUT SEED NPROC STEPS

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE=${STAGE:?STAGE=1p or 2}; ARM=${ARM:?ARM=pure|hybrid|transformer}; INIT=${INIT:?INIT=checkpoint}; DATA=${DATA:?DATA=jsonl}
export MVC_NUM_SPEECH_TOKENS=2048 MVC_NUM_CODEC_LEVELS=10 MVC_ENABLE_PLAN_TOKENS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "$ARM" in
  pure) EXTRA="" ;;
  hybrid) EXTRA="--hybrid_attention_top_k 4" ;;
  transformer) EXTRA="--backbone transformer --model_name EleutherAI/pythia-1.4b" ;;
  *) echo "ARM must be pure, hybrid or transformer"; exit 1 ;;
esac
case "$STAGE" in
  1p) TIER=tags+plan; STEPS=${STEPS:-120000}; REASONW=0.3; LR=1e-4; LRB=2e-5; WARM=1000; SAVE=20000; VALE=4000
      CFG=stage1p; LOAD="--init_from $INIT --splice_plan_vocab"; PEW="" ;;
  2)  TIER=${TIER:-full}; STEPS=${STEPS:-40000}; REASONW=1.0; LR=5e-5; LRB=1e-5; WARM=500; SAVE=10000; VALE=2000
      CFG=stage2_v2; LOAD="--init_from $INIT"
      PEW="--pause_exit_weight 3.0 --pause_codes_file data/silence_codes_fish_s2.json" ;;
  *) echo "STAGE must be 1p or 2"; exit 1 ;;
esac
VAL=${VAL:-data/${CFG}_validation.jsonl}
OUT=${OUT:-checkpoints/${STAGE}_${ARM}_${TIER//+/}}
SEEDARG=${SEED:+--seed $SEED}
python -m torch.distributed.run --standalone --nproc_per_node=${NPROC:-4} train.py \
    --stage 1 --dataset_stage auto $EXTRA \
    --plan_tier $TIER --plan_injection --plan_scaffold_dropout 0.3 \
    --plan_loss_weight 1.0 --min_plan_coverage 1.0 --plan_tag_weight 1.0 \
    --reasoning_weight $REASONW \
    --depth_module --depth_dim 1024 --depth_layers 4 --depth_arch mamba2 \
    --depth_feedback all --depth_cross_frame 0 --depth_level_decay 0.85 --depth_cond prefix \
    --codec_num_levels 10 --codebook_sizes "4096,1024,1024,1024,1024,1024,1024,1024,1024,1024" \
    --full_finetune --mtp_num_heads 0 \
    --data_path "$DATA" --val_data_path "$VAL" --output_dir "$OUT" \
    --speaker_conditioning --speaker_input_injection --num_speakers 32768 --speaker_dim 256 \
    --semantic_weight 1.5 --eos_weight 1.0 --audio_continue_weight 0.1 \
    --lr $LR --lr_min $LR --lr_backbone $LRB --lr_new_params $LR \
    --adam_eps 1e-5 --weight_decay 0.0 \
    --batch_size 1 --grad_accum 8 --max_steps $STEPS \
    --warmup_steps $WARM --save_every $SAVE --val_every $VALE \
    --max_seq_len 2048 --log_every 100 $PEW $SEEDARG $LOAD
