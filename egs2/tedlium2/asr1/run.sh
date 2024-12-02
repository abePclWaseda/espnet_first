#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test dev"

# asr_config=conf/train_asr_e_branchformer.yaml
asr_config="/mnt/kiso-qnap/abe/b4/espnet/egs2/librispeech_100/asr1/conf/tuning/train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic.yaml"
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --nj 8 \
    --ngpu 2 \
    --stage 12 \
    --stop_stage 13 \
    --gpu_inference true \
    --inference_nj 2 \
    --feats_type raw \
    --audio_format "flac.ark" \
    --hugging_face_model_name_or_path "openai-community/gpt2" \
    --token_type "hugging_face" \
    --nbpe 500 \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"
