from espnet2.bin.asr_inference import Speech2Text

# (Pdb) p(speech2text_kwargs)
# {'asr_train_config': 'exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/config.yaml', 'asr_model_file': 'exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/valid.acc.ave.pth', 'transducer_conf': None, 'lm_train_config': None, 'lm_file': None, 'ngram_file': None, 'token_type': None, 'bpemodel': None, 'device': 'cuda', 'maxlenratio': 0.0, 'minlenratio': 0.0, 'dtype': 'float32', 'beam_size': 20, 'ctc_weight': 0.3, 'lm_weight': 0.0, 'ngram_weight': 0.9, 'penalty': 0.0, 'nbest': 1, 'normalize_length': False, 'streaming': False, 'enh_s2t_task': False, 'multi_asr': False, 'quantize_asr_model': False, 'quantize_lm': False, 'quantize_modules': ['Linear'], 'quantize_dtype': 'qint8', 'hugging_face_decoder': False, 'hugging_face_decoder_conf': {}, 'time_sync': False, 'prompt_token_file': None, 'lang_prompt_token': None, 'nlp_prompt_token': None}
speech2text = Speech2Text.from_pretrained(
    asr_train_config="/mnt/kiso-qnap/abe/b4/espnet/egs2/librispeech_100/asr1/exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/config.yaml",
    asr_model_file="/mnt/kiso-qnap/abe/b4/espnet/egs2/librispeech_100/asr1/exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/valid.acc.ave.pth",
    transducer_conf=None,
    lm_train_config=None,
    lm_file=None,
    ngram_file=None,
    token_type=None,
    bpemodel=None,
    device="cuda",
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.0,
    ngram_weight=0.9,
    penalty=0.0,
    nbest=1,
)

# (Pdb) p(speech2text_kwargs)
# {'asr_train_config': 'exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/config.yaml', 'asr_model_file': 'exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/valid.acc.ave.pth', 'transducer_conf': None, 'lm_train_config': 'exp/lm_train_transformer_gpt2_en_hugging_face/config.yaml', 'lm_file': 'exp/lm_train_transformer_gpt2_en_hugging_face/valid.loss.ave.pth', 'ngram_file': None, 'token_type': None, 'bpemodel': None, 'device': 'cuda', 'maxlenratio': 0.0, 'minlenratio': 0.0, 'dtype': 'float32', 'beam_size': 20, 'ctc_weight': 0.3, 'lm_weight': 0.1, 'ngram_weight': 0.9, 'penalty': 0.0, 'nbest': 1, 'normalize_length': False, 'streaming': False, 'enh_s2t_task': False, 'multi_asr': False, 'quantize_asr_model': False, 'quantize_lm': False, 'quantize_modules': ['Linear'], 'quantize_dtype': 'qint8', 'hugging_face_decoder': False, 'hugging_face_decoder_conf': {}, 'time_sync': False, 'prompt_token_file': None, 'lang_prompt_token': None, 'nlp_prompt_token': None, 'partial_ar': False, 'threshold_probability': 0.99, 'max_seq_len': 5, 'max_mask_parallel': -1}
speech2text_for_lm = Speech2Text.from_pretrained(
    asr_train_config="/mnt/kiso-qnap2/yuabe/b4/espnet/egs2/librispeech_100/asr1/exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/config.yaml",
    asr_model_file="/mnt/kiso-qnap2/yuabe/b4/espnet/egs2/librispeech_100/asr1/exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/valid.acc.ave.pth",
    transducer_conf=None,
    lm_train_config="/mnt/kiso-qnap2/yuabe/b4/espnet/egs2/librispeech_100/asr1/exp/lm_train_transformer_gpt2_en_hugging_face/config.yaml",
    lm_file="/mnt/kiso-qnap2/yuabe/b4/espnet/egs2/librispeech_100/asr1/exp/lm_train_transformer_gpt2_en_hugging_face/valid.loss.ave.pth",
    ngram_file=None,
    token_type=None,
    bpemodel=None,
    device="cuda",
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.0,
    ngram_weight=0.9,
    penalty=0.0,
    nbest=1,
    partial_ar=False,
    threshold_probability=0.99,
    max_seq_len=5,
    max_mask_parallel=-1
)

asr_model = speech2text.asr_model
lm = speech2text_for_lm.beam_search.scorers['lm']

print(asr_model.decoder.decoders[0].self_attn.linear_q.weight[0, 0:10])
