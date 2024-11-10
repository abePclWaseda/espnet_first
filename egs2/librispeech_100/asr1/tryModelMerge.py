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
    lm_weight=0.1,
    ngram_weight=0.9,
    penalty=0.0,
    nbest=1,
    # partial_ar=False,
    # threshold_probability=0.99,
    # max_seq_len=5,
    # max_mask_parallel=-1 デコーダを作成したESPnet(kiso-qnap/abe)のバージョンが古く, これらの引数を受け付けない.
)

asr_model = speech2text.asr_model
lm = speech2text_for_lm.beam_search.scorers['lm']
# import pdb;pdb.set_trace()
print(asr_model.decoder.decoder.h[0].attn.c_attn.weight[0, 0:10])
print(lm.decoder.h[0].attn.c_attn.weight[0, 0:10])

import soundfile
speech, rate = soundfile.read("/mnt/kiso-qnap/abe/b4/espnet/egs2/librispeech_100/asr1/e_thankyou_02.wav")
nbests = speech2text(speech)

print([x[0] for x in nbests])

import torch
num_layers = len(asr_model.decoder.decoder.h)
assert num_layers == len(lm.decoder.h), "レイヤー数が一致しません。"

with torch.no_grad():
    for i in range(num_layers):
        asr_layer = asr_model.decoder.decoder.h[i]
        lm_layer = lm.decoder.h[i]
        
        asr_attn = asr_layer.attn
        lm_attn = lm_layer.attn
        
        asr_attn_params = asr_attn.state_dict()
        lm_attn_params = lm_attn.state_dict()
        
        for name in asr_attn_params:
            if name in lm_attn_params:
                asr_param = asr_attn_params[name]
                lm_param = lm_attn_params[name].to(asr_param.device)
                averaged_param = (asr_param + lm_param) / 2
                asr_attn_params[name] = averaged_param
            else:
                print(f"警告: {name} が lm_attn_params に存在しません。")
        
        asr_attn.load_state_dict(asr_attn_params)

print(asr_model.decoder.decoder.h[0].attn.c_attn.weight[0, 0:10])

nbests = speech2text(speech)

print([x[0] for x in nbests])