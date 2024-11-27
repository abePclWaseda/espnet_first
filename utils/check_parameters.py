import torch

# モデルファイルのパス
model_file = "egs2/librispeech_100/asr1/exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp_1117GPT2Decoder/valid.acc.ave.pth"
# model_lm_file = "/mnt/kiso-qnap/abe/b4/espnet/egs2/librispeech_100/asr1/exp/lm_train_transformer_gpt2_en_hugging_face/gpt2_pretrained.pth" # gpt2のパラメータがロードできているかの確認に使える.

# state_dictのロード
state_dict = torch.load(model_file, map_location="cpu")

# クロスアテンション層のキーが存在するか確認
cross_attention_keys = [k for k in state_dict.keys() if "crossattention" in k]
# print(f"クロスアテンション層のパラメータ数: {len(cross_attention_keys)}")
# print(f"クロスアテンション層のキー: {cross_attention_keys}")
# print(state_dict.keys())

# decoder.decoder のキーを取得
decoder_keys = [key for key in state_dict.keys() if key.startswith("decoder.decoder")]
print(f"decoder.decoder のキー: {decoder_keys}")


# 必要に応じて特定のパラメータ値を表示（最初の10個の値を表示する例）
# if cross_attention_keys:
#     for key in cross_attention_keys:
#         print(f"{key}: {state_dict[key].flatten()[:10]}") 

# specific_key = "decoder.decoder.h.0.attn.c_attn.weight"
# if specific_key in state_dict.keys():
#     print(f"{specific_key}: {state_dict[specific_key].flatten()[0:10]}")
# else:
#     print(f"{specific_key} はキーに含まれていません。")