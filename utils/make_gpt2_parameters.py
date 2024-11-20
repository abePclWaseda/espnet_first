import torch
from transformers import GPT2Model

model_name = 'gpt2' 
model = GPT2Model.from_pretrained(model_name)

model_state_dict = model.state_dict()
# wte_weight = model_state_dict.pop("wte.weight")
# huggingface_pretrained_opt_lm.py を参考にしてもわからなかったので、keyに無理やりlm.decoderをつける
modified_state_dict = {}
modified_state_dict['lm.lm_head.weight'] = model_state_dict['wte.weight']
for key, value in model_state_dict.items():
    new_key = 'lm.decoder.' + key
    modified_state_dict[new_key] = value

# modified_state_dict['lm.lm_head.weight'] = wte_weight

output_path = '/mnt/kiso-qnap/abe/b4/espnet/egs2/librispeech_100/asr1/exp/lm_train_transformer_gpt2_en_hugging_face/gpt2_pretrained.pth'
torch.save(modified_state_dict, output_path)

print(f"修正されたモデルのパラメータが '{output_path}' に保存されました。")

