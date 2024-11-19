import torch
from transformers import GPT2Model

model_name = 'gpt2' 
model = GPT2Model.from_pretrained(model_name)

model_state_dict = model.state_dict()

output_path = '/mnt/kiso-qnap/abe/b4/espnet/egs2/librispeech_100/asr1/exp/lm_train_transformer_gpt2_en_hugging_face/gpt2_pretrained.pth'
torch.save(model_state_dict, output_path)

print(f"モデルのパラメータが '{output_path}' に保存されました。")
