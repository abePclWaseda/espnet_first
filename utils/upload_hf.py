from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ログインしてない場合は以下を実行
from huggingface_hub import login
# login(token="your-hf-token")  # トークンは https://huggingface.co/settings/tokens で取得

# 事前学習済みモデルを読み込み
model_name = "/mnt/kiso-qnap/abe/b4/espnet/egs2/librispeech_100/asr1/exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/valid.acc.ave_10best.pth" 
hf_repo_name = "abePclWaseda/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondet_raw_en_hf_openai-gpt2_sp"

# 1. モデルとトークナイザを読み込み
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Hugging Face Hub にアップロード
print("Pushing model and tokenizer to Hugging Face Hub...")
model.push_to_hub(hf_repo_name)
tokenizer.push_to_hub(hf_repo_name)

print(f"Model successfully uploaded to https://huggingface.co/{hf_repo_name}")
