#!/usr/bin/env python3
#  2022, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers Decoder."""

import copy
import logging
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

try:
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
    from transformers.file_utils import ModelOutput

    is_transformers_available = True
except ImportError:
    is_transformers_available = False

from espnet.nets.scorer_interface import BatchScorerInterface


class HuggingFaceTransformersDecoder(AbsDecoder, BatchScorerInterface):
    """Hugging Face Transformers Decoder.

    Args:
        encoder_output_size: dimension of encoder attention
        model_name_or_path: Hugging Face Transformers model name
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        model_name_or_path: str,
        causal_lm: bool = False,
        prefix: str = "",
        postfix: str = "",
    ):
        super().__init__()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        self.causal_lm = causal_lm

        if self.causal_lm:
            config = AutoConfig.from_pretrained(model_name_or_path)
            config.add_cross_attention = True

            # 設定を使用してモデルをロード
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
            object.__setattr__(self, 'hf_generate', model)
            self.decoder = get_hugging_face_model_network(model)

            if hasattr(self.decoder, "word_embeddings"):
                self.decoder_word_embeddings = self.decoder.word_embeddings
            elif hasattr(self.decoder, "embed_in"):
                self.decoder_word_embeddings = self.decoder.embed_in
            elif hasattr(self.decoder, "embed_tokens"):
                self.decoder_word_embeddings = self.decoder.embed_tokens
            elif hasattr(self.decoder, "wte"):
                self.decoder_word_embeddings = self.decoder.wte
            else:
                raise Exception("Can not find the word embeddings attribute")

            if (
                self.decoder.config.pad_token_id is not None
                and self.decoder.config.pad_token_id != -1
            ):
                self.decoder_pad_token_id = self.decoder.config.pad_token_id
            else:
                self.decoder_pad_token_id = 1

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.tokenizer_padding_side = tokenizer.padding_side

            self.prefix = self.decoder_word_embeddings(
                tokenizer.encode(prefix, return_tensors="pt").long()
            ).detach()

            self.postfix = self.decoder_word_embeddings(
                tokenizer.encode(postfix, return_tensors="pt").long()
            ).detach()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

            if hasattr(model, "model"):
                self.decoder = model.model.decoder
            else:
                self.decoder = model.decoder

        model.resize_token_embeddings(vocab_size)

        self.lm_head = get_hugging_face_model_lm_head(model)

        self.model_name_or_path = model_name_or_path

        self.decoder_pretrained_params = copy.deepcopy(self.decoder.state_dict())
        self.lm_head_pretrained_params = copy.deepcopy(self.lm_head.state_dict())

        if encoder_output_size != self.decoder.config.hidden_size:
            self.linear_in = torch.nn.Linear(
                encoder_output_size, self.decoder.config.hidden_size
            )
        else:
            self.linear_in = torch.nn.Identity()

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad: input tensor (batch, maxlen_out, #mels)
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        enc_out = self.linear_in(hs_pad)

        if self.causal_lm:
            args, no_loss_lengths = self.add_prefix_postfix(
                enc_out, hlens, ys_in_pad, ys_in_lens
            )
        else:
            args = {"return_dict": True}

            if self.decoder.__class__.__name__ == "MBartDecoder":
                ys_in_pad[:, 0] = 2

            args["input_ids"] = ys_in_pad
            mask = (~make_pad_mask(ys_in_lens)).to(ys_in_pad.device).float()
            args["attention_mask"] = mask

            args["encoder_hidden_states"] = enc_out
            hs_mask = (~make_pad_mask(hlens)).to(hs_pad.device).float()
            args["encoder_attention_mask"] = hs_mask

        x = self.decoder(**args).last_hidden_state

        if self.causal_lm:
            if self.tokenizer_padding_side == "left":
                x = torch.vstack(
                    [
                        F.pad(
                            x[i, -ys_in_lens[i] :, :],
                            (0, 0, 0, ys_in_lens.max() - ys_in_lens[i]),
                        ).unsqueeze(0)
                        for i in range(x.shape[0])
                    ]
                )
            else:
                x = torch.vstack(
                    [
                        F.pad(
                            x[
                                i,
                                no_loss_lengths[i] : no_loss_lengths[i] + ys_in_lens[i],
                                :,
                            ],
                            (0, 0, 0, ys_in_lens.max() - ys_in_lens[i]),
                        ).unsqueeze(0)
                        for i in range(x.shape[0])
                    ]
                )

        x = self.lm_head(x)

        return x, ys_in_lens

    def reload_pretrained_parameters(self):
        self.decoder.load_state_dict(self.decoder_pretrained_params)

        if self.lm_head_pretrained_params is not None:
            self.lm_head.load_state_dict(self.lm_head_pretrained_params)

        logging.info("Pretrained Transformers model parameters reloaded!")

    def add_prefix_postfix(self, enc_out, hlens, ys_in_pad, ys_in_lens):
        args = {}

        hlens_max = (hlens + ys_in_lens).max()

        enc_out_list = []

        for i in range(len(hlens)):
            enc_out_element = [
                self.prefix.to(enc_out.device),
                enc_out[i : i + 1, : hlens[i], :],
                self.postfix.to(enc_out.device),
                self.decoder_word_embeddings(
                    ys_in_pad[i : i + 1, 1 : ys_in_lens[i]]
                ).to(enc_out.device),
            ]

            padding = self.decoder_word_embeddings(
                torch.tensor([[self.decoder_pad_token_id]]).to(enc_out.device)
            ).expand(-1, hlens_max - (hlens[i] + ys_in_lens[i]), -1)

            if self.tokenizer_padding_side == "left":
                enc_out_element.insert(0, padding)
            else:
                enc_out_element.insert(len(enc_out_element), padding)

            enc_out_list.append(torch.cat(enc_out_element, dim=1))

        args["inputs_embeds"] = torch.vstack(enc_out_list)

        no_loss_lengths = self.prefix.size(1) + hlens + self.postfix.size(1) - 1
        inputs_lengths = no_loss_lengths + ys_in_lens

        hs_mask = (~make_pad_mask(inputs_lengths)).to(enc_out.device).float()

        if self.tokenizer_padding_side == "left":
            args["attention_mask"] = hs_mask.flip([1])
        else:
            args["attention_mask"] = hs_mask

        # ここで encoder_hidden_states と encoder_attention_mask を追加
        args["encoder_hidden_states"] = enc_out
        hs_mask = (~make_pad_mask(hlens)).to(enc_out.device).float()
        args["encoder_attention_mask"] = hs_mask

        args["return_dict"] = True

        return args, no_loss_lengths

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor = None,
        *,
        cache: List[torch.Tensor] = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # import pdb;pdb.set_trace()
        memory = self.linear_in(memory)
        # トークンをエンベディング. ここの model_inputs作る処理がよくわからない.
        if self.causal_lm:
            # causal_lm モデルの場合 (今回はこちら？)
            # import pdb;pdb.set_trace()
            inputs_embeds = self.decoder_word_embeddings(tgt[:, -1:])  # (batch, 1, hidden_size)
            model_inputs = {
                "inputs_embeds": inputs_embeds,
                "encoder_hidden_states": memory,
                "encoder_attention_mask": memory_mask,
                "past_key_values": cache,
                "return_dict": True,
                "use_cache": True
            }
        else:
            # encoder-decoder モデルの場合
            model_inputs = {
                "input_ids": tgt,
                "encoder_hidden_states": memory,
                "encoder_attention_mask": memory_mask,
                "past_key_values": cache,
                "return_dict": True,
            }

        # デコーダを前向き計算
        outputs = self.decoder(**model_inputs)

        # 新しいキャッシュを取得
        new_cache = outputs.past_key_values

        # 最後のトークンの隠れ状態を取得
        last_hidden_state = outputs.last_hidden_state  # (batch, 1, hidden_size)

        # LM Head を通してロジットを取得
        logits = self.lm_head(last_hidden_state)  # (batch, 1, vocab_size)

         # ログソフトマックスを適用してスコアを取得
        next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size) ビームサーチ内での演算時にテンソルの形状不一致が発生していたため, シーケンスの最後のタイムステップのみを使用させている.
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        # import pdb;pdb.set_trace()
        return log_probs, new_cache

    def score(self, ys, state, x, speech=None):
        # model_kwargs = {
        #     "encoder_outputs": ModelOutput(
        #         last_hidden_state=self.linear_in(x).unsqueeze(0)
        #     ),
        # }
        # TODO(brian): caching
        # import pdb;pdb.set_trace()
        # model_inputs = self.hf_generate.prepare_inputs_for_generation(
        #     ys.unsqueeze(0), **model_kwargs
        # )
        # outputs = self.hf_generate(
        #     **model_inputs,
        #     return_dict=True,
        #     output_attentions=False,
        #     output_hidden_states=False
        # )
        # next_token_logits = outputs.logits[:, -1, :]
        # next_token_scores = torch.nn.functional.log_softmax(
        #     next_token_logits, dim=-1
        # )  # (batch_size * num_beams, vocab_size)
        # return next_token_scores.squeeze(0), None
    
        # 状態がない場合はNoneに設定
        if state is None:
            decoder_state = None
        else:
            decoder_state = state
        
        # サブシーケントマスクを作成
        from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
        ys_mask = subsequent_mask(ys.size(0), device=ys.device).unsqueeze(0)  # (1, ylen, ylen)

        # `forward_one_step`を呼び出し
        log_probs, new_cache = self.forward_one_step(
            ys.unsqueeze(0),
            ys_mask,
            x.unsqueeze(0),
            memory_mask=None,
            cache=decoder_state,
        )

        # バッチ次元を削除
        log_probs = log_probs.squeeze(0)

        return log_probs, new_cache

    def batch_score(
        self,
        ys: torch.Tensor, # 今回はtensor([[50256]], device='cuda:0')
        states: List[Any], # 今回は[None]
        xs: torch.Tensor, # 今回のxs.shapeはtorch.Size([1, 260, 256])
        speech: torch.Tensor = None, # 今回はNone
    ) -> Tuple[torch.Tensor, List[Any]]:
        # import pdb;pdb.set_trace() # 現在はGPUを使っているので, ここで止まる.
        # model_kwargs = {
        #     "encoder_outputs": ModelOutput(last_hidden_state=self.linear_in(xs)),
        # }
        # model_inputs = self.hf_generate.prepare_inputs_for_generation(
        #     ys, **model_kwargs
        # )
        # outputs = self.hf_generate(
        #     **model_inputs,
        #     return_dict=True,
        #     output_attentions=False,
        #     output_hidden_states=False
        # )
        # next_token_logits = outputs.logits[:, -1, :]
        # next_token_scores = torch.nn.functional.log_softmax(
        #     next_token_logits, dim=-1
        # )  # (batch_size * num_beams, vocab_size)
        # # import pdb;pdb.set_trace()
        # return next_token_scores, None

        # バッチサイズの取得
        n_batch = len(ys)

        # デコーダの状態を設定
        if states is None or len(states) == 0 or states[0] is None:
            batch_state = None
        else:
            # states: [batch, layer] の状態を [layer, batch] に変換
            num_layers = len(states[0])
            batch_state = []
            for layer_idx in range(num_layers):
                # 各層について、バッチ全体のキーとバリューを収集
                layer_past = []
                for b in range(n_batch):
                    layer_past.append(states[b][layer_idx])
                # 各テンソルをバッチ次元でスタック
                # layer_past は [(key1, value1), (key2, value2), ..., (keyN, valueN)]
                # これを (keys, values) の形に変換
                stacked_layer_past = tuple(
                    torch.stack([layer_past[b][i] for b in range(n_batch)], dim=0)
                    for i in range(len(layer_past[0]))
                )
                batch_state.append(stacked_layer_past)
            batch_state = tuple(batch_state)  # リストをタプルに変換

        # 将来のトークン情報をマスク
        from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
        ys_mask = subsequent_mask(ys.size(1), device=ys.device).unsqueeze(0)  # (1, ylen, ylen)

        # エンコーダのマスクを設定（今回は不要？）
        memory_mask = None  

        # forward_one_step を呼び出し
        log_probs, new_cache = self.forward_one_step(
            ys,
            ys_mask,
            xs,
            memory_mask=memory_mask,
            cache=batch_state,
        )

        # 状態を [layer, batch] から [batch, layer] に戻す
        state_list = []
        num_layers = len(new_cache)
        for b in range(n_batch):
            state_b = []
            for layer_idx in range(num_layers):
                # 各層のバッチ内の状態を取得
                layer_past = tuple(past_state[b] for past_state in new_cache[layer_idx])
                state_b.append(layer_past)
            state_list.append(state_b)
        # import pdb;pdb.set_trace()
        # 元コードでは, 返り値の二つ目がNoneなので, state_listが必要かわからない.
        return log_probs, state_list

def get_hugging_face_model_network(model):
    if hasattr(model, "transformer"):
        network = model.transformer
    elif hasattr(model, "gpt_neox"):
        network = model.gpt_neox
    elif hasattr(model, "model"):
        network = model.model
    else:
        raise Exception("Can not find the network attribute")

    return network


def get_hugging_face_model_lm_head(model):
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
    elif hasattr(model, "embed_out"):
        lm_head = model.embed_out
    else:
        raise Exception("Can not find the LM head attribute")

    return lm_head
