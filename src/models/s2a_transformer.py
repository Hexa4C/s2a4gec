# This code defines a gec model that use a seq2edit part to constraint its generation procedure
# author: Jiquan Li
# email: lijiquan@mail.ustc.edu.cn

from typing import Any, Dict, List, Optional, Tuple
from fairseq import utils
from fairseq.models.fairseq_encoder import EncoderOut
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    FairseqDropout
)
from fairseq.models.transformer import Linear, TransformerModel, TransformerDecoder
from fairseq.models.transformer import base_architecture as transformer_base_arch


@register_model("s2a_transformer")
class S2ATransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return S2ADecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        succ_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            succ_tokens=succ_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out
        
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
        with_orig_lprobs: bool = False,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample, with_orig_lprobs)
        
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
        with_orig_lprobs: bool = False,
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample, with_orig_lprobs)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError


class S2ADecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        self.s2a_dim = self.output_embed_dim * 2
        self.act = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'gelu')
        )
        self.encfc1 = Linear(self.s2a_dim, self.s2a_dim, bias=True)
        self.encfc2 = Linear(self.s2a_dim, 3, bias=True)
        # self.s2a_linear = nn.Linear(
        #     self.s2a_dim, 3, bias=False
        # )
        # nn.init.normal_(
        #     self.s2a_linear.weight, mean=0, std=self.s2a_dim ** -0.5
        # )
        self.s2a_softmax = nn.Softmax(dim=2)

    def forward(
        self,
        prev_output_tokens,
        succ_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            succ_tokens=succ_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        succ_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            succ_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
    
    def extract_features_scriptable(
        self,
        prev_output_tokens,
        succ_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            succ_tokens = succ_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        succ_embed = self.embed_tokens(succ_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        s2a_features = torch.cat((x, succ_embed), dim=-1)
        # s2a_out = self.s2a_linear(s2a_features)
        s2a_out = self.encfc2(self.act(self.encfc1(s2a_features)))

        return x, {"attn": [attn], "inner_states": inner_states, "s2a_out": s2a_out, "succ_tokens": succ_tokens}

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
        with_orig_lprobs: bool = False,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        bsz, length, v_size = logits.size()
        s2a_out = net_output[1]['s2a_out']
        s2a_probs = self.s2a_softmax(s2a_out.float())
        succ_tokens = net_output[1]['succ_tokens']
        token_probs = self.s2a_softmax(logits.float())
        # x = torch.arange(bsz * length)
        # gen_token_probs.view(bsz * length, -1)[x, succ_tokens.view(-1)] = 0.

        gen_token_probs = token_probs.scatter(2, succ_tokens.unsqueeze(-1), 0.)
        gen_token_probs[:, :, -1] = 0.
        rest_sum = gen_token_probs.sum(dim=2)
        gen_token_probs = gen_token_probs / rest_sum.unsqueeze(-1)

        gen_token_probs = gen_token_probs * s2a_probs[:, :, 2].unsqueeze(-1)
        skip_probs = torch.zeros_like(gen_token_probs)
        skip_probs[:, :, -1] = s2a_probs[:, :, 0]
        copy_probs = torch.zeros_like(gen_token_probs).scatter_(2, succ_tokens.unsqueeze(-1), s2a_probs[:, :, 1].unsqueeze(-1))
        total_probs = gen_token_probs + skip_probs + copy_probs
        if with_orig_lprobs:
            if log_probs:
                return torch.log(total_probs + 1e-10).type_as(logits), utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
            else:
                return total_probs.type_as(logits), utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            if log_probs:
                return torch.log(total_probs + 1e-10).type_as(logits)
            else:
                return total_probs.type_as(logits)
        # if log_probs:
        #     return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        # else:
        #     return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)


@register_model_architecture('s2a_transformer', 's2a_transformer')
def base_architecture(args):
    transformer_base_arch(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("s2a_transformer", "s2a_transformer_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)
