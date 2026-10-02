# backbones.py: Puts the transformer (Pythia-1.4B) behind the same interface as Mamba-2, including cached stepping.

from __future__ import annotations

import torch
import torch.nn as nn


class _NeoXInner(nn.Module):
    def __init__(self, neox):
        super().__init__()
        self.neox = neox

    def __getattr__(self, name):
        if name == "embedding":
            return self.neox.embed_in
        if name == "layers":
            return self.neox.layers
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name == "embedding":
            self._modules["neox"].embed_in = value
            return
        super().__setattr__(name, value)

    def forward(self, input_ids: torch.Tensor, inference_params=None, **kw) -> torch.Tensor:
        if inference_params is None:
            return self.neox(input_ids=input_ids).last_hidden_state
        past = getattr(inference_params, "hf_past", None)
        if int(inference_params.seqlen_offset) == 0:
            past = None
        out = self.neox(input_ids=input_ids, past_key_values=past, use_cache=True)
        inference_params.hf_past = out.past_key_values
        return out.last_hidden_state


class TransformerBackbone(nn.Module):
    def __init__(self, model_name: str, device=None, dtype=None):
        super().__init__()
        from transformers import GPTNeoXForCausalLM
        m = GPTNeoXForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        if device is not None:
            m = m.to(device)
        cfg = m.config
        cfg.d_model = cfg.hidden_size
        self.config = cfg
        self.backbone = _NeoXInner(m.gpt_neox)
        self.lm_head = m.embed_out
        self.max_position_embeddings = getattr(cfg, "max_position_embeddings", None)

    def forward(self, input_ids: torch.Tensor, **kw):
        return self.lm_head(self.backbone(input_ids))


def load_backbone(kind: str, model_name: str, device=None, dtype=None):
    if kind == "mamba":
        from mamba_ssm import MambaLMHeadModel
        return MambaLMHeadModel.from_pretrained(model_name, device=device,
                                                dtype=dtype)
    if kind == "transformer":
        return TransformerBackbone(model_name, device=device, dtype=dtype)
    raise ValueError(f"unknown backbone kind {kind!r} (mamba|transformer)")

DEFAULT_MODELS = {"mamba": "state-spaces/mamba2-1.3b",
                  "transformer": "EleutherAI/pythia-1.4b"}
