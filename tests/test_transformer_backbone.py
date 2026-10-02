# test_transformer_backbone.py: The transformer backbone's interface.

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SMALL = "EleutherAI/pythia-70m"


@pytest.fixture(scope="module")
def bb():
    pytest.importorskip("transformers")
    from backbones import load_backbone
    try:
        return load_backbone("transformer", SMALL, device="cpu",
                             dtype=torch.float32)
    except Exception as e:
        pytest.skip(f"cannot load {SMALL}: {type(e).__name__}")


def test_hidden_states_shape(bb):
    ids = torch.randint(0, 1000, (2, 16))
    h = bb.backbone(ids)
    assert h.shape[:2] == (2, 16)
    assert h.shape[2] == bb.config.d_model


def test_config_exposes_d_model(bb):
    assert bb.config.d_model == bb.config.hidden_size


def test_lm_head_maps_hidden_to_vocab(bb):
    h = bb.backbone(torch.randint(0, 1000, (1, 8)))
    assert bb.lm_head(h).shape[-1] == bb.config.vocab_size


def test_layers_is_indexable_for_hybrid_graft(bb):
    layers = bb.backbone.layers
    assert len(layers) == bb.config.num_hidden_layers
    assert isinstance(layers[0], nn.Module)


def test_embedding_get(bb):
    emb = bb.backbone.embedding
    assert isinstance(emb, nn.Module)
    assert emb.weight.shape[1] == bb.config.d_model


def test_embedding_set_is_actually_used_in_forward(bb):
    original = bb.backbone.embedding
    d = bb.config.d_model

    class Marker(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.calls = 0

        @property
        def weight(self):
            return self.base.weight

        def forward(self, ids):
            self.calls += 1
            return torch.zeros(*ids.shape, d)

    marker = Marker(original)
    try:
        bb.backbone.embedding = marker
        assert bb.backbone.embedding is marker, "getter must see the swap"
        out = bb.backbone(torch.randint(0, 1000, (1, 8)))
        assert marker.calls == 1, "the swapped embedding was NOT used in forward"
        assert out.shape == (1, 8, d)
    finally:
        bb.backbone.embedding = original


def test_params_registered_exactly_once(bb):
    ids = [id(p) for p in bb.parameters()]
    assert len(ids) == len(set(ids))


def test_context_limit_is_reported(bb):
    assert bb.max_position_embeddings is not None
    assert bb.max_position_embeddings >= 512

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
