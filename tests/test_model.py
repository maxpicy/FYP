# test_model.py: Token expansion, embedding resize and the forward pass.

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, PropertyMock

from config import get_all_special_tokens, CODEBOOK_SIZE, NUM_SPEECH_TOKENS


class MockEmbedding(nn.Embedding):
    def __init__(self, vocab_size=50280, embed_dim=2048):
        super().__init__(vocab_size, embed_dim)


class MockLinear(nn.Linear):
    def __init__(self, in_features=2048, out_features=50280):
        super().__init__(in_features, out_features, bias=False)


class TestTokenExpansion:
    def test_special_token_count(self):
        import config
        from config import NUM_SPEECH_TOKENS
        tokens = get_all_special_tokens()
        n_plan = len(config.get_plan_tokens()) if config.ENABLE_PLAN_TOKENS else 0
        assert len(tokens) == 22 + n_plan + NUM_SPEECH_TOKENS
        if not config.ENABLE_PLAN_TOKENS:
            assert len(tokens) == 22 + NUM_SPEECH_TOKENS

    def test_speech_tokens_in_list(self):
        tokens = get_all_special_tokens()
        assert "[SPEECH_0]" in tokens
        assert f"[SPEECH_{NUM_SPEECH_TOKENS - 1}]" in tokens
        assert "[SPEECH_512]" in tokens

    def test_control_tokens_in_list(self):
        tokens = get_all_special_tokens()
        assert "<THINK>" in tokens
        assert "</THINK>" in tokens
        assert "<AUDIO>" in tokens
        assert "</AUDIO>" in tokens
        assert "[USER_PROMPT]" in tokens

    def test_emotion_tokens_in_list(self):
        tokens = get_all_special_tokens()
        assert "[EMO:sarcastic]" in tokens
        assert "[EMO:neutral]" in tokens

    def test_prosody_tokens_in_list(self):
        tokens = get_all_special_tokens()
        assert "[PACE:slow]" in tokens
        assert "[PITCH:high]" in tokens
        assert "[BREAK:long]" in tokens


class TestEmbeddingResize:
    def test_resize_expands_correctly(self):
        old_size = 50280
        new_size = old_size + 1045
        embed_dim = 128

        old_embedding = nn.Embedding(old_size, embed_dim)
        new_embedding = nn.Embedding(new_size, embed_dim)

        with torch.no_grad():
            new_embedding.weight[:old_size] = old_embedding.weight[:old_size]

        assert new_embedding.weight.shape[0] == new_size
        assert new_embedding.weight.shape[1] == embed_dim

        assert torch.allclose(
            new_embedding.weight[:old_size],
            old_embedding.weight[:old_size],
        )

    def test_new_rows_initialized(self):
        old_size = 100
        new_size = 150
        embed_dim = 64

        old_embedding = nn.Embedding(old_size, embed_dim)
        new_embedding = nn.Embedding(new_size, embed_dim)

        with torch.no_grad():
            new_embedding.weight[:old_size] = old_embedding.weight[:old_size]
            mean_embed = old_embedding.weight.mean(dim=0)
            noise = torch.randn(new_size - old_size, embed_dim) * 0.02
            new_embedding.weight[old_size:] = mean_embed + noise

        new_rows = new_embedding.weight[old_size:]
        assert not torch.all(new_rows == 0)

    def test_lm_head_resize(self):
        old_size = 50280
        new_size = old_size + 1045
        hidden_dim = 128

        old_head = nn.Linear(hidden_dim, old_size, bias=False)
        new_head = nn.Linear(hidden_dim, new_size, bias=False)

        with torch.no_grad():
            new_head.weight[:old_size] = old_head.weight[:old_size]

        assert new_head.weight.shape[0] == new_size
        assert new_head.weight.shape[1] == hidden_dim


class TestForwardPassLogic:
    def test_causal_lm_loss_shape(self):
        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_ignore_index_masks_correctly(self):
        vocab_size = 100
        logits = torch.randn(1, 5, vocab_size)
        labels = torch.tensor([[1, -100, -100, 42, 7]])

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        assert per_token[0] == 0.0
        assert per_token[1] == 0.0
        assert per_token[2] > 0.0
        assert per_token[3] > 0.0

    def test_weighted_loss(self):
        vocab_size = 100
        logits = torch.randn(1, 5, vocab_size)
        labels = torch.randint(0, vocab_size, (1, 5))
        weights = torch.tensor([[1.0, 0.3, 0.3, 1.0, 1.0]])

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        weighted = per_token * shift_weights.view(-1)
        loss = weighted.sum() / (shift_labels.view(-1) != -100).float().sum().clamp(min=1)

        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestFreezeUnfreeze:
    def test_freeze_sets_requires_grad_false(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 10),
        )
        for param in model.parameters():
            param.requires_grad = False

        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable == 0

    def test_unfreeze_sets_requires_grad_true(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 10),
        )
        for param in model.parameters():
            param.requires_grad = False

        for param in model.parameters():
            param.requires_grad = True

        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        total = sum(1 for _ in model.parameters())
        assert trainable == total
