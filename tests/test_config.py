# test_config.py: Token definitions and stage configuration.

import pytest
from config import (
    CODEBOOK_SIZE,
    NUM_SPEECH_TOKENS,
    SPEAKER_TOKEN,
    EMOTION_TAGS,
    PROSODY_TAGS,
    STAGE_CONFIGS,
    ModelConfig,
    get_all_special_tokens,
    get_control_tokens,
    get_speech_tokens,
    get_stage_config,
)


class TestTokenDefinitions:
    def test_speech_tokens_count(self):
        tokens = get_speech_tokens()
        assert len(tokens) == CODEBOOK_SIZE

    def test_speech_tokens_format(self):
        tokens = get_speech_tokens()
        assert tokens[0] == "[SPEECH_0]"
        assert tokens[NUM_SPEECH_TOKENS - 1] == f"[SPEECH_{NUM_SPEECH_TOKENS - 1}]"
        assert tokens[512] == "[SPEECH_512]"

    def test_speech_tokens_unique(self):
        tokens = get_speech_tokens()
        assert len(set(tokens)) == len(tokens)

    def test_control_tokens_count(self):
        import config
        tokens = get_control_tokens()
        n_plan = len(config.get_plan_tokens()) if config.ENABLE_PLAN_TOKENS else 0
        assert len(tokens) == 6 + len(EMOTION_TAGS) + len(PROSODY_TAGS) + n_plan
        if not config.ENABLE_PLAN_TOKENS:
            assert len(tokens) == 22

    def test_all_special_tokens_count(self):
        tokens = get_all_special_tokens()
        expected = len(get_control_tokens()) + CODEBOOK_SIZE
        assert len(tokens) == expected

    def test_all_tokens_unique(self):
        tokens = get_all_special_tokens()
        assert len(set(tokens)) == len(tokens)

    def test_emotion_tags_format(self):
        for tag in EMOTION_TAGS:
            assert tag.startswith("[EMO:")
            assert tag.endswith("]")

    def test_prosody_tags_format(self):
        for tag in PROSODY_TAGS:
            assert tag.startswith("[") and tag.endswith("]")
            prefix = tag[1:].split(":")[0]
            assert prefix in ("PACE", "PITCH", "BREAK")

    def test_num_speech_tokens_exists(self):
        assert NUM_SPEECH_TOKENS > 0

    def test_codebook_size_is_alias(self):
        assert CODEBOOK_SIZE == NUM_SPEECH_TOKENS

    def test_speaker_token_in_all_special(self):
        tokens = get_all_special_tokens()
        assert SPEAKER_TOKEN in tokens


class TestModelConfig:
    def test_model_config_crh_defaults(self):
        cfg = ModelConfig()
        assert cfg.num_crh_heads == 0
        assert cfg.crh_discount == 0.85
        assert cfg.crh_codebook_size == 2048

    def test_model_config_speaker_defaults(self):
        cfg = ModelConfig()
        assert cfg.use_speaker_conditioning is False
        assert cfg.speaker_embedding_dim == 256
        assert cfg.num_speakers == 1000

    def test_model_config_spec_defaults(self):
        cfg = ModelConfig()
        assert cfg.num_spec_heads == 0
        assert cfg.spec_discount == 0.8


class TestStageConfigs:
    def test_all_stages_exist(self):
        for stage in [1, 2, 3, 4]:
            config = get_stage_config(stage)
            assert config is not None
            assert config.name != ""

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError):
            get_stage_config(0)
        with pytest.raises(ValueError):
            get_stage_config(6)

    def test_stage1_uses_lora(self):
        config = get_stage_config(1)
        assert config.use_lora is True
        assert config.lora_rank > 0

    def test_stage1_masks_reasoning(self):
        config = get_stage_config(1)
        assert config.mask_reasoning is True
        assert config.include_reasoning_in_sequence is False

    def test_stage2_includes_reasoning(self):
        config = get_stage_config(2)
        assert config.include_reasoning_in_sequence is True
        assert config.mask_reasoning is False
        assert config.lambda_reasoning > 0.0

    def test_stage2_full_finetune(self):
        config = get_stage_config(2)
        assert config.use_lora is False

    def test_learning_rates_positive(self):
        for stage_num, config in STAGE_CONFIGS.items():
            assert config.backbone_lr >= 0, f"Stage {stage_num} backbone_lr"
            assert config.new_params_lr > 0, f"Stage {stage_num} new_params_lr"

    def test_loss_weights_valid(self):
        for stage_num, config in STAGE_CONFIGS.items():
            assert config.lambda_reasoning >= 0.0, f"Stage {stage_num} lambda_r"
            assert config.lambda_speech >= 0.0, f"Stage {stage_num} lambda_s"

    def test_stage_lr_decreases(self):
        s1 = get_stage_config(1)
        s2 = get_stage_config(2)
        s3 = get_stage_config(3)
        assert s2.backbone_lr <= s1.backbone_lr
        assert s3.backbone_lr <= s2.backbone_lr
