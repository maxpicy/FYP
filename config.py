# config.py: Token vocabulary and constants: control, tag, plan and speech tokens, and the codec tier sizes read
# from MVC_NUM_SPEECH_TOKENS / MVC_NUM_CODEC_LEVELS / MVC_ENABLE_PLAN_TOKENS.

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"

USER_PROMPT_TOKEN = "[USER_PROMPT]"

THINK_START_TOKEN = "<THINK>"
THINK_END_TOKEN = "</THINK>"

AUDIO_START_TOKEN = "<AUDIO>"
AUDIO_END_TOKEN = "</AUDIO>"

SPEAKER_TOKEN = "[SPEAKER]"

EMOTION_TAGS = [
    "[EMO:neutral]",
    "[EMO:sarcastic]",
    "[EMO:angry]",
    "[EMO:sad]",
    "[EMO:joyful]",
    "[EMO:fearful]",
    "[EMO:ironic]",
    "[EMO:deadpan]",
]

PROSODY_TAGS = [
    "[PACE:slow]",
    "[PACE:normal]",
    "[PACE:fast]",
    "[PITCH:low]",
    "[PITCH:mid]",
    "[PITCH:high]",
    "[BREAK:short]",
    "[BREAK:long]",
]

_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off"}


def _env_flag(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name, default)
    value = str(raw).strip().lower()
    if value in _TRUE_STRINGS:
        return True
    if value in _FALSE_STRINGS:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a recognised boolean. Use one of "
        f"{sorted(_TRUE_STRINGS)} or {sorted(_FALSE_STRINGS)}."
    )

ENABLE_PLAN_TOKENS: bool = _env_flag("MVC_ENABLE_PLAN_TOKENS", "0")

PLAN_START_TOKEN = "<PLAN>"
PLAN_END_TOKEN = "</PLAN>"

PLAN_WORD_SEP_TOKEN = "[PW]"

PITCH_BINS = 16
DURATION_BINS = 16
ENERGY_BINS = 8

PLAN_PITCH_PREFIX = "[P:"
PLAN_DURATION_PREFIX = "[D:"
PLAN_ENERGY_PREFIX = "[E:"
STYLE_TOKEN_PREFIX = "[STY:"
PLAN_TOKEN_SUFFIX = "]"

NUM_STYLE_TOKENS: int = int(os.environ.get("MVC_NUM_STYLE_TOKENS", "0"))
if NUM_STYLE_TOKENS < 0 or NUM_STYLE_TOKENS > 4096:
    raise ValueError(
        f"MVC_NUM_STYLE_TOKENS={NUM_STYLE_TOKENS} out of range [0, 4096] — this is a "
        f"small pooled codebook (1-4 tokens per utterance, §3.3), not a codec."
    )
if NUM_STYLE_TOKENS > 0 and not ENABLE_PLAN_TOKENS:
    raise ValueError(
        f"MVC_NUM_STYLE_TOKENS={NUM_STYLE_TOKENS} requires MVC_ENABLE_PLAN_TOKENS=1 "
        f"(the style tokens are part of the P²-CoT plan block). Set both or neither."
    )


def get_plan_pitch_tokens() -> List[str]:
    return [f"{PLAN_PITCH_PREFIX}{i}{PLAN_TOKEN_SUFFIX}" for i in range(PITCH_BINS)]


def get_plan_duration_tokens() -> List[str]:
    return [f"{PLAN_DURATION_PREFIX}{i}{PLAN_TOKEN_SUFFIX}" for i in range(DURATION_BINS)]


def get_plan_energy_tokens() -> List[str]:
    return [f"{PLAN_ENERGY_PREFIX}{i}{PLAN_TOKEN_SUFFIX}" for i in range(ENERGY_BINS)]


def get_style_tokens() -> List[str]:
    return [f"{STYLE_TOKEN_PREFIX}{i}{PLAN_TOKEN_SUFFIX}" for i in range(NUM_STYLE_TOKENS)]


def get_plan_tokens() -> List[str]:
    return (
        [PLAN_START_TOKEN, PLAN_END_TOKEN, PLAN_WORD_SEP_TOKEN]
        + get_plan_pitch_tokens()
        + get_plan_duration_tokens()
        + get_plan_energy_tokens()
        + get_style_tokens()
    )

NUM_SPEECH_TOKENS = int(os.environ.get("MVC_NUM_SPEECH_TOKENS", "65536"))
CODEBOOK_SIZE = NUM_SPEECH_TOKENS
SPEECH_TOKEN_PREFIX = "[SPEECH_"
SPEECH_TOKEN_SUFFIX = "]"


def get_speech_tokens() -> List[str]:
    return [f"{SPEECH_TOKEN_PREFIX}{i}{SPEECH_TOKEN_SUFFIX}" for i in range(CODEBOOK_SIZE)]


def get_control_tokens() -> List[str]:
    tokens = [
        USER_PROMPT_TOKEN,
        THINK_START_TOKEN,
        THINK_END_TOKEN,
        AUDIO_START_TOKEN,
        AUDIO_END_TOKEN,
        SPEAKER_TOKEN,
    ] + EMOTION_TAGS + PROSODY_TAGS
    if ENABLE_PLAN_TOKENS:
        tokens = tokens + get_plan_tokens()
    return tokens


def get_all_special_tokens() -> List[str]:
    return get_control_tokens() + get_speech_tokens()


@dataclass
class TrainingStageConfig:
    name: str
    description: str

    backbone_lr: float
    new_params_lr: float

    freeze_backbone: bool = False
    use_lora: bool = False
    lora_rank: int = 64
    lora_target_modules: List[str] = field(default_factory=lambda: ["in_proj", "out_proj"])

    lambda_reasoning: float = 0.0
    lambda_speech: float = 1.0

    mask_prompt: bool = True
    mask_reasoning: bool = True
    include_reasoning_in_sequence: bool = False

    warmup_steps: int = 1000
    scheduler_type: str = "cosine"

    max_seq_length: int = 2048
    gradient_checkpointing: bool = False

    freeze_backbone_for_mtp: bool = False
    mtp_healing: bool = False

STAGE_1_CONFIG = TrainingStageConfig(
    name="codec_alignment",
    description="Stage 1: Teach backbone to predict speech tokens from text (no CoT).",
    backbone_lr=1e-4,
    new_params_lr=3e-4,
    freeze_backbone=False,
    use_lora=True,
    lora_rank=64,
    lambda_reasoning=0.0,
    lambda_speech=1.0,
    mask_prompt=True,
    mask_reasoning=True,
    include_reasoning_in_sequence=False,
    warmup_steps=2000,
    max_seq_length=2048,
)

STAGE_2_CONFIG = TrainingStageConfig(
    name="cot_pretraining",
    description="Stage 2: Teach model to generate reasoning before speech tokens.",
    backbone_lr=5e-5,
    new_params_lr=5e-5,
    freeze_backbone=False,
    use_lora=False,
    lambda_reasoning=0.3,
    lambda_speech=1.0,
    mask_prompt=True,
    mask_reasoning=False,
    include_reasoning_in_sequence=True,
    warmup_steps=3000,
    max_seq_length=4096,
)

STAGE_3_CONFIG = TrainingStageConfig(
    name="expressive_sft",
    description="Stage 3: Fine-tune on expressive/emotional data with CoT.",
    backbone_lr=1e-5,
    new_params_lr=1e-5,
    freeze_backbone=False,
    use_lora=False,
    lambda_reasoning=0.3,
    lambda_speech=1.0,
    mask_prompt=True,
    mask_reasoning=False,
    include_reasoning_in_sequence=True,
    warmup_steps=500,
    max_seq_length=4096,
)

STAGE_4_CONFIG = TrainingStageConfig(
    name="dpo",
    description="Stage 4: Direct Preference Optimization for prosody alignment.",
    backbone_lr=5e-6,
    new_params_lr=5e-6,
    freeze_backbone=False,
    use_lora=False,
    lambda_reasoning=0.0,
    lambda_speech=1.0,
    mask_prompt=True,
    mask_reasoning=True,
    include_reasoning_in_sequence=True,
    warmup_steps=200,
    max_seq_length=4096,
)

STAGE_5_CONFIG = TrainingStageConfig(
    name="mtp_healing",
    description=(
        "Stage 5: MTP Healing [11] — freeze backbone and fine-tune only "
        "CRH + speculative prediction heads on self-generated data using NLL loss. "
        "Restores MTP accuracy after DPO/RL stages which may degrade speculative "
        "decoding acceptance rates."
    ),
    backbone_lr=0.0,
    new_params_lr=1e-4,
    freeze_backbone=False,
    use_lora=False,
    lambda_reasoning=0.0,
    lambda_speech=1.0,
    mask_prompt=True,
    mask_reasoning=True,
    include_reasoning_in_sequence=True,
    warmup_steps=100,
    max_seq_length=4096,
    freeze_backbone_for_mtp=True,
    mtp_healing=True,
)

STAGE_CONFIGS = {
    1: STAGE_1_CONFIG,
    2: STAGE_2_CONFIG,
    3: STAGE_3_CONFIG,
    4: STAGE_4_CONFIG,
    5: STAGE_5_CONFIG,
}


def get_stage_config(stage: int) -> TrainingStageConfig:
    if stage not in STAGE_CONFIGS:
        raise ValueError(f"Invalid stage {stage}. Must be one of {list(STAGE_CONFIGS.keys())}")
    return STAGE_CONFIGS[stage]


@dataclass
class ModelConfig:
    model_name: str = "state-spaces/mamba2-1.3b"
    codebook_size: int = NUM_SPEECH_TOKENS
    num_crh_heads: int = 0
    crh_discount: float = 0.85
    crh_codebook_size: int = 2048
    num_spec_heads: int = 0
    spec_discount: float = 0.8
    shared_spec_weights: bool = True
    use_speaker_conditioning: bool = False
    speaker_embedding_dim: int = 256
    num_speakers: int = 1000
    num_mtp_heads: int = 0
    mtp_discount: float = 0.8
    dtype: str = "bfloat16"

DEFAULT_MODEL_CONFIG = ModelConfig()
