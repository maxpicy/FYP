# conftest.py: puts the repository root on sys.path and provides shared test fixtures.

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def model():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from model import MambaCoTModel
    m = MambaCoTModel(
        model_name="state-spaces/mamba2-1.3b",
        device="cuda",
        dtype=torch.bfloat16,
        mtp_num_heads=3,
        mtp_gamma=0.8,
    )
    return m


@pytest.fixture(scope="session")
def tokenizer():
    from tokenizer import load_base_tokenizer, expand_tokenizer
    tok = load_base_tokenizer()
    expand_tokenizer(tok)
    return tok


@pytest.fixture(scope="session")
def registry(tokenizer):
    from tokenizer import TokenRegistry
    return TokenRegistry(tokenizer)


@pytest.fixture(scope="session")
def jsonl_path():
    samples = [
        {
            "prompt": "Oh great, another Monday.",
            "reasoning": "<THINK> The speaker uses sarcasm. [EMO:sarcastic] with [PACE:slow] emphasis on great, [PITCH:low] falling contour. </THINK>",
            "audio_file": "test_001.wav",
            "emotion_label": "sarcastic",
            "emotion_confidence": 0.87,
            "speech_tokens": [42, 917, 203, 81, 556, 312, 44, 901],
        },
        {
            "prompt": "I am so happy today!",
            "reasoning": "<THINK> Genuine joy expressed. [EMO:joyful] with [PACE:fast] upbeat delivery, [PITCH:high] rising contour. </THINK>",
            "audio_file": "test_002.wav",
            "emotion_label": "joyful",
            "emotion_confidence": 0.92,
            "speech_tokens": [100, 200, 300, 400, 500, 600, 700, 800],
        },
        {
            "prompt": "The weather is nice.",
            "reasoning": "<THINK> Neutral statement. [EMO:neutral] with [PACE:normal] delivery, [PITCH:mid] flat contour. </THINK>",
            "audio_file": "test_003.wav",
            "emotion_label": "neutral",
            "emotion_confidence": 0.95,
            "speech_tokens": [10, 20, 30, 40, 50],
        },
    ]

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    yield path

    os.unlink(path)
