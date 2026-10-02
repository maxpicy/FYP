# test_intelligibility_normalizer.py: The two WER text normalisers.

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.intelligibility import (IntelligibilityScorer, word_error_rate,
                                  whisper_text_normalizer)


@pytest.fixture(scope="module")
def whisper_norm():
    try:
        return whisper_text_normalizer("openai/whisper-tiny.en")
    except Exception as e:
        pytest.skip(f"whisper tokenizer not available offline: {e}")


def test_legacy_path_is_default_and_unchanged():
    r = word_error_rate("Hello, world!", "hello world")
    assert r["wer"] == 0.0 and r["normalizer"] == "simple"
    r2 = word_error_rate("pay 5 dollars", "pay dollars")
    assert r2["wer"] == 0.0


def test_whisper_path_expands_titles_and_keeps_numbers(whisper_norm):
    r = word_error_rate("Mr. Smith arrived.", "mister smith arrived", text_normalizer=whisper_norm)
    assert r["wer"] == 0.0 and r["normalizer"] == "whisper"
    r2 = word_error_rate("pay 5 dollars", "pay dollars", text_normalizer=whisper_norm)
    assert r2["wer"] > 0.0


def test_the_two_paths_disagree_where_the_field_would(whisper_norm):
    ref, hyp = "It costs $5.", "it costs five dollars"
    simple = word_error_rate(ref, hyp)["wer"]
    whisper = word_error_rate(ref, hyp, text_normalizer=whisper_norm)["wer"]
    assert simple != whisper


def test_scorer_validates_the_normalizer_argument():
    with pytest.raises(ValueError):
        IntelligibilityScorer(normalizer="jiwer")
    s = IntelligibilityScorer(normalizer="whisper")
    assert s.normalizer == "whisper" and s._text_norm is None
