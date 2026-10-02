# test_plan_extract.py: The prosody plan extractor and quantiser.

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from plan_extract import (
    DUR_EDGES_FRAMES,
    FRAME_RATE_HZ,
    MIN_WORDS_FOR_SPEAKER_BINS,
    N_ENERGY_BINS,
    N_PITCH_BINS,
    PITCH_UNVOICED_BIN,
    PLAN_WORD_TOKEN,
    PlanAlignmentError,
    PlanExtractError,
    PlanExtractor,
    PlanQuantiseError,
    PlanTokenError,
    dequantise_plan,
    extract_word_plan,
    fit_speaker_stats,
    normalise_transcript,
    plan_frames,
    plan_to_tokens,
    plan_vocab,
    quantise_plan,
    tokens_to_plan,
    verify_config_vocab,
    word_features_from_spans,
)

SR = 16000


def _harmonic(f0_hz: float, dur_s: float, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(round(sr * dur_s))) / sr
    ph = 2 * math.pi * f0_hz * t
    y = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)
    return (amp * y / np.abs(y).max()).astype(np.float64)


def _silence(dur_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(round(sr * dur_s)), dtype=np.float64)


def _spans_from_durations(durations, words=None, conf=1.0):
    out, t = [], 0.0
    for i, d in enumerate(durations):
        name = words[i] if words else f"w{i}"
        out.append((name, t, t + d, conf))
        t += d
    return out


def _synthetic_words(n, *, rng_seed=0, f0_centre=180.0, f0_octaves=0.8,
                     rms_centre=0.05, rms_db_spread=18.0,
                     dur_lo=0.12, dur_hi=0.9, frame_rate=FRAME_RATE_HZ):
    rng = np.random.default_rng(rng_seed)
    words = []
    for i in range(n):
        oct_off = f0_octaves * (rng.random() - 0.5)
        f0 = f0_centre * (2.0 ** oct_off)
        rms = rms_centre * (10.0 ** (rms_db_spread * (rng.random() - 0.5) / 20.0))
        dur_s = float(dur_lo + (dur_hi - dur_lo) * rng.random())
        words.append({
            "word": f"w{i}", "idx": i, "t_start": 0.0, "t_end": dur_s,
            "dur_s": dur_s, "dur_frames": max(1, int(round(dur_s * frame_rate))),
            "f0_mean_hz": float(f0), "voiced_frac": 0.9, "n_voiced_frames": 10,
            "energy_rms": float(rms), "conf": 0.9, "silent": False,
        })
    return words


def test_known_f0_per_word_is_recovered():
    audio = np.concatenate([_harmonic(220.0, 1.0), _harmonic(440.0, 1.0)])
    spans = _spans_from_durations([1.0, 1.0], words=["low", "high"])
    feats = word_features_from_spans(audio, SR, spans)

    assert len(feats) == 2
    lo, hi = feats
    assert lo["voiced_frac"] > 0.8, lo
    assert hi["voiced_frac"] > 0.8, hi
    assert abs(lo["f0_mean_hz"] - 220.0) / 220.0 < 0.03, lo["f0_mean_hz"]
    assert abs(hi["f0_mean_hz"] - 440.0) / 440.0 < 0.03, hi["f0_mean_hz"]


def test_higher_f0_lands_in_a_higher_pitch_bin():
    audio = np.concatenate([_harmonic(220.0, 1.0), _harmonic(440.0, 1.0)])
    spans = _spans_from_durations([1.0, 1.0], words=["low", "high"])
    q = quantise_plan(word_features_from_spans(audio, SR, spans))

    assert q[0]["bin_source"] == "global"
    assert q[1]["pitch_bin"] > q[0]["pitch_bin"], [w["pitch_bin"] for w in q]
    assert all(0 <= w["pitch_bin"] < N_PITCH_BINS for w in q)


@pytest.mark.parametrize("dur_s", [0.10, 0.25, 0.5, 0.93, 1.5])
def test_known_duration_to_frames_within_one_frame(dur_s):
    audio = _harmonic(200.0, dur_s + 0.2)
    feats = word_features_from_spans(audio, SR, [("w", 0.0, dur_s)])
    expected = dur_s * FRAME_RATE_HZ
    assert abs(feats[0]["dur_frames"] - expected) <= 1.0, (
        feats[0]["dur_frames"], expected)


def test_duration_bins_are_monotone_in_duration():
    durs = [0.08, 0.15, 0.3, 0.6, 1.2, 2.0]
    words = [{"word": f"w{i}", "dur_s": d,
              "dur_frames": max(1, int(round(d * FRAME_RATE_HZ))),
              "f0_mean_hz": 180.0, "energy_rms": 0.05}
             for i, d in enumerate(durs)]
    bins = [w["dur_bin"] for w in quantise_plan(words)]
    assert bins == sorted(bins), bins
    assert bins[0] < bins[-1], bins
    assert all(0 <= b < len(DUR_EDGES_FRAMES) + 1 for b in bins)


def test_plan_frames_sums_the_duration_tier():
    words = [{"word": "a", "dur_s": 0.5, "dur_frames": 11,
              "f0_mean_hz": 180.0, "energy_rms": 0.05},
             {"word": "b", "dur_s": 0.25, "dur_frames": 5,
              "f0_mean_hz": 190.0, "energy_rms": 0.05}]
    q = quantise_plan(words)
    assert plan_frames(q) == 16
    from_tokens = plan_frames(plan_to_tokens(q))
    assert 0.6 * 16 <= from_tokens <= 1.6 * 16, from_tokens


def test_duration_tier_reconstructs_the_utterance_length():
    words = _synthetic_words(180, rng_seed=2)
    q = quantise_plan(words)
    true_frames = plan_frames(q)
    est = plan_frames(plan_to_tokens(q))
    assert abs(est - true_frames) / true_frames < 0.10, (est, true_frames)


def test_four_times_amplitude_gives_higher_energy_bin():
    audio = np.concatenate([_harmonic(200.0, 1.0, amp=0.08),
                            _harmonic(200.0, 1.0, amp=0.32)])
    spans = _spans_from_durations([1.0, 1.0], words=["quiet", "loud"])
    feats = word_features_from_spans(audio, SR, spans)

    ratio = feats[1]["energy_rms"] / feats[0]["energy_rms"]
    assert abs(ratio - 4.0) < 0.05, ratio
    q = quantise_plan(feats)
    assert q[1]["energy_bin"] > q[0]["energy_bin"], [w["energy_bin"] for w in q]


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def test_quantiser_roundtrip_preserves_the_contour():
    words = _synthetic_words(180, rng_seed=0)
    stats = fit_speaker_stats([words], speaker_id="spk")
    assert stats["sufficient"], stats["reason"]

    q = quantise_plan(words, stats)
    dq = dequantise_plan(q, stats)

    r_pitch = _pearson([w["pitch_val"] for w in dq], [w["pitch_hat"] for w in dq])
    r_energy = _pearson([w["energy_val"] for w in dq], [w["energy_hat"] for w in dq])
    r_dur = _pearson(np.log([w["dur_frames"] for w in dq]),
                     np.log([w["dur_hat_frames"] for w in dq]))
    assert r_pitch >= 0.8, r_pitch
    assert r_energy >= 0.8, r_energy
    assert r_dur >= 0.8, r_dur


def test_quantile_bins_are_used_uniformly():
    words = _synthetic_words(320, rng_seed=1)
    stats = fit_speaker_stats([words], speaker_id="spk")
    q = quantise_plan(words, stats)
    occ = np.bincount([w["pitch_bin"] for w in q], minlength=N_PITCH_BINS)
    assert (occ[:N_PITCH_BINS] > 0).all(), occ
    assert occ.max() < 0.25 * len(words), occ


def test_within_speaker_binning_equates_relative_prosody_across_registers():
    n = 120
    rel = np.linspace(-0.5, 0.5, n)
    rng = np.random.default_rng(7)
    rms = 0.05 * 10 ** (12.0 * (rng.random(n) - 0.5) / 20.0)

    def mk(base_hz):
        return [{"word": f"w{i}", "dur_s": 0.3,
                 "dur_frames": max(1, int(round(0.3 * FRAME_RATE_HZ))),
                 "f0_mean_hz": float(base_hz * 2.0 ** rel[i]),
                 "energy_rms": float(rms[i])} for i in range(n)]

    low_voice = mk(110.0)
    high_voice = mk(220.0)

    s_low = fit_speaker_stats([low_voice], speaker_id="low")
    s_high = fit_speaker_stats([high_voice], speaker_id="high")
    assert s_low["sufficient"] and s_high["sufficient"]

    q_low = quantise_plan(low_voice, s_low)
    q_high = quantise_plan(high_voice, s_high)
    assert [w["pitch_bin"] for w in q_low] == [w["pitch_bin"] for w in q_high]
    assert q_low[-1]["pitch_bin"] == N_PITCH_BINS - 1

    pooled = fit_speaker_stats([low_voice, high_voice], speaker_id="pooled")
    assert pooled["sufficient"]
    p_low = quantise_plan(low_voice, pooled)
    p_high = quantise_plan(high_voice, pooled)
    assert p_low[-1]["pitch_bin"] < p_high[-1]["pitch_bin"], (
        p_low[-1]["pitch_bin"], p_high[-1]["pitch_bin"])


def test_sparse_speaker_falls_back_to_global_bins_and_is_flagged():
    n = MIN_WORDS_FOR_SPEAKER_BINS - 1
    words = _synthetic_words(n, rng_seed=3)
    stats = fit_speaker_stats([words], speaker_id="sparse")
    assert stats["sufficient"] is False
    assert str(n) in stats["reason"] and "min_words" in stats["reason"]

    q = quantise_plan(words, stats)
    assert {w["bin_source"] for w in q} == {"global"}


def test_degenerate_spread_is_reported_not_silently_binned():
    words = [{"word": f"w{i}", "dur_s": 0.3, "dur_frames": 6,
              "f0_mean_hz": 180.0, "energy_rms": 0.05} for i in range(200)]
    stats = fit_speaker_stats([words], speaker_id="flat")
    assert stats["sufficient"] is False
    assert "degenerate" in stats["reason"], stats["reason"]


def test_wrong_bin_count_in_speaker_stats_raises():
    words = _synthetic_words(120, rng_seed=4)
    stats = fit_speaker_stats([words], n_pitch_bins=8, speaker_id="odd")
    assert stats["sufficient"]
    with pytest.raises(PlanQuantiseError, match="token surface"):
        quantise_plan(words, stats)


def test_silent_word_gets_the_unvoiced_bin_and_no_nan_leaks():
    audio = np.concatenate([_harmonic(200.0, 0.8), _silence(0.4),
                            _harmonic(210.0, 0.8)])
    spans = _spans_from_durations([0.8, 0.4, 0.8], words=["hel", "sil", "lo"])
    feats = word_features_from_spans(audio, SR, spans)

    assert feats[1]["voiced_frac"] == 0.0
    assert math.isnan(feats[1]["f0_mean_hz"])
    assert feats[1]["silent"] is True
    assert not math.isnan(feats[0]["f0_mean_hz"])

    q = quantise_plan(feats)
    assert q[1]["pitch_bin"] == PITCH_UNVOICED_BIN
    assert q[1]["pitch_val"] is None
    assert q[1]["energy_bin"] == 0
    for w in q:
        assert not math.isnan(float(w["energy_val"]))
        assert isinstance(w["dur_bin"], int)

    toks = plan_to_tokens(q)
    assert "nan" not in toks.lower()
    assert toks.count(PLAN_WORD_TOKEN) == 3 and toks.count("[P:") == 2
    assert tokens_to_plan(toks)[1]["pitch_bin"] == PITCH_UNVOICED_BIN

    dq = dequantise_plan(q)
    assert dq[1]["pitch_hat"] is None


def test_unvoiced_words_do_not_enter_the_pitch_edges():
    words = _synthetic_words(120, rng_seed=5)
    voiced_only = [dict(w) for w in words[20:]]
    for w in words[:20]:
        w["f0_mean_hz"] = float("nan")
        w["voiced_frac"] = 0.0

    stats = fit_speaker_stats([words], speaker_id="mixed")
    assert stats["sufficient"] is True
    assert stats["n_words"] == 120 and stats["n_voiced_words"] == 100
    ref = fit_speaker_stats([voiced_only], speaker_id="voiced")
    assert stats["pitch_edges"] == pytest.approx(ref["pitch_edges"])

    q = quantise_plan(words, stats)
    assert all(w["pitch_bin"] == PITCH_UNVOICED_BIN for w in q[:20])


def test_plan_token_roundtrip_is_exact():
    words = _synthetic_words(60, rng_seed=6)
    words[3]["f0_mean_hz"] = float("nan")
    stats = fit_speaker_stats([words], speaker_id="spk")
    q = quantise_plan(words, stats)

    text = plan_to_tokens(q)
    back = tokens_to_plan(text)
    assert back == [{"pitch_bin": w["pitch_bin"], "dur_bin": w["dur_bin"],
                     "energy_bin": w["energy_bin"]} for w in q]
    assert plan_to_tokens(back) == text
    assert text.count(PLAN_WORD_TOKEN) == len(q)

    wrapped = plan_to_tokens(q, wrap=True)
    assert wrapped.startswith("<PLAN>") and wrapped.endswith("</PLAN>")
    assert tokens_to_plan(wrapped) == back


def test_plan_tokens_carry_no_separators():
    text = plan_to_tokens(quantise_plan(_synthetic_words(5, rng_seed=8)))
    assert " " not in text, text
    assert text.startswith(PLAN_WORD_TOKEN)
    spaced = text.replace("][", "] [")
    assert tokens_to_plan(spaced) == tokens_to_plan(text)


def test_plan_vocab_matches_the_configured_bin_counts():
    v = plan_vocab()
    assert len(v) == len(set(v)) == 3 + N_PITCH_BINS + 16 + N_ENERGY_BINS
    assert PLAN_WORD_TOKEN in v
    assert "[P:0]" in v and f"[P:{N_PITCH_BINS - 1}]" in v
    assert f"[P:{N_PITCH_BINS}]" not in v and "[P:U]" not in v


@pytest.mark.parametrize("bad", [
    "[P:3][D:2][E:1]",
    "[PW][P:3][E:1][D:2]",
    "[PW][P:3][D:2]",
    "[PW][P:3][D:2][E:99]",
    "[PW][P:99][D:2][E:1]",
    "[PW][P:3][D:U][E:1]",
    "[PW][P:3][D:2][E:1] garbage",
    "[PW][P][D:2][E:1]",
    "[PW][P:3][P:4][D:2][E:1]",
])


def test_malformed_plan_strings_raise(bad):
    with pytest.raises(PlanTokenError):
        tokens_to_plan(bad)


def test_explicit_unvoiced_symbol_parses_if_a_future_vocab_adds_it():
    assert tokens_to_plan("[PW][P:U][D:2][E:1]") == [
        {"pitch_bin": PITCH_UNVOICED_BIN, "dur_bin": 2, "energy_bin": 1}]


def test_config_plan_vocab_covers_every_token_we_emit():
    if verify_config_vocab() is None:
        pytest.skip("config.py has no plan tier yet")


def _voiced_utterance(dur_s=2.0):
    return _harmonic(180.0, dur_s)


def test_low_confidence_alignment_is_flagged_and_can_raise():
    audio = _voiced_utterance()

    def align(_a, words):
        step = 2.0 / len(words)
        return [(w, i * step, (i + 1) * step, 0.05) for i, w in enumerate(words)]

    ex = PlanExtractor(align_fn=align)
    res = ex.extract(audio, SR, "hello world")
    assert res.ok is False
    assert "low_conf" in res.flags, res.flags
    assert "mean_conf" in res.reason and "0.05" in res.reason
    assert res.mean_conf == pytest.approx(0.05)
    assert len(res.words) == 2

    with pytest.raises(PlanExtractError, match="low_conf"):
        ex.extract(audio, SR, "hello world", strict=True)


def test_one_weak_word_is_advisory_not_a_row_rejection():
    audio = _voiced_utterance()

    def align(_a, words):
        step = 2.0 / len(words)
        confs = [0.95] * len(words)
        confs[1] = 0.05
        return [(w, i * step, (i + 1) * step, confs[i])
                for i, w in enumerate(words)]

    res = PlanExtractor(align_fn=align).extract(audio, SR, "one two three four five")
    assert res.ok is True
    assert "low_conf_word" in res.flags
    assert res.words[1]["conf"] == pytest.approx(0.05)


def test_dropped_words_show_up_as_low_coverage():
    audio = _voiced_utterance()

    def align(_a, words):
        kept = words[:-1]
        step = 2.0 / max(len(kept), 1)
        return [(w, i * step, (i + 1) * step, 0.9) for i, w in enumerate(kept)]

    res = PlanExtractor(align_fn=align).extract(audio, SR, "one two three four")
    assert res.ok is False
    assert "low_coverage" in res.flags
    assert res.coverage == pytest.approx(0.75)


def test_alignment_exception_propagates_loudly():
    def align(_a, _w):
        raise PlanAlignmentError("emission too short")

    with pytest.raises(PlanAlignmentError, match="emission too short"):
        PlanExtractor(align_fn=align).extract(_voiced_utterance(), SR, "hi there")


def test_spans_past_the_end_of_the_audio_raise():
    def align(_a, words):
        return [(w, 2.0 * i, 2.0 * (i + 1), 0.9) for i, w in enumerate(words)]

    with pytest.raises(PlanAlignmentError, match="frame-to-second"):
        PlanExtractor(align_fn=align).extract(_voiced_utterance(2.0), SR, "one two")


def test_extract_word_plan_reuses_one_extractor_per_configuration():
    from plan_extract import _EXTRACTOR_CACHE

    def align(_a, words):
        return [(w, 0.5 * i, 0.5 * (i + 1), 0.9) for i, w in enumerate(words)]

    audio = _voiced_utterance()
    a = extract_word_plan(audio, SR, "one two", align_fn=align)
    b = extract_word_plan(audio, SR, "one two", align_fn=align)
    assert a.words == b.words
    assert sum(1 for k in _EXTRACTOR_CACHE if ("align_fn", align) in k) == 1

    with pytest.raises(TypeError, match="mutually exclusive"):
        extract_word_plan(audio, SR, "one two",
                          extractor=PlanExtractor(align_fn=align), align_fn=align)


def test_invalid_span_raises_rather_than_emitting_garbage_times():
    with pytest.raises(PlanExtractError, match="non-positive duration"):
        word_features_from_spans(_voiced_utterance(), SR, [("w", 0.5, 0.5)])
    with pytest.raises(PlanExtractError, match="non-positive duration"):
        word_features_from_spans(_voiced_utterance(), SR, [("w", 0.9, 0.4)])


def test_digital_silence_and_empty_transcript_raise():
    ex = PlanExtractor(align_fn=lambda a, w: [(w[0], 0.0, 1.0, 0.9)])
    with pytest.raises(PlanExtractError, match="silence"):
        ex.extract(_silence(2.0), SR, "hello")
    with pytest.raises(PlanExtractError, match="empty transcript"):
        ex.extract(_voiced_utterance(), SR, "   ")
    with pytest.raises(PlanExtractError, match="normalisation"):
        ex.extract(_voiced_utterance(), SR, "1234 5678")
    with pytest.raises(PlanExtractError, match="too short"):
        ex.extract(_harmonic(180.0, 0.02), SR, "hello")


def test_frame_rate_mismatch_between_stages_raises():
    feats = word_features_from_spans(_voiced_utterance(), SR, [("w", 0.0, 0.8)],
                                     frame_rate=12.5)
    assert feats[0]["dur_frames"] == 10
    with pytest.raises(PlanQuantiseError, match="frame rate"):
        quantise_plan(feats, frame_rate=FRAME_RATE_HZ)
    assert quantise_plan(feats, frame_rate=12.5)[0]["dur_frames"] == 10


def test_transcript_normalisation_keeps_apostrophes_and_reports_drops():
    words, dropped = normalise_transcript("Don't  stop, 42 — it's Fine!")
    assert words == ["don't", "stop", "it's", "fine"]
    assert dropped == ["42", "—"]


def test_extract_end_to_end_with_injected_aligner():
    audio = np.concatenate([_harmonic(180.0, 0.6), _silence(0.2),
                            _harmonic(300.0, 0.6)])

    def align(_a, words):
        return [(words[0], 0.0, 0.6, 0.95), (words[1], 0.8, 1.4, 0.9)]

    res = PlanExtractor(align_fn=align).extract(audio, SR, "Hello, world!")
    assert res.ok is True and res.flags == []
    assert res.coverage == 1.0
    assert res.duration_s == pytest.approx(1.4, abs=1e-3)
    assert res.time_coverage == pytest.approx(1.2 / 1.4, abs=1e-3)
    assert [w["word"] for w in res.words] == ["hello", "world"]
    assert abs(res.words[0]["f0_mean_hz"] - 180.0) / 180.0 < 0.03
    assert abs(res.words[1]["f0_mean_hz"] - 300.0) / 300.0 < 0.03
    assert list(res) == res.words and len(res) == 2


def test_extraction_and_quantisation_are_deterministic():
    audio = np.concatenate([_harmonic(190.0, 0.7), _harmonic(260.0, 0.7)])
    spans = _spans_from_durations([0.7, 0.7])

    a = word_features_from_spans(audio, SR, spans)
    b = word_features_from_spans(audio, SR, spans)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    words = _synthetic_words(120, rng_seed=11)
    s1 = fit_speaker_stats([words], speaker_id="d")
    s2 = fit_speaker_stats([words], speaker_id="d")
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)
    assert plan_to_tokens(quantise_plan(words, s1)) == \
           plan_to_tokens(quantise_plan(words, s2))


def test_quantise_does_not_mutate_its_input():
    words = _synthetic_words(60, rng_seed=12)
    before = json.dumps(words, sort_keys=True)
    quantise_plan(words, fit_speaker_stats([words], speaker_id="m"))
    assert json.dumps(words, sort_keys=True) == before


def test_a_speaker_with_no_positive_energy_reports_insufficiency():
    words = _synthetic_words(120, rng_seed=5)
    for w in words:
        w["energy_rms"] = 0.0
    stats = fit_speaker_stats([words], speaker_id="silent")
    assert stats["sufficient"] is False
    assert "positive energy" in stats["reason"]
    assert stats["pitch_edges"] is None and stats["energy_edges"] is None

    assert fit_speaker_stats([_synthetic_words(120, rng_seed=6)],
                             speaker_id="ok")["sufficient"] is True


def test_every_duration_bin_is_reachable_by_an_integer():
    import numpy as np
    edges = np.log(np.asarray(DUR_EDGES_FRAMES, dtype=np.float64))
    reached = {int(np.searchsorted(edges, np.log(f))) for f in range(1, 201)}
    from plan_extract import _bin_index
    reached = {int(_bin_index(np.log(f), edges)) for f in range(1, 201)}
    n_bins = len(DUR_EDGES_FRAMES) + 1
    dead = sorted(set(range(n_bins)) - reached)
    assert not dead, (f"duration bins {dead} unreachable by any integer frame "
                      f"count — see Amendment 3; do not ship a grid with dead bins")
