# tokenizer.py: Extends the GPT-NeoX tokenizer with the project's tokens; TokenRegistry gives O(1) id lookups and
# the speech-token mask.

import torch
from typing import Dict, List, Optional, Set, Tuple
from transformers import AutoTokenizer

BASE_TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"

BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"

CONTROL_TOKENS = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]

USER_PROMPT_TOKEN = "[USER_PROMPT]"

THINK_START_TOKEN = "<THINK>"
THINK_END_TOKEN = "</THINK>"

COT_DELIMITER_TOKENS = [THINK_START_TOKEN, THINK_END_TOKEN]

AUDIO_START_TOKEN = "<AUDIO>"
AUDIO_END_TOKEN = "</AUDIO>"

AUDIO_DELIMITER_TOKENS = [AUDIO_START_TOKEN, AUDIO_END_TOKEN]

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

from config import NUM_SPEECH_TOKENS, SPEAKER_TOKEN
SPEECH_TOKEN_PREFIX = "[SPEECH_"
SPEECH_TOKENS = [f"[SPEECH_{i}]" for i in range(NUM_SPEECH_TOKENS)]

from config import (
    ENABLE_PLAN_TOKENS,
    ENERGY_BINS,
    DURATION_BINS,
    NUM_STYLE_TOKENS,
    PITCH_BINS,
    PLAN_END_TOKEN,
    PLAN_START_TOKEN,
    PLAN_WORD_SEP_TOKEN,
    get_plan_duration_tokens,
    get_plan_energy_tokens,
    get_plan_pitch_tokens,
    get_plan_tokens,
    get_style_tokens,
)

PLAN_TOKENS: List[str] = get_plan_tokens() if ENABLE_PLAN_TOKENS else []


def get_all_new_tokens() -> List[str]:
    all_tokens = []
    all_tokens.extend(CONTROL_TOKENS)
    all_tokens.append(USER_PROMPT_TOKEN)
    all_tokens.extend(COT_DELIMITER_TOKENS)
    all_tokens.extend(AUDIO_DELIMITER_TOKENS)
    all_tokens.append(SPEAKER_TOKEN)
    all_tokens.extend(EMOTION_TAGS)
    all_tokens.extend(PROSODY_TAGS)
    all_tokens.extend(PLAN_TOKENS)
    all_tokens.extend(SPEECH_TOKENS)
    return all_tokens


def load_base_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(BASE_TOKENIZER_NAME)


def expand_tokenizer(tokenizer) -> int:
    all_new = get_all_new_tokens()

    special_tokens_dict = {
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "additional_special_tokens": all_new,
    }

    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    return num_added


def _resolve_exact(tokenizer, token: str) -> Optional[int]:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        return None
    return token_id if tokenizer.convert_ids_to_tokens(token_id) == token else None


class TokenRegistry:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.bos_id: int = tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        self.eos_id: int = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        self.pad_id: int = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        self.user_prompt_id: int = tokenizer.convert_tokens_to_ids(USER_PROMPT_TOKEN)
        self.think_start_id: int = tokenizer.convert_tokens_to_ids(THINK_START_TOKEN)
        self.think_end_id: int = tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
        self.audio_start_id: int = tokenizer.convert_tokens_to_ids(AUDIO_START_TOKEN)
        self.audio_end_id: int = tokenizer.convert_tokens_to_ids(AUDIO_END_TOKEN)
        self.speaker_id: int = tokenizer.convert_tokens_to_ids(SPEAKER_TOKEN)

        self.emotion_ids: Set[int] = set(
            tokenizer.convert_tokens_to_ids(EMOTION_TAGS)
        )
        self.prosody_ids: Set[int] = set(
            tokenizer.convert_tokens_to_ids(PROSODY_TAGS)
        )
        self.speech_token_ids: Set[int] = set(
            tokenizer.convert_tokens_to_ids(SPEECH_TOKENS)
        )

        speech_id_list = sorted(self.speech_token_ids)
        self.speech_token_id_min: int = speech_id_list[0]
        self.speech_token_id_max: int = speech_id_list[-1]

        assert self.speech_token_id_max - self.speech_token_id_min + 1 == NUM_SPEECH_TOKENS, (
            f"Speech token IDs are not contiguous: "
            f"min={self.speech_token_id_min}, max={self.speech_token_id_max}, "
            f"expected range={NUM_SPEECH_TOKENS}"
        )

        self._speech_value_to_id: Dict[int, int] = {
            i: tokenizer.convert_tokens_to_ids(f"[SPEECH_{i}]")
            for i in range(NUM_SPEECH_TOKENS)
        }
        self._speech_id_to_value: Dict[int, int] = {
            v: k for k, v in self._speech_value_to_id.items()
        }

        self._init_plan_ids(tokenizer)

    def _init_plan_ids(self, tokenizer) -> None:
        self.plan_enabled: bool = False
        self.plan_start_id: Optional[int] = None
        self.plan_end_id: Optional[int] = None
        self.plan_word_sep_id: Optional[int] = None
        self.pitch_ids: List[int] = []
        self.duration_ids: List[int] = []
        self.energy_ids: List[int] = []
        self.style_ids: List[int] = []
        self.plan_token_ids: Set[int] = set()
        self.plan_token_id_min: Optional[int] = None
        self.plan_token_id_max: Optional[int] = None
        self._plan_id_to_bin: Dict[int, Tuple[str, int]] = {}

        probe = _resolve_exact(tokenizer, PLAN_START_TOKEN)

        if not ENABLE_PLAN_TOKENS:
            if probe is not None:
                raise RuntimeError(
                    f"Tokenizer contains {PLAN_START_TOKEN} but MVC_ENABLE_PLAN_TOKENS "
                    f"is off in this process. The vocabulary and the code would "
                    f"disagree about the P²-CoT plan block — export "
                    f"MVC_ENABLE_PLAN_TOKENS=1 (and the same MVC_NUM_STYLE_TOKENS) "
                    f"before importing config/tokenizer."
                )
            return

        if probe is None:
            raise RuntimeError(
                f"MVC_ENABLE_PLAN_TOKENS=1 but this tokenizer has no "
                f"{PLAN_START_TOKEN} token. expand_tokenizer() must be called on "
                f"the tokenizer in a process where the gate is already on (the "
                f"token list is built at import time)."
            )

        expected = get_plan_tokens()
        resolved: List[int] = []
        missing: List[str] = []
        for token in expected:
            token_id = _resolve_exact(tokenizer, token)
            if token_id is None:
                missing.append(token)
            else:
                resolved.append(token_id)
        if missing:
            raise RuntimeError(
                f"{len(missing)} P²-CoT plan token(s) missing from the tokenizer "
                f"(e.g. {missing[:4]}). The tokenizer was expanded with a different "
                f"PITCH/DURATION/ENERGY_BINS or MVC_NUM_STYLE_TOKENS than this "
                f"process defines ({PITCH_BINS}/{DURATION_BINS}/{ENERGY_BINS}/"
                f"{NUM_STYLE_TOKENS})."
            )

        n_pitch, n_dur, n_energy = PITCH_BINS, DURATION_BINS, ENERGY_BINS
        self.plan_start_id, self.plan_end_id, self.plan_word_sep_id = resolved[:3]
        cursor = 3
        self.pitch_ids = resolved[cursor:cursor + n_pitch]; cursor += n_pitch
        self.duration_ids = resolved[cursor:cursor + n_dur]; cursor += n_dur
        self.energy_ids = resolved[cursor:cursor + n_energy]; cursor += n_energy
        self.style_ids = resolved[cursor:cursor + NUM_STYLE_TOKENS]

        self.plan_token_ids = set(resolved)
        if len(self.plan_token_ids) != len(resolved):
            raise RuntimeError(
                "Plan token ids are not unique — two plan tokens resolved to the "
                "same id, which means the tokenizer collapsed them (check that "
                "expand_tokenizer() added them as additional_special_tokens)."
            )
        self.plan_token_id_min = min(resolved)
        self.plan_token_id_max = max(resolved)

        if self.plan_token_id_max - self.plan_token_id_min + 1 != len(resolved):
            raise RuntimeError(
                f"Plan token ids are not contiguous: min={self.plan_token_id_min}, "
                f"max={self.plan_token_id_max}, count={len(resolved)}. The plan "
                f"block must be added in one run by expand_tokenizer()."
            )

        if self.plan_token_id_max >= self.speech_token_id_min:
            raise RuntimeError(
                f"Plan block (ids {self.plan_token_id_min}..{self.plan_token_id_max}) "
                f"overlaps or follows the speech block (from "
                f"{self.speech_token_id_min}). Plan tokens must be added BEFORE "
                f"the speech tokens — see the ordering note in config.py."
            )

        for family, ids in (
            ("pitch", self.pitch_ids),
            ("duration", self.duration_ids),
            ("energy", self.energy_ids),
            ("style", self.style_ids),
        ):
            for bin_index, token_id in enumerate(ids):
                self._plan_id_to_bin[token_id] = (family, bin_index)

        self.plan_enabled = True

    def _require_plan(self) -> None:
        if not self.plan_enabled:
            raise RuntimeError(
                "P²-CoT plan tokens are not in this tokenizer (built with "
                "MVC_ENABLE_PLAN_TOKENS=0). Export MVC_ENABLE_PLAN_TOKENS=1 before "
                "importing config/tokenizer, or branch on `registry.plan_enabled` / "
                "`registry.plan_start_id is None`."
            )

    @staticmethod
    def _as_bin_index(value, family: str, num_bins: int) -> int:
        if isinstance(value, (str, bytes)) or isinstance(value, bool):
            raise ValueError(f"{family} bin {value!r} is not an integer bin index")
        try:
            index = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{family} bin {value!r} is not an integer bin index"
            ) from None
        if index != value:
            raise ValueError(
                f"{family} bin {value!r} is not integral (would truncate to {index})"
            )
        if not 0 <= index < num_bins:
            raise ValueError(
                f"{family} bin {index} out of range [0, {num_bins})"
            )
        return index

    def _bin_to_id(self, ids: List[int], value, family: str, num_bins: int) -> int:
        self._require_plan()
        return ids[self._as_bin_index(value, family, num_bins)]

    def _id_to_bin(self, token_id: int, family: str) -> int:
        self._require_plan()
        entry = self._plan_id_to_bin.get(int(token_id))
        if entry is None or entry[0] != family:
            raise ValueError(
                f"token id {token_id} is not a {family} plan token "
                f"(got {entry[0] if entry else 'non-plan token'})"
            )
        return entry[1]

    def pitch_bin_to_id(self, bin_index) -> int:
        return self._bin_to_id(self.pitch_ids, bin_index, "pitch", PITCH_BINS)

    def id_to_pitch_bin(self, token_id: int) -> int:
        return self._id_to_bin(token_id, "pitch")

    def duration_bin_to_id(self, bin_index) -> int:
        return self._bin_to_id(self.duration_ids, bin_index, "duration", DURATION_BINS)

    def id_to_duration_bin(self, token_id: int) -> int:
        return self._id_to_bin(token_id, "duration")

    def energy_bin_to_id(self, bin_index) -> int:
        return self._bin_to_id(self.energy_ids, bin_index, "energy", ENERGY_BINS)

    def id_to_energy_bin(self, token_id: int) -> int:
        return self._id_to_bin(token_id, "energy")

    def style_index_to_id(self, index) -> int:
        return self._bin_to_id(self.style_ids, index, "style", NUM_STYLE_TOKENS)

    def id_to_style_index(self, token_id: int) -> int:
        return self._id_to_bin(token_id, "style")

    def is_plan_token(self, token_id: int) -> bool:
        if self.plan_token_id_min is None:
            return False
        return self.plan_token_id_min <= token_id <= self.plan_token_id_max

    def build_plan_mask(self, input_ids: torch.Tensor) -> torch.BoolTensor:
        if self.plan_token_id_min is None:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        return (input_ids >= self.plan_token_id_min) & (input_ids <= self.plan_token_id_max)

    def is_speech_token(self, token_id: int) -> bool:
        return self.speech_token_id_min <= token_id <= self.speech_token_id_max

    def is_emotion_token(self, token_id: int) -> bool:
        return token_id in self.emotion_ids

    def is_prosody_token(self, token_id: int) -> bool:
        return token_id in self.prosody_ids

    def speech_value_to_id(self, value: int) -> int:
        return self._speech_value_to_id[value]

    def speech_id_to_value(self, token_id: int) -> int:
        return self._speech_id_to_value[token_id]

    def build_speech_mask(self, input_ids: torch.Tensor) -> torch.BoolTensor:
        return (input_ids >= self.speech_token_id_min) & (input_ids <= self.speech_token_id_max)
