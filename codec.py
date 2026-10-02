# codec.py: Codec wrappers and the load_codec factory (the Fish S2 codec itself is in scripts/fish_codec_util.py).

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio

from config import NUM_SPEECH_TOKENS

logger = logging.getLogger(__name__)


class CodecWrapper:
    def __init__(
        self,
        model,
        device: str,
        sample_rate: int,
        frame_rate: float,
        codebook_size: int,
    ):
        self.model = model
        self.device = device
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.codebook_size = codebook_size

    def encode(self, audio_path: str, max_duration_s: float = 30.0) -> List[int]:
        path = Path(audio_path)
        if not path.exists():
            raise ValueError(f"Audio file not found: {audio_path}")

        try:
            waveform, sr = torchaudio.load(str(path))
        except Exception as e:
            raise ValueError(f"Failed to load audio {audio_path}: {e}") from e

        return self.encode_waveform(waveform, sr, max_duration_s=max_duration_s)

    def encode_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        max_duration_s: float = 30.0,
    ) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> Tuple[np.ndarray, int]:
        raise NotImplementedError

    @property
    def num_codebooks(self) -> int:
        return 1

    def encode_waveform_multi(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        max_duration_s: float = 30.0,
    ) -> List[List[int]]:
        return [self.encode_waveform(waveform, sample_rate, max_duration_s=max_duration_s)]

    def decode_multi(self, codes: List[List[int]]) -> Tuple[np.ndarray, int]:
        if not codes:
            return np.array([], dtype=np.float32), self.sample_rate
        return self.decode(codes[0])

    @property
    def tokens_per_second(self) -> float:
        return self.frame_rate


class XCodec2Wrapper(CodecWrapper):
    MODEL_ID = "HKUSTAudio/xcodec2"
    SAMPLE_RATE = 16000
    FRAME_RATE = 50.0
    CODEBOOK_SIZE = 65536

    def __init__(self, device: str = "cuda"):
        from xcodec2.modeling_xcodec2 import XCodec2Model, BigCodecConfig
        from huggingface_hub import hf_hub_download
        import safetensors.torch

        logger.info("Loading X-Codec2 from %s...", self.MODEL_ID)

        config = BigCodecConfig.from_pretrained(self.MODEL_ID)
        model = XCodec2Model(config)

        weights_path = hf_hub_download(self.MODEL_ID, "model.safetensors")
        state_dict = safetensors.torch.load_file(weights_path)
        remapped = {
            k.replace(".act.beta", ".act.bias"): v for k, v in state_dict.items()
        }
        model.load_state_dict(remapped, strict=True)
        model = model.to(device).eval()

        logger.info("X-Codec2 loaded on %s", device)

        super().__init__(
            model=model,
            device=device,
            sample_rate=self.SAMPLE_RATE,
            frame_rate=self.FRAME_RATE,
            codebook_size=self.CODEBOOK_SIZE,
        )

    def encode_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        max_duration_s: float = 30.0,
    ) -> List[int]:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.SAMPLE_RATE
            )

        if max_duration_s > 0:
            max_samples = int(max_duration_s * self.SAMPLE_RATE)
            if waveform.shape[1] > max_samples:
                logger.warning(
                    "Truncating audio from %.1fs to %.1fs",
                    waveform.shape[1] / self.SAMPLE_RATE,
                    max_duration_s,
                )
                waveform = waveform[:, :max_samples]

        if waveform.shape[1] == 0:
            logger.warning("Empty waveform — returning empty token list")
            return []

        with torch.no_grad():
            codes = self.model.encode_code(waveform.to(self.device))

        tokens = codes.squeeze().cpu().tolist()
        if isinstance(tokens, (int, float)):
            tokens = [int(tokens)]

        out_of_range = [t for t in tokens if t < 0 or t >= self.CODEBOOK_SIZE]
        if out_of_range:
            logger.warning(
                "%d tokens out of range [0, %d), clamping",
                len(out_of_range),
                self.CODEBOOK_SIZE,
            )
            tokens = [max(0, min(t, self.CODEBOOK_SIZE - 1)) for t in tokens]

        return tokens

    def decode(self, tokens: List[int]) -> Tuple[np.ndarray, int]:
        if not tokens:
            return np.array([], dtype=np.float32), self.SAMPLE_RATE

        codes = torch.tensor([[tokens]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            audio = self.model.decode_code(codes)

        waveform = audio.squeeze().cpu().numpy()
        return waveform.astype(np.float32), self.SAMPLE_RATE


class MimiCodecWrapper(CodecWrapper):
    MODEL_ID = "kyutai/mimi"
    SAMPLE_RATE = 24000
    FRAME_RATE = 12.5
    CODEBOOK_SIZE = 2048
    DEFAULT_NUM_QUANTIZERS = 8

    def __init__(
        self,
        device: str = "cuda",
        num_quantizers: int = DEFAULT_NUM_QUANTIZERS,
    ):
        from transformers import MimiModel

        logger.info("Loading Mimi from %s (num_quantizers=%d)...",
                    self.MODEL_ID, num_quantizers)
        model = MimiModel.from_pretrained(self.MODEL_ID).to(device).eval()

        max_q = getattr(model.config, "num_quantizers", 32)
        if not 1 <= num_quantizers <= max_q:
            raise ValueError(
                f"num_quantizers={num_quantizers} out of range "
                f"[1, {max_q}] for Mimi"
            )
        self.num_quantizers = num_quantizers

        super().__init__(
            model=model,
            device=device,
            sample_rate=self.SAMPLE_RATE,
            frame_rate=self.FRAME_RATE,
            codebook_size=self.CODEBOOK_SIZE,
        )

    @property
    def num_codebooks(self) -> int:
        return self.num_quantizers

    def _prepare_input(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        max_duration_s: float,
    ) -> Optional[torch.Tensor]:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.SAMPLE_RATE
            )
        if max_duration_s > 0:
            cap = int(max_duration_s * self.SAMPLE_RATE)
            if waveform.shape[1] > cap:
                logger.warning(
                    "Truncating audio from %.1fs to %.1fs",
                    waveform.shape[1] / self.SAMPLE_RATE,
                    max_duration_s,
                )
                waveform = waveform[:, :cap]
        if waveform.shape[1] == 0:
            logger.warning("Empty waveform — returning empty token list")
            return None
        return waveform.unsqueeze(0).to(self.device)

    def _encode_codes(self, prepared: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model.encode(
                prepared, num_quantizers=self.num_quantizers, return_dict=True,
            )
        codes = out.audio_codes if hasattr(out, "audio_codes") else out[0]
        return codes[0]

    def encode_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        max_duration_s: float = 30.0,
    ) -> List[int]:
        prepared = self._prepare_input(waveform, sample_rate, max_duration_s)
        if prepared is None:
            return []
        codes = self._encode_codes(prepared)
        return codes[0].cpu().tolist()

    def encode_waveform_multi(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        max_duration_s: float = 30.0,
    ) -> List[List[int]]:
        prepared = self._prepare_input(waveform, sample_rate, max_duration_s)
        if prepared is None:
            return []
        codes = self._encode_codes(prepared)
        return codes.cpu().tolist()

    def decode(self, tokens: List[int]) -> Tuple[np.ndarray, int]:
        if not tokens:
            return np.array([], dtype=np.float32), self.SAMPLE_RATE
        T = len(tokens)
        codes = torch.zeros(
            1, self.num_quantizers, T, dtype=torch.long, device=self.device,
        )
        codes[0, 0] = torch.tensor(tokens, dtype=torch.long, device=self.device)
        return self._decode_codes(codes)

    def decode_multi(self, codes: List[List[int]]) -> Tuple[np.ndarray, int]:
        if not codes or not codes[0]:
            return np.array([], dtype=np.float32), self.SAMPLE_RATE
        if len(codes) != self.num_quantizers:
            logger.warning(
                "decode_multi got %d codebooks, expected %d — padding/truncating",
                len(codes), self.num_quantizers,
            )
            T = len(codes[0])
            full = torch.zeros(
                self.num_quantizers, T, dtype=torch.long, device=self.device,
            )
            for k, layer in enumerate(codes[: self.num_quantizers]):
                full[k] = torch.tensor(layer, dtype=torch.long, device=self.device)
            codes_t = full.unsqueeze(0)
        else:
            codes_t = torch.tensor([codes], dtype=torch.long, device=self.device)
        return self._decode_codes(codes_t)

    def _decode_codes(self, codes: torch.Tensor) -> Tuple[np.ndarray, int]:
        with torch.no_grad():
            out = self.model.decode(codes, return_dict=True)
        audio = out.audio_values if hasattr(out, "audio_values") else out[0]
        waveform = audio.squeeze().cpu().numpy()
        return waveform.astype(np.float32), self.SAMPLE_RATE

    def codes_to_latents(self, codes) -> torch.Tensor:
        codes_t = codes if torch.is_tensor(codes) else torch.tensor(
            codes, dtype=torch.long, device=self.device)
        if codes_t.dim() == 2:
            codes_t = codes_t.unsqueeze(0)
        codes_t = codes_t.to(self.device)
        with torch.no_grad():
            return self.model.quantizer.decode(codes_t)

    def encode_latents(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        max_duration_s: float = 30.0,
    ) -> Optional[torch.Tensor]:
        prepared = self._prepare_input(waveform, sample_rate, max_duration_s)
        if prepared is None:
            return None
        m = self.model
        with torch.no_grad():
            emb = m.encoder(prepared)
            enc = m.encoder_transformer(emb.transpose(1, 2), return_dict=True)
            emb = enc[0].transpose(1, 2)
            if getattr(m, "downsample", None) is not None:
                emb = m.downsample(emb)
        return emb

    def load_decoder_ft(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        sd = ckpt.get("decoder_state", ckpt)
        buckets = {"upsample": {}, "decoder_transformer": {}, "decoder": {}}
        for k, v in sd.items():
            name, sub = k.split(".", 1)
            buckets[name][sub] = v
        loaded = 0
        for name, part in buckets.items():
            mod = getattr(self.model, name, None)
            if mod is not None and part:
                mod.load_state_dict(part)
                loaded += len(part)
        logger.info("Loaded fine-tuned decoder from %s (%d tensors)", path, loaded)

    def decode_latents(self, latents: torch.Tensor) -> Tuple[np.ndarray, int]:
        lat = latents if latents.dim() == 3 else latents.unsqueeze(0)
        lat = lat.to(self.device)
        m = self.model
        with torch.no_grad():
            emb = m.upsample(lat) if getattr(m, "upsample", None) is not None else lat
            dec = m.decoder_transformer(emb.transpose(1, 2), return_dict=True)
            emb = dec[0].transpose(1, 2)
            audio = m.decoder(emb)
        waveform = audio.squeeze().cpu().numpy()
        return waveform.astype(np.float32), self.SAMPLE_RATE


def load_codec(codec_name: str = "xcodec2", device: str = "cuda", **kwargs) -> CodecWrapper:
    if codec_name == "xcodec2":
        try:
            return XCodec2Wrapper(device=device, **kwargs)
        except ImportError as e:
            raise ImportError(
                "xcodec2 package not installed. Run: pip install xcodec2"
            ) from e
    elif codec_name == "mimi":
        try:
            return MimiCodecWrapper(device=device, **kwargs)
        except ImportError as e:
            raise ImportError(
                "Mimi requires transformers >= 4.45 (with MimiModel). "
                "Run: pip install -U transformers"
            ) from e
    elif codec_name == "qwen12hz":
        raise NotImplementedError(
            "Qwen-12Hz codec not implemented; Mimi (codec_name='mimi') is the "
            "recommended Phase-9 production tier."
        )
    else:
        raise ValueError(
            f"Unknown codec: {codec_name!r}. "
            f"Supported: 'xcodec2', 'mimi', 'qwen12hz' (NotImplemented)."
        )


def decode_speech_to_file(
    token_ids: List[int],
    registry,
    output_path: str,
    codec_name: str = "xcodec2",
    device: str = "cuda",
) -> None:
    import soundfile as sf

    codec_values = [
        registry.speech_id_to_value(tid)
        for tid in token_ids
        if registry.is_speech_token(tid)
    ]

    if not codec_values:
        raise ValueError("No speech tokens found in the input sequence")

    codec = load_codec(codec_name, device=device)
    waveform, sr = codec.decode(codec_values)
    sf.write(output_path, waveform, sr)
    logger.info("Saved decoded audio to %s (%.2fs)", output_path, len(waveform) / sr)
