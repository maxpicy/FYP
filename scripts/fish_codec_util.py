# fish_codec_util.py: Loads the Fish S2 codec from a fish-speech checkout (FISH_SPEECH_REPO, FISH_CODEC_PTH).

import os
import sys
from pathlib import Path

import torch

_REPO_DEFAULT = "/scratch/users/ntu/mwong048/FYP-Mamba/external/fish-speech"
_REPO_FALLBACK = "/home/max/fish-speech"
_CKPT_DEFAULT = "/scratch/users/ntu/mwong048/FYP-Mamba/fish_codec.pth"
_CKPT_FALLBACK = "/mnt/c/Users/Max/PycharmProjects/FYP-Mamba/fish_pilot/codec.pth"


def _resolve(env_var, default, fallback):
    value = os.environ.get(env_var)
    if value:
        return value
    for candidate in (default, fallback):
        if Path(candidate).exists():
            return candidate
    return default


def fish_repo_path():
    return _resolve("FISH_SPEECH_REPO", _REPO_DEFAULT, _REPO_FALLBACK)


def fish_ckpt_path():
    return _resolve("FISH_CODEC_PTH", _CKPT_DEFAULT, _CKPT_FALLBACK)


def load_fish_codec(device):
    repo = fish_repo_path()
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    try:
        OmegaConf.register_new_resolver("eval", eval)
    except Exception:
        pass

    cfg = OmegaConf.load(f"{repo}/fish_speech/configs/modded_dac_vq.yaml")
    model = instantiate(cfg)

    ckpt = fish_ckpt_path()
    sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if any("generator" in k for k in sd):
        sd = {k.replace("generator.", ""): v
              for k, v in sd.items() if "generator." in k}
    model.load_state_dict(sd, strict=False, assign=True)
    model.eval().to(device)
    return model


def encode_np(model, wav_np, sr, device, max_duration_s=None):
    import numpy as np
    import soxr

    try:
        wav = np.asarray(wav_np, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=0) if wav.shape[0] < wav.shape[1] else wav.mean(axis=1)
        if wav.ndim != 1 or wav.size == 0:
            return None

        if max_duration_s is not None:
            max_samples = int(max_duration_s * sr)
            if wav.shape[0] > max_samples:
                wav = wav[:max_samples]

        target_sr = int(model.sample_rate)
        if int(sr) != target_sr:
            wav = soxr.resample(wav, int(sr), target_sr)
        wav = np.ascontiguousarray(wav, dtype=np.float32)
        if wav.size == 0:
            return None

        x = torch.from_numpy(wav).float()[None, None, :].to(device)
        lens = torch.tensor([x.shape[-1]], device=device, dtype=torch.long)
        with torch.no_grad():
            indices, idx_lens = model.encode(x, lens)
        codes = indices[0]
        if idx_lens is not None:
            t_valid = int(idx_lens[0])
            if 0 < t_valid <= codes.shape[-1]:
                codes = codes[:, :t_valid]
        if codes.numel() == 0:
            return None
        return [row.tolist() for row in codes.cpu()]
    except Exception:
        return None
