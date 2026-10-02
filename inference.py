# inference.py: Decoding helpers shared with the generator: prefix construction, constrained decoding, token extraction.

from __future__ import annotations

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def build_inference_prefix(
    tokenizer,
    registry,
    prompt: str,
    stage: int,
) -> torch.Tensor:
    if stage not in (1, 2):
        raise ValueError(f"Unsupported stage {stage}; must be 1 or 2")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if stage == 1:
        prefix = (
            [registry.bos_id, registry.user_prompt_id]
            + prompt_ids
            + [registry.audio_start_id]
        )
    else:
        prefix = (
            [registry.bos_id, registry.user_prompt_id]
            + prompt_ids
            + [registry.think_start_id]
        )

    return torch.tensor([prefix], dtype=torch.long)


def extract_speech_token_ids(
    generated_ids: List[int],
    registry,
    prefix_len: int,
) -> List[int]:
    if prefix_len > len(generated_ids):
        return []

    tail = generated_ids[prefix_len:]
    speech_ids: List[int] = []
    started = False

    for tid in tail:
        if registry.is_speech_token(tid):
            speech_ids.append(tid)
            started = True
        elif started:
            break
        elif tid == registry.audio_end_id or tid == registry.eos_id:
            break

    return speech_ids


def _flatten_generation_output(output) -> List[int]:
    if hasattr(output, "sequences"):
        tensor = output.sequences
    else:
        tensor = output

    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 2:
            return tensor[0].tolist()
        return tensor.tolist()

    return list(tensor)


def _sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    eff_temperature = max(temperature, 1e-4)

    if top_k and top_k > 0:
        top_vals, top_ix = torch.topk(logits, k=min(top_k, logits.numel()))
        masked = torch.full_like(logits, float("-inf"))
        masked[top_ix] = top_vals
        logits = masked

    probs = torch.softmax(logits / eff_temperature, dim=-1)

    if top_p and top_p < 1.0:
        sorted_probs, sorted_ix = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-12)
        next_ix = torch.multinomial(sorted_probs, 1)
        return int(sorted_ix.gather(-1, next_ix).item())
    return int(torch.multinomial(probs, 1).item())


def _build_speech_mask(model, device) -> torch.Tensor:
    reg = model.token_registry
    vocab_size = model.backbone.config.vocab_size
    allowed = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    allowed[reg.speech_token_id_min : reg.speech_token_id_max + 1] = True
    return allowed


def _speech_constrained_decode(
    model,
    prefix: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    level_cycle: int = 0,
    level_temps: Optional[List[float]] = None,
    stop_ids: Optional[List[int]] = None,
) -> List[int]:
    device = prefix.device
    allowed = _build_speech_mask(model, device)
    neg_inf = torch.tensor(float("-inf"), device=device)

    level_masks = None
    if level_cycle > 0:
        reg = model.token_registry
        id_min = reg.speech_token_id_min
        n_speech = reg.speech_token_id_max - id_min + 1
        csize = n_speech // level_cycle
        level_masks = []
        for lvl in range(level_cycle):
            m = torch.zeros_like(allowed)
            m[id_min + lvl * csize: id_min + (lvl + 1) * csize] = True
            level_masks.append(m & allowed)

    stop_mask = None
    if stop_ids:
        stop_mask = torch.zeros_like(allowed)
        for sid in stop_ids:
            stop_mask[sid] = True

    seq = prefix.clone()
    speech_ids: List[int] = []

    with torch.no_grad():
        for i in range(max_new_tokens):
            out = model(input_ids=seq)
            logits = out["logits"][0, -1].float()
            step_mask = (
                level_masks[i % level_cycle] if level_masks is not None else allowed
            )
            if stop_mask is not None and (level_cycle == 0 or i % level_cycle == 0):
                step_mask = step_mask | stop_mask
            logits = torch.where(step_mask, logits, neg_inf)

            step_temp = temperature
            if level_masks is not None and level_temps:
                step_temp = level_temps[(i % level_cycle) % len(level_temps)]
            next_id = _sample_from_logits(
                logits, temperature=step_temp, top_p=top_p, top_k=top_k,
            )
            if stop_ids and next_id in stop_ids:
                break
            speech_ids.append(next_id)
            seq = torch.cat(
                [seq, torch.tensor([[next_id]], device=device, dtype=seq.dtype)],
                dim=-1,
            )

    return speech_ids


def speculative_generate(
    model,
    prefix: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    speech_only: bool = False,
) -> tuple[List[int], dict]:
    if temperature > 0:
        raise NotImplementedError(
            "speculative_generate currently supports greedy decoding only "
            "(temperature=0). Sampling-mode speculative requires rejection "
            "sampling (Leviathan et al.) and is deferred to a later phase."
        )

    reg = model.token_registry
    device = prefix.device

    if speech_only:
        speech_mask = _build_speech_mask(model, device)
        neg_inf = torch.tensor(float("-inf"), device=device)

        def constrain(logits: torch.Tensor) -> torch.Tensor:
            return torch.where(speech_mask, logits, neg_inf)
    else:
        def constrain(logits: torch.Tensor) -> torch.Tensor:
            return logits

    seq = prefix.clone()
    generated: List[int] = []
    iterations = 0
    drafts_accepted = 0

    with torch.no_grad():
        while len(generated) < max_new_tokens:
            iterations += 1

            out = model(input_ids=seq)
            main_logits = constrain(out["logits"][0, -1].float())
            t = int(main_logits.argmax().item())
            generated.append(t)
            if len(generated) >= max_new_tokens:
                seq = torch.cat(
                    [seq, torch.tensor([[t]], device=device, dtype=seq.dtype)],
                    dim=-1,
                )
                break

            hidden_T = out["hidden_states"][0, -1]
            draft = int(model.mtp_module.predict_offset_2(hidden_T).item())
            if speech_only:
                h_normed = model.mtp_module.spec_shared_norm(hidden_T.unsqueeze(0))
                spec_logits = constrain(
                    model.mtp_module.spec_shared_head(h_normed)[0].float()
                )
                draft = int(spec_logits.argmax().item())

            seq = torch.cat(
                [seq, torch.tensor([[t, draft]], device=device, dtype=seq.dtype)],
                dim=-1,
            )
            out_v = model(input_ids=seq)
            verify_logits = constrain(out_v["logits"][0, -2].float())
            verify_t1 = int(verify_logits.argmax().item())

            if verify_t1 == draft:
                generated.append(draft)
                drafts_accepted += 1
            else:
                seq = seq[:, :-1]
                seq[0, -1] = t
                generated.append(verify_t1)
                seq = torch.cat(
                    [seq, torch.tensor([[verify_t1]], device=device, dtype=seq.dtype)],
                    dim=-1,
                )

    if len(generated) > max_new_tokens:
        generated = generated[:max_new_tokens]

    meta = {
        "iterations": iterations,
        "drafts_proposed": iterations,
        "drafts_accepted": drafts_accepted,
        "acceptance_rate": drafts_accepted / max(iterations, 1),
        "tokens_emitted": len(generated),
        "forward_passes": 2 * iterations,
    }
    return generated, meta


def _cot_constrained_decode(
    model,
    prefix: torch.Tensor,
    *,
    max_new_tokens: int,
    max_think_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[List[int], List[int], dict]:
    reg = model.token_registry
    device = prefix.device
    speech_mask = _build_speech_mask(model, device)
    neg_inf = torch.tensor(float("-inf"), device=device)

    seq = prefix.clone()
    think_ids: List[int] = []
    speech_ids: List[int] = []
    meta = {"think_truncated": False, "transition": None}

    THINKING, AUDIO = 0, 1
    state = THINKING
    think_count = 0
    tokens_used = 0

    def emit(token_id: int):
        nonlocal seq, tokens_used
        seq = torch.cat(
            [seq, torch.tensor([[token_id]], device=device, dtype=seq.dtype)],
            dim=-1,
        )
        tokens_used += 1

    with torch.no_grad():
        while tokens_used < max_new_tokens:
            if state == THINKING:
                if think_count >= max_think_tokens:
                    meta["think_truncated"] = True
                    meta["transition"] = "forced"
                    emit(reg.think_end_id)
                    if tokens_used < max_new_tokens:
                        emit(reg.audio_start_id)
                    state = AUDIO
                    continue

                out = model(input_ids=seq)
                logits = out["logits"][0, -1].float()
                next_id = _sample_from_logits(
                    logits, temperature=temperature, top_p=top_p, top_k=top_k,
                )

                if next_id == reg.think_end_id:
                    meta["transition"] = "think_end"
                    emit(next_id)
                    if tokens_used < max_new_tokens:
                        emit(reg.audio_start_id)
                    state = AUDIO
                elif next_id == reg.audio_start_id:
                    meta["transition"] = "audio_start_direct"
                    emit(reg.think_end_id)
                    if tokens_used < max_new_tokens:
                        emit(next_id)
                    state = AUDIO
                else:
                    think_ids.append(next_id)
                    think_count += 1
                    emit(next_id)

            else:
                out = model(input_ids=seq)
                logits = out["logits"][0, -1].float()
                logits = torch.where(speech_mask, logits, neg_inf)
                next_id = _sample_from_logits(
                    logits, temperature=temperature, top_p=top_p, top_k=top_k,
                )
                speech_ids.append(next_id)
                emit(next_id)

    return think_ids, speech_ids, meta


def generate_speech(
    model,
    prompt: str,
    output_path: str,
    *,
    stage: int = 1,
    max_new_tokens: int = 1500,
    max_think_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    constrained: bool = False,
    speculative: bool = False,
    codec=None,
    codec_name: str = "xcodec2",
    device: Optional[str] = None,
) -> dict:
    import soundfile as sf
    from codec import load_codec

    if device is None:
        device = getattr(model, "device", "cuda")

    tokenizer = model.tokenizer
    registry = model.token_registry

    input_ids = build_inference_prefix(tokenizer, registry, prompt, stage)
    prefix_len = input_ids.shape[1]
    input_ids = input_ids.to(device)

    cot_meta = None
    spec_meta = None
    think_ids: List[int] = []

    if speculative and stage == 2:
        raise NotImplementedError(
            "speculative=True with stage=2 (CoT-aware decode) is not "
            "supported in Phase 11 — the THINK→AUDIO state machine and "
            "the draft-and-verify loop have multiple interaction points "
            "that need careful design. Use stage=1 + speculative for now."
        )

    if speculative:
        gen_tokens, spec_meta = speculative_generate(
            model, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            speech_only=constrained,
        )
        generated_ids = input_ids[0].tolist() + gen_tokens
        if constrained:
            speech_ids = list(gen_tokens)
        else:
            speech_ids = extract_speech_token_ids(
                generated_ids, registry, prefix_len,
            )
    elif constrained and stage == 2:
        think_ids, speech_ids, cot_meta = _cot_constrained_decode(
            model, input_ids,
            max_new_tokens=max_new_tokens,
            max_think_tokens=max_think_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        boundary_pair = [registry.think_end_id, registry.audio_start_id]
        if cot_meta and cot_meta.get("transition") == "audio_start_direct":
            pass
        generated_ids = (
            input_ids[0].tolist() + think_ids + boundary_pair + speech_ids
        )
    elif constrained:
        speech_ids = _speech_constrained_decode(
            model, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        generated_ids = input_ids[0].tolist() + speech_ids
    else:
        max_length = prefix_len + max_new_tokens
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        generated_ids = _flatten_generation_output(output)
        speech_ids = extract_speech_token_ids(generated_ids, registry, prefix_len)

    if not speech_ids:
        raise ValueError(
            f"No speech tokens generated (prefix_len={prefix_len}, "
            f"total_generated={len(generated_ids)})"
        )

    if codec is None:
        codec = load_codec(codec_name, device=device)

    codec_values = [registry.speech_id_to_value(tid) for tid in speech_ids]
    waveform, sr = codec.decode(codec_values)
    sf.write(output_path, waveform, sr)

    duration = len(waveform) / sr if sr else 0.0
    logger.info(
        "Generated %d speech tokens -> %s (%.2fs)",
        len(speech_ids),
        output_path,
        duration,
    )

    result = {
        "output_path": output_path,
        "num_speech_tokens": len(speech_ids),
        "duration_seconds": duration,
        "sample_rate": sr,
        "token_ids": generated_ids,
    }
    if cot_meta is not None:
        result["cot_meta"] = cot_meta
        result["think_token_count"] = len(think_ids)
    if spec_meta is not None:
        result["spec_meta"] = spec_meta
    return result
