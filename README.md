# Emotional TTS
### Name: Maximilian Wong Junlin
### FYP ID: CCDS25-1035
### Project Title: generating emotional speech (emotional TTS)
### Supervisor: Prof Chng Eng Siong

### Start Date: 12 Jan 2026
### End Date: 19 Oct 2026



## Overview
Many current TTS models suffer from a fundamental limitation: they map text directly to acoustic features without understanding the underlying context. They perform well but fail to express emotion naturally or accurately, and struggle to convey more nuanced natural speech. While Transformer-based models can be forced to reason using Chain-of-Thought (CoT), their quadratic complexity makes streaming reasoning expensive and introduces latency.

This project measures whether a pure Mamba-2 backbone can learn block-sequential CoT for expressive speech. Before it speaks, the model writes a think block with three parts:
* a short description of the delivery;
* emotion, pace and pitch tags;
* a word-level prosody plan of pitch, duration and energy.

It then generates the speech as Fish S2 codec tokens. The backbone generates codebook 0, and a small Mamba-2 depth module generates codebooks 1 to 9. A hybrid (the top 4 Mamba layers replaced by attention) and a transformer (Pythia-1.4B) were trained the same way, on the same data, as the comparison.

With their own plans, the transformer leads on every evaluation list, and the hybrid shows no reliable difference from the pure model. The paper below has the full results.


## Video Updates

Playlist: 

[YouTube Playlist](https://youtube.com/playlist?list=PLSZgsZc5eaKnr3baTQ-nRyDfnee3KgHU4)

## Roadmap / Milestones
* Phase 1: Environment setup and Mamba-2 baseline integration. (done)
* Phase 2: Build the CoT dataset: delivery descriptions from teacher LLMs and published style captions, plus word-level prosody plans measured from the audio. (done)
* Phase 3: Train the Mamba-2 models on the dataset, with a hybrid and a transformer trained the same way. (done)
* Phase 4: Post-training and optimising the models: the two-stage recipe and the decode. (done)
* Phase 5: Benchmark on the field's evaluation lists (Seed-TTS, LibriSpeech-PC, EmergentTTS-Eval). (done)
<img width="2752" height="1502" alt="unnamed" src="https://github.com/user-attachments/assets/b4e31f16-dbd8-4c7b-a3c2-81c4ab7c8918" />


## Models and Data
The weights, the evaluation results and the training corpora are in one Hugging Face repository, [maxpicy/p2cot-tts](https://huggingface.co/maxpicy/p2cot-tts):
* `two_stage_v2/`: the two-stage models for each backbone (C3, C2, and C3 trained with a second seed).
* `stage1p_bases/`: the Stage 1′ bases they were trained from.
* `stage_a/`: the one-stage (Stage A) models.
* `path_b_renderer/`: the fine-tuned s1-mini renderer used for Path B, with the voice 4200 / 4201 reference clips.
* `eval/`: every evaluated cell (think blocks, codes, per-row scores), the paired statistics, and an audio sample.
* `corpus/`: the Stage 1′ and Stage 2 training corpora, in parquet.
* `code/`: a copy of this repository's code.

## Setup
The models were trained and evaluated with Python 3.11, CUDA 11.8 and A100 GPUs, in the `pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel` container. Install the model environment first:

```bash
pip install -r requirements.txt --no-build-isolation
```

Decoding codes to audio needs a second environment, the Fish environment (`requirements-fish.txt`), plus two downloads:
* a [fish-speech](https://github.com/fishaudio/fish-speech) checkout at commit `bc72d1f`;
* `codec.pth` from [fishaudio/openaudio-s1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini), which is gated and licensed CC BY-NC-SA 4.0.

Point `FISH_SPEECH_REPO` at the checkout and `FISH_CODEC_PTH` at `codec.pth`. `env/` lists both environments exactly as installed.

## Recreating the Results
**1. Download a model and the prompt lists**
```bash
hf download maxpicy/p2cot-tts --include "two_stage_v2/pure_c3/*" --local-dir hf
hf download maxpicy/p2cot-tts --include "eval/prompt_sets/*" --local-dir hf
```

**2. Decode**

`decode.sh` uses the paper's decode settings: voice 4200, seed 1234, fp32, decode H temperatures, the two repairs, and the cached state. It writes the think block and the codes. If `FISH_PYTHON` is set, it also writes wav files.
```bash
FISH_PYTHON=/path/to/fish/env/bin/python bash decode.sh hf/two_stage_v2/pure_c3 hf/eval/prompt_sets/ext_seedtts_en.jsonl out/pure_c3 selfplan
```
Use `noplan` instead of `selfplan` for the no-plan cell. `ROWS=a:b` decodes only those rows, with the same per-row seeds as a full run.

**3. Score**

The scoring follows the field's protocol: whisper-large-v3 with the Whisper normaliser for WER, plus UTMOS22 and completion.
```bash
python scripts/eval_suite.py --arm selfplan out/pure_c3/selfplan.jsonl out/pure_c3/selfplan \
    --whisper openai/whisper-large-v3 --whisper_norm whisper --out out/pure_c3/scores.json
```

**4. Compare cells**

The paper's scores per cell and per row are under `eval/scores/` on Hugging Face. Paired contrasts come with a bootstrap interval, an exact sign test and Holm correction:
```bash
python scripts/paired_stats.py --pair A.json:selfplan B.json:selfplan
```

**5. Path B**

Path B regenerates codebooks 1 to 9 with the fine-tuned s1-mini renderer, keeping our codebook 0. Download [fishaudio/openaudio-s1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini), copy `path_b_renderer/model.pth` and `config.json` over its files, then:
```bash
python scripts/s1mini_render.py --codes out/pure_c3/selfplan.jsonl --out out/pure_c3/pathb.jsonl \
    --checkpoint <s1-mini folder with our model.pth> --ref_dir hf/path_b_renderer/refs \
    --voice_map 4200=spk4200,4201=spk4201 --resume
```

**6. Retrain Stage 2 from a Stage 1′ base**
```bash
python scripts/fetch_corpus.py --config stage2_v2 --out_dir data
hf download maxpicy/p2cot-tts --include "stage1p_bases/pure/*" --local-dir hf
STAGE=2 ARM=pure TIER=full INIT=hf/stage1p_bases/pure/model.safetensors DATA=data/stage2_v2_train.jsonl bash train.sh
```
* `TIER=full` trains C3, and `TIER=tags+plan` trains C2 from the same file.
* `SEED=2` gives the second training seed.
* Stage 2 runs 40K steps on 4 A100s; the pure C3 run took about 4 hours.

`STAGE=1p` trains Stage 1′ (`scripts/fetch_corpus.py --config stage1p`) from a Stage 1 base. The Stage 1 bases are not released yet.

**Tests**
```bash
python -m pytest tests -q
```
Tests that need a GPU or model weights skip themselves when those are absent.

## Licence
To be decided before the models and data are released. The training data include non-commercial and share-alike sources, and the Fish codec is CC BY-NC-SA 4.0.

## Contact
* **Email**: [max.wjl@gmail.com](mailto:max.wjl@gmail.com)
* **LinkedIn**: [linkedin.com/in/maximilian-wong-933008b5](https://www.linkedin.com/in/maximilian-wong-933008b5)

## Paper
* Maximilian Wong Junlin, “[Paper title],” Final Year Project report, Nanyang Technological University, 2026. Available: [https://www.ntu.edu.sg/](https://www.ntu.edu.sg/)
