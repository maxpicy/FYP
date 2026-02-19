# FYP

## Overview
Many Current TTS models suffer from a fundamental limitation, they map text directly to acoustic features without understanding the underlying context. They are highly performant but lack empathy, struggling to properly render subtext, irony, or sarcasm. While Transformer-based models can be forced to reason using Chain-of-Thought (CoT), their quadratic complexity makes streaming Reasoning expensive and introduces  latency.

This Project solves this by integrating Mamba-2 with Recall with Reasoning (RwR), by distilling reasoning traces from a teacher LLM, this model actively generates a thought chain to evaluate emotional intent before generating speech within $O(N)$ linear time and with a constant memory footprint $O(1)$.

## Video Updates

Playlist: 

[YouTube Playlist](https://youtube.com/playlist?list=PLSZgsZc5eaKnr3baTQ-nRyDfnee3KgHU4)

## Roadmap / Milestones
Phase 1: Environment setup and Mamba-2 baseline integration.
Phase 2: Synthesize the Audio-CoT reasoning dataset using GPT-4o teacher distillation.
Phase 3: Train the Mamba-2 layers on the generated dataset.
Phase 4: Benchmark against other models.
