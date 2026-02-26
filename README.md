# Emotional TTS
### Name: Maximilian Wong Junlin
### FYP ID: CCDS25-1035
### Project Title: generating emotional speech (emotional TTS)
### Supervisor: Prof Chng Eng Siong

### Start Date: 12 Jan 2026
### End Date: 19 Oct 2026



## Overview
Many Current TTS models suffer from a fundamental limitation, they map text directly to acoustic features without understanding the underlying context, performing well but failing to express emotion natrually or accurately, struggling to properly convey more nuanced natrualspeech. While Transformer-based models can be forced to reason using Chain-of-Thought (CoT), their quadratic complexity makes streaming Reasoning expensive and introduces  latency.

This Project aims address the challenges of latency and emotional expressivity by integrating Mamba-2 with Interleaved Recall with Reasoning (RwR) in a TTS context.


## Video Updates

Playlist: 

[YouTube Playlist](https://youtube.com/playlist?list=PLSZgsZc5eaKnr3baTQ-nRyDfnee3KgHU4)

## Roadmap / Milestones
* Phase 1: Environment setup and Mamba-2 baseline integration.
* Phase 2: Synthesize the Audio-CoT reasoning dataset using GPT-4o teacher distillation.
* Phase 3: Pretrain the Mamba-2 layers on the generated dataset.
* Phase 4: Post-training and optimizing model
* Phase 4: Benchmark against other models.
<img width="2752" height="1502" alt="unnamed" src="https://github.com/user-attachments/assets/b4e31f16-dbd8-4c7b-a3c2-81c4ab7c8918" />


## References

*	H. Zen, K. Tokuda, and A. W. Black, “Statistical parametric speech synthesis,” Speech Commun., vol. 51, no. 11, pp. 1039–1064, Nov. 2009, doi: 10.1016/j.specom.2009.04.004. Available: http://dx.doi.org/10.1016/j.specom.2009.04.004
*	Y. Chen et al., “F5-TTS: A fairytaler that fakes fluent and faithful speech with flow matching,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Stroudsburg, PA, USA: Association for Computational Linguistics, 2025, pp. 6255–6271. doi: 10.18653/v1/2025.acl-long.313. Available: http://dx.doi.org/10.18653/v1/2025.acl-long.313
*	S. E. Eskimez et al., “E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS,” arXiv [eess.AS], Jun. 25, 2024. Available: http://arxiv.org/abs/2406.18009
*	Z. Du et al., “CosyVoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens,” arXiv [cs.SD], Jul. 07, 2024. Available: http://arxiv.org/abs/2407.05407
*	Z. Du et al., “CosyVoice 2: Scalable streaming speech synthesis with large language models,” arXiv [cs.SD], Dec. 13, 2024. Available: http://arxiv.org/abs/2412.10117
*	Z. Du et al., “CosyVoice 3: Towards in-the-wild speech generation via scaling-up and post-training,” arXiv [cs.SD], May 23, 2025. Available: http://arxiv.org/abs/2505.17589
*	Z. Ye et al., “Llasa: Scaling train-time and inference-time compute for Llama-based speech synthesis,” arXiv [eess.AS], Feb. 06, 2025. Available: http://arxiv.org/abs/2502.04128
*	B. Han et al., “VALL-E R: Robust and efficient zero-shot Text-to-speech synthesis via monotonic alignment,” arXiv [cs.CL], Jun. 12, 2024. Available: http://arxiv.org/abs/2406.07855
*	H. Hu et al., “Qwen3-TTS Technical Report,” arXiv [cs.SD], Jan. 22, 2026. doi: 10.48550/arXiv.2601.15621. Available: http://dx.doi.org/10.48550/arXiv.2601.15621. [Accessed: Feb. 20, 2026]
*	A. Vaswani et al., “Attention is all you need,” Aug. 23, 2025. doi: 10.65215/mdcm8z23. Available: https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf. [Accessed: Feb. 20, 2026]
*	H. E. Shim et al., “Generating consistent prosodic patterns from open-source TTS systems,” in Interspeech 2025, ISCA: ISCA, Aug. 2025, pp. 5383–5387. doi: 10.21437/interspeech.2025-2159. Available: http://dx.doi.org/10.21437/interspeech.2025-2159
*	C. Pouw, A. Alishahi, and W. Zuidema, “A linguistically motivated analysis of intonational phrasing in text-to-speech systems: Revealing gaps in syntactic sensitivity,” in Proceedings of the 29th Conference on Computational Natural Language Learning, Stroudsburg, PA, USA: Association for Computational Linguistics, 2025, pp. 126–140. doi: 10.18653/v1/2025.conll-1.9. Available: http://dx.doi.org/10.18653/v1/2025.conll-1.9
*	S. Que and A. Ragni, “VisualSpeech: Enhancing prosody modeling in TTS using video,” in Interspeech 2025, ISCA: ISCA, Aug. 2025, pp. 3778–3782. doi: 10.21437/interspeech.2025-1494. Available: http://dx.doi.org/10.21437/interspeech.2025-1494
*	T. Dao and A. Gu, “Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality,” arXiv [cs.LG], May 31, 2024. Available: http://arxiv.org/abs/2405.21060
*	A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv [cs.LG], Dec. 01, 2023. Available: http://arxiv.org/abs/2312.00752
*	S. Kumar, N. Patel, H. Wang, and Y. Zhang, “MambaVoiceCloning: Efficient and expressive text-to-speech via state-space modeling and diffusion control,” in The Fourteenth International Conference on Learning Representations, Oct. 2025. Available: https://openreview.net/pdf?id=0oXyMbPMtP. [Accessed: Feb. 20, 2026]
*	J. Wei et al., “Chain-of-thought prompting elicits reasoning in large language models,” arXiv [cs.CL], Jan. 27, 2022. Available: http://arxiv.org/abs/2201.11903
*	“MINI-OMNI-REASONER: Token-Level Thinking-in-Speaking in Large Speech Models.” Available: https://arxiv.org/html/2508.15827v2. [Accessed: Feb. 20, 2026]
*	Z. Xie and C. Wu, “Mini-Omni: Language models can hear, talk while thinking in streaming,” arXiv [cs.AI], Aug. 29, 2024. Available: http://arxiv.org/abs/2408.16725
*	Y. Yang et al., “Interleaved Speech-text language models for simple streaming text-to-speech synthesis,” arXiv [eess.AS], Aug. 09, 2025. Available: http://arxiv.org/abs/2412.16102
*	J.-Y. Ma, T. Fang, Z. Zhang, H. Zhang, H. Mi, and D. Yu, “Recall with reasoning: Chain-of-thought distillation for mamba’s long-context memory and extrapolation,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Stroudsburg, PA, USA: Association for Computational Linguistics, 2025, pp. 4714–4720. doi: 10.18653/v1/2025.emnlp-main.235. Available: http://dx.doi.org/10.18653/v1/2025.emnlp-main.235
