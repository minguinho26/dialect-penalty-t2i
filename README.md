# Not Safe for All: Auditing the Dialect Penalty in Text-to-Image Safety Pipelines

[![arXiv](https://img.shields.io/badge/arXiv-2608.29589-b31b1b.svg)](https://arxiv.org/abs/2608.29589)

Official code for our EMNLP 2026 Findings paper.

This paper investigates the **dialect penalty** in text-to-image safety systems, where filters flag inputs based on the dialect being used instead of the actual meaning of the prompt. We test text-level filters, an unguarded generator, and image-based guardrails using roughly 23,000 paired Standard American English (SAE) and dialect prompts. Our findings reveal that certain safety mechanisms heavily over-censor harmless dialect prompts, while others completely fail to catch toxic ones.

**Dialects:** African American Vernacular English (AAVE), Chicano English (ChcE), Colloquial Singapore English (CollSgE), Indian English
(IndE), Jamaican English (JamE), each paired against SAE.

**Dataset:** https://huggingface.co/datasets/Minguinho-zeze/dialect-penalty-t2i

## Layout

| Directory | Contents |
|---|---|
| `common/` | Shared utilities, safety-model wrappers, credentials, environment checks |
| `data_construction/` | Dialect translation and benign prompt generation |
| `stage1_text_level/` | Text filters: NSFW-T, Latent Guard, keyword, OpenAI Moderation |
| `stage2_image_level/` | End-to-end: Stable Diffusion, PromptGuard, Safe Latent Diffusion |
| `stage3_mitigation/` | ERM / balanced sampling / GroupDRO, and Table 8 |
| `extended_analysis/` | Round-trip audit, evaluator audit, typo-vs-dialect direction |
| `analysis_notebooks/` | Figures and tables |
| `results/` | Aggregate tables, plus per-prompt `experiment_outputs/` |

## Setup

```bash
git clone https://github.com/minguinho26/dialect-penalty-t2i.git
cd dialect-penalty-t2i

pip install -r requirements.txt          # stages 1 and 3, analysis
pip install -r requirements-stage2.txt   # only if running stage 2

python common/check_env.py               # verify before running anything
```

`requirements.txt` does not install torch. Container images ship torch with torchvision and
torchaudio built against the same ABI, and replacing torch alone breaks the others in a way that
kills `import transformers`. Use the image's torch; see the header of `requirements.txt` if you
need to install it yourself. `check_env.py` reports what is broken and how to fix it.

**API keys** are read from environment variables only, never hardcoded. Copy `.env.example` to
`.env` (gitignored) and fill in what you need: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, `HF_TOKEN`. Verify with `python common/env_keys.py`. Only data construction and
the back-translation audit need these.

**Data and weights.** Run scripts from the repository root; paths are root-relative.

```bash
hf auth login && python common/fetch_data.py   # prompts from HuggingFace
python common/prepare_evaluator_weights.py     # Q16, LAION NSFW detector
```

The multi-head classifier is not ours, so we don't ship it: download its five `.pt` files from [Unsafe Diffusion](https://github.com/YitingQu/unsafe-diffusion/tree/main/checkpoints/multi-headed) into `./multihead_checkpoints/`. The [Latent Guard](https://github.com/rt219/LatentGuard) checkpoint likewise comes from its own repository. Any script requiring them will throw a helpful error on startup if they are missing.

## Reproducing the paper

```bash
# Stage 1: text-level filters
python stage1_text_level/exp_text_level_nsfw_t.py
python stage1_text_level/exp_text_level_latentguard.py
python stage1_text_level/ablation_openai_moderation_api.py

# Stage 2: end-to-end pipeline (GPU; generated images are not released)
python stage2_image_level/exp_image_level_promptguard.py --dialect all
python stage2_image_level/exp_image_level_safe_latent_diffusion_one_loop.py --dialect all

# Stage 3: mitigation, Table 8 (see stage3_mitigation/README.md)
cd stage3_mitigation && bash run_all.sh
```

We've saved the per-prompt outputs for all experiments in `results/experiment_outputs/`. This means you can easily verify the numbers reported in our paper without having to spend money on API calls or run heavy GPU inference yourself.

> **Mitigation backbone.** For Stage 3, we don't just keep training the existing NSFW-T model. While
> `eval_only_exp1.py` tests `michellejieli/NSFW_text_classifier` as a zero-shot baseline,
> `train_group_dro.py` takes the same base model (`distilbert/distilbert-base-uncased`) and trains a
> completely new classifier from scratch using our paired dialect data. Since NSFW-T is also just a
> fine-tuned DistilBERT, doing it this way lets us isolate the effect of the training data and
> objective, rather than accidentally inheriting the flaws of NSFW-T's decision boundary.

We've released some of the DistilBERT classifiers we trained in Stage 3 on the Hub, so you can try them without retraining: [`...-guardrail-erm-balsampling`](https://huggingface.co/Minguinho-zeze/dialect-penalty-t2i-guardrail-erm-balsampling) (ERM with group-balanced sampling) and [`...-guardrail-groupdro`](https://huggingface.co/Minguinho-zeze/dialect-penalty-t2i-guardrail-groupdro) (Group DRO).

## Attribution

Our toxic prompts are sourced from **T2I-RiskyPrompt** ([repo](https://github.com/datar001/T2I-RiskyPrompt),
[paper](https://arxiv.org/abs/2510.22300)), which doesn't specify an explicit license. We are sharing them
along with our dialect translations strictly for non-commercial safety research, but we will gladly take
them down if the original authors ask us to.

```bibtex
@article{zhang2025t2iriskyprompt,
  title   = {T2I-RiskyPrompt: A Benchmark for Safety Evaluation, Attack, and Defense on Text-to-Image Model},
  author  = {Zhang, Chenyu and Zhang, Tairen and Wang, Lanjun and Chen, Ruidong and Li, Wenhui and Liu, Anan},
  journal = {arXiv preprint arXiv:2510.22300},
  year    = {2025}
}
```

Our harmless prompts come from **COCO Captions** (CC BY 4.0), and we followed the **EnDive** framework ([paper](https://arxiv.org/abs/2504.07100)) for selecting and translating dialects. The `stage3_mitigation/group_DRO/loss.py` script was borrowed from **Sagawa et al.** ([repo](https://github.com/kohpangwei/group_DRO), MIT). Our multi-head image evaluator is **Unsafe Diffusion** ([repo](https://github.com/YitingQu/unsafe-diffusion)) — those trained classifier heads are theirs, and we don't redistribute them. Like T2I-RiskyPrompt, that repository does not state an explicit license, so please follow up with its authors for terms. You can find the full data provenance and exact model details in [DATA.md](DATA.md).

## Ethics

This repository is designed to audit dialect bias in T2I safety systems. We include toxic prompts purely for testing how well the filters handle them.

We do not release any generated images. Note that all dialect translations were generated by LLMs and checked for semantic consistency using round-trip translations, rather than by native speakers. Therefore, these translations act as controlled stress tests for the filters—they should not be treated as authentic representations of any specific speech community. Additionally, all image-level labels rely on automated classifiers without human verification.

## License

MIT ([LICENSE](LICENSE)), except `stage3_mitigation/group_DRO/`, which keeps its upstream MIT
license. Redistributed third-party data remains under its original terms; see Attribution and
[DATA.md](DATA.md).

## Citation

```bibtex
@misc{audit_dialect,
      title={Not Safe for All: Auditing the Dialect Penalty in Text-to-Image Safety Pipelines}, 
      author={Minkyu Kim and Juhwan Choi and YoungBin Kim},
      booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
      year={2026},
      url={https://arxiv.org/abs/2608.29589}, 
}
```
