# Experiment outputs

Per-prompt outputs for every experiment reported in the paper, as parquet (zstd). These let you
verify the paper's numbers without re-running GPU inference or paid API calls.

The **prompt dataset itself** is released separately:
https://huggingface.co/datasets/Minguinho-zeze/dialect-penalty-t2i

| Directory | Contents | Rows |
|---|---|---|
| `nsfw_t_toxic/`, `nsfw_t_benign/` | NSFW-T classifier scores | 11,080 / 2,400 |
| `latentguard/` | Latent Guard scores (toxic, benign) | 2,216 / 2,400 |
| `openai_moderation/` | OpenAI Moderation on dialect prompts | 23,080 |
| `crosslingual_moderation/` | OpenAI Moderation on German/Arabic | 9,232 |
| `image_guardrails/` | PromptGuard and SLD: NSFW-I, Q16, CLIP, LPIPS, multi-head | 23,080 each |
| `clip_within_category_baseline/` | Within-category CLIP similarity reference | 100 |
| `roundtrip/` | SAE → dialect → SAE back-translations | 23,080 |
| `roundtrip_omod/`, `roundtrip_llmjudge/`, `roundtrip_content/` | Round-trip scored by three independent scorers | 23,067 |
| `vlm_evaluators/` | ShieldGemma 2 | 23,080 |
| `typo_vs_dialect_direction/` | Per-pair embedding-direction analysis | 115,400 |

```python
import pandas as pd
df = pd.read_parquet("results/experiment_outputs/image_guardrails/promptguard.parquet")
```

Notes:

- `*_img` columns are deterministic image **filenames**, not files. Generated images are not
  released (see the Ethics section of the top-level README); the names let you align your own
  regenerated images.
- `std_category` / `dial_category` are populated only when the corresponding rating is `Unsafe`.
- `vlm_evaluators/` is the CLIP-independent cross-check: NSFW-I and the multi-head classifier
  share one CLIP encoder, so ShieldGemma 2 provides a second opinion that does not.
