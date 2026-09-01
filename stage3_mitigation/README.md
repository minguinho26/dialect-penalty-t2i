# Stage 3: mitigation (Table 8)

Trains a dialect-robust text safety classifier under three regimes and reports the paper's
dialect-penalty metrics.

## What this does *not* do

It does **not** continue training from the NSFW-T checkpoint. `eval_only_exp1.py` evaluates
`michellejieli/NSFW_text_classifier` zero-shot as a baseline; `train_group_dro.py` starts from the
*same backbone* (`distilbert/distilbert-base-uncased`) and trains a fresh classifier on our paired
dialect data. NSFW-T is itself a DistilBERT fine-tune, so this isolates the effect of the training
distribution and objective rather than inheriting NSFW-T's decision boundary.

## Run

```bash
pip install -r requirements-minimal.txt   # stage 3 needs no vision or API deps
python ../common/check_env.py             # verify the environment before training
bash run_all.sh          # from this directory; regenerates data/ then runs every experiment
```

On a container image that already ships torch (RunPod, Colab, NGC), keep the image's torch.
`requirements-minimal.txt` deliberately does not pin it: those images build torchvision and
torchaudio against their own torch, so replacing torch alone leaves both compiled against the old
ABI. transformers touches them at import time and dies before reaching any of our code:

```
RuntimeError: operator torchvision::nms does not exist
OSError: libtorchaudio.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...
```

transformers guards these imports with `is_torchvision_available()` / `is_torchaudio_available()`,
but those only check whether the package is *installed*, not whether it *loads* — so a broken
install passes the guard and then fails. Stage 3 needs neither package:

```bash
pip uninstall -y torchvision torchaudio
```

`python ../common/check_env.py` reports exactly which package is broken and which command fixes
it.

Or a single run:

```bash
python prepare_data.py
python train_group_dro.py --robust --seed 0 --output_dir ./results/dro_s0
```

Python 3.10 (the original environment) and 3.12 both work; the numpy pin switches automatically,
because numpy 1.24.3 has no 3.12 wheel and its source build fails on that interpreter.

**Pin `transformers`.** `warmup_ratio` was deprecated and then removed upstream; on a release that
dropped it, `TrainingArguments` raises `TypeError`. The script detects this and converts the ratio
to an equivalent `warmup_steps` rather than silently training without warmup, but the pinned
version reproduces the reported runs exactly.

`data/` is rebuilt from `original_dataset/` by `prepare_data.py`, so it is not checked in.

## Groups

12 groups: `g = 6 * label + dialect_idx`, with `label` 0 = benign / 1 = toxic and
`dialect_idx` indexing `["SAE", "AAVE", "ChcE", "CollSgE", "IndE", "JamE"]`. Both
`prepare_data.py` and `train_group_dro.py` define this identically; changing one requires changing
the other.

## Training configuration

Values below are the defaults in `train_group_dro.py`. Read them from the source, not from any
saved `training_args` dump.

| Setting | Value |
|---|---|
| Backbone | `distilbert/distilbert-base-uncased` |
| Max sequence length | 512 |
| Epochs | 3 |
| Batch size | 16 (no gradient accumulation) |
| Optimizer | AdamW (β₁ 0.9, β₂ 0.999, ε 1e-8) |
| Learning rate | 5e-5, linear schedule |
| Warmup ratio | 0.19 (converted to `warmup_steps` on transformers releases that dropped `warmup_ratio`) |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Precision | fp16 when CUDA is available |
| Seeds | 0–9 (mean ± std reported) |

GroupDRO (`--robust`), from `group_DRO/loss.py` (Sagawa et al., MIT):

| Setting | Value | Flag |
|---|---|---|
| Adversarial step size η | 0.01 | `--robust_step_size` |
| Group-loss EMA γ | 0.1 | `--gamma` |
| Generalization adjustment C | 0.0 | `--generalization_adjustment` |
| Group-balanced sampling | on | disable with `--no_group_balance` |
| Model selection | worst-group accuracy | `--metric_for_best_model` |

## Metrics

Per dialect *d*, against SAE:

```
ΔTPR(d) = TPR(d) − TPR(SAE)      min_dTPR  = worst under-detection
ΔFPR(d) = FPR(d) − FPR(SAE)      max_dFPR  = worst over-censorship
```

Computed in `build_compute_metrics` during evaluation, and re-derived from raw TPR/FPR by
`ensure_delta_metrics` in `gather_all_results.py` for runs predating that change. Both use the same
formula.

## Outputs

`results/<run>/test_metrics.json` per run; `gather_all_results.py` aggregates them into
`results/all_experiments_{wide,long,pretty}.csv`, and `make_table8_markdown.py` renders Table 8.
The released tables are `results/table8_v2*.md`.
