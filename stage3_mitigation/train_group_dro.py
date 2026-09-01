"""
Group DRO fine-tuning for NSFW text classifier.
Adapts the LossComputer from https://github.com/kohpangwei/group_DRO
to the HuggingFace Trainer API.
"""
import inspect
import math
import os
import sys
import pathlib
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
import torch.nn.functional as F


# Make `common/` and the sibling `group_DRO/` importable regardless of CWD.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from group_DRO.loss import LossComputer  # noqa: E402
from common.common_utils import DIALECTS_WITH_SAE as DIALECTS  # noqa: E402
N_DIALECTS = len(DIALECTS)


# =========================================================
# Dataset
# =========================================================
class NSFWGroupDataset(Dataset):
    """CSV columns: text, label, group  (extra columns ignored)."""
    def __init__(self, csv_path, tokenizer, max_length=512, n_groups=None):
        df = pd.read_csv(csv_path)
        self.texts  = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.groups = df["group"].astype(int).tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

        self.n_groups = n_groups or (max(self.groups) + 1)
        self._group_counts = torch.zeros(self.n_groups, dtype=torch.float)
        for g in self.groups:
            self._group_counts[g] += 1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
            "group_idx":      torch.tensor(self.groups[idx], dtype=torch.long),
        }

    # LossComputer interface
    def group_counts(self):  return self._group_counts
    def group_str(self, idx): return f"g{idx:02d}"


# =========================================================
# Metrics:  worst-group / per-dialect / SAE-gap
# =========================================================
def build_compute_metrics(val_dataset):
    groups_all = np.array(val_dataset.groups)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds   = np.argmax(logits, axis=-1)
        correct = (preds == labels).astype(np.float32)

        m = {"accuracy": float(correct.mean())}

        # per-group acc
        per_group = []
        for g in range(val_dataset.n_groups):
            mask = (groups_all == g)
            if mask.sum() == 0:
                continue
            acc_g = float(correct[mask].mean())
            m[f"acc_g{g:02d}"] = acc_g
            per_group.append(acc_g)
        m["worst_group_acc"] = float(np.min(per_group))
        m["mean_group_acc"]  = float(np.mean(per_group))

        # per-dialect (averaged over both labels)
        for d_idx, d_name in enumerate(DIALECTS):
            mask = np.isin(groups_all, [d_idx, N_DIALECTS + d_idx])
            if mask.sum() > 0:
                m[f"acc_{d_name}"] = float(correct[mask].mean())

        labels_arr = np.asarray(labels)
        for d_idx, d_name in enumerate(DIALECTS):
            d_mask = np.isin(groups_all, [d_idx, N_DIALECTS + d_idx])
            if d_mask.sum() == 0:
                continue
            y    = labels_arr[d_mask]
            yhat = preds[d_mask]
            pos  = (y == 1)
            neg  = (y == 0)
            if pos.sum() > 0:
                m[f"TPR_{d_name}"] = float(yhat[pos].mean())
            if neg.sum() > 0:
                m[f"FPR_{d_name}"] = float(yhat[neg].mean())

        # fairness summary
        tprs = [m[f"TPR_{d}"] for d in DIALECTS if f"TPR_{d}" in m]
        fprs = [m[f"FPR_{d}"] for d in DIALECTS if f"FPR_{d}" in m]
        if tprs: m["TPR_spread"] = float(max(tprs) - min(tprs))
        if fprs: m["FPR_spread"] = float(max(fprs) - min(fprs))

        if "TPR_SAE" in m and "FPR_SAE" in m:
            dtprs, dfprs = [], []
            for d in DIALECTS:
                if d == "SAE":
                    continue
                if f"TPR_{d}" in m:
                    m[f"dTPR_{d}"] = m[f"TPR_{d}"] - m["TPR_SAE"]; dtprs.append(m[f"dTPR_{d}"])
                if f"FPR_{d}" in m:
                    m[f"dFPR_{d}"] = m[f"FPR_{d}"] - m["FPR_SAE"]; dfprs.append(m[f"dFPR_{d}"])
            if dtprs:
                m["min_dTPR"]      = float(min(dtprs))               # worst under-detection
                m["mean_abs_dTPR"] = float(np.mean(np.abs(dtprs)))
            if dfprs:
                m["max_dFPR"]      = float(max(dfprs))               # worst over-censorship
                m["mean_abs_dFPR"] = float(np.mean(np.abs(dfprs)))

        # SAE (dialect_idx==0) vs non-SAE
        sae = np.isin(groups_all, [0, N_DIALECTS])
        m["acc_SAE"]        = float(correct[sae].mean())
        m["acc_nonSAE"]     = float(correct[~sae].mean())
        m["gap_SAE_nonSAE"] = m["acc_SAE"] - m["acc_nonSAE"]
        return m
    return compute_metrics


# =========================================================
# Custom Trainer:  Group DRO loss + group-balanced sampling
# =========================================================
class GroupDROTrainer(Trainer):
    def __init__(self, *args, loss_computer=None,
                 group_balanced_sampling=True, n_groups=None, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_computer is None:
            raise ValueError("loss_computer is required.")
        self.loss_computer = loss_computer
        self.group_balanced_sampling = group_balanced_sampling
        self.n_groups = n_groups or loss_computer.n_groups

        self._g_loss_sum = torch.zeros(self.n_groups)
        self._g_count    = torch.zeros(self.n_groups)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        group_idx = inputs.pop("group_idx", None)
        if group_idx is None:
            raise ValueError("`group_idx` missing — set remove_unused_columns=False.")

        outputs = model(**inputs)
        logits  = outputs.logits
        labels  = inputs["labels"]

        loss = self.loss_computer.loss(
            logits, labels, group_idx.to(logits.device),
            is_training=model.training,
        )

        if model.training:
            with torch.no_grad():
                per_sample = F.cross_entropy(logits, labels, reduction="none")
                g_cpu = group_idx.detach().cpu()
                p_cpu = per_sample.detach().cpu()
                for g in range(self.n_groups):
                    mask = (g_cpu == g)
                    n = mask.sum().item()
                    if n > 0:
                        self._g_loss_sum[g] += p_cpu[mask].sum().item()
                        self._g_count[g]    += n

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        # per-group avg loss since last log
        if self._g_count.sum() > 0:
            for g in range(self.n_groups):
                if self._g_count[g] > 0:
                    logs[f"loss_g{g:02d}"] = (
                        self._g_loss_sum[g] / self._g_count[g]
                    ).item()

            # also surface worst-group / mean / max-min spread for at-a-glance reading
            valid = self._g_count > 0
            avg = (self._g_loss_sum[valid] / self._g_count[valid])
            logs["loss_g_worst"] = avg.max().item()
            logs["loss_g_mean"]  = avg.mean().item()
            logs["loss_g_spread"] = (avg.max() - avg.min()).item()

            if getattr(self.loss_computer, "is_robust", False) and \
               hasattr(self.loss_computer, "adv_probs"):
                q = self.loss_computer.adv_probs.detach().cpu()
                for g in range(self.n_groups):
                    logs[f"q_g{g:02d}"] = float(q[g])

            # reset accumulators
            self._g_loss_sum.zero_()
            self._g_count.zero_()

        super().log(logs, *args, **kwargs)

    def _get_train_sampler(self, *args, **kwargs):
        if not self.group_balanced_sampling:
            return super()._get_train_sampler(*args, **kwargs)
        ds      = self.train_dataset
        counts  = ds._group_counts.numpy()
        groups  = np.array(ds.groups)
        weights = 1.0 / counts[groups]
        return WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(ds),
            replacement=True,
        )


# =========================================================
# Main
# =========================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="./data/train.csv",
                   help="long-format train CSV (used in full; no inner split)")
    p.add_argument("--test_csv",  default="./data/val.csv",
                   help="held-out test set; used ONCE after training is done")
    p.add_argument("--inner_val_ratio", type=float, default=0.1,
                   help="inner-split fraction; only used when --inner_split is set")
    p.add_argument("--inner_split", action="store_true",
                   help="(ablation) reproduce original behavior: train on a 90%% "
                        "inner split of train_csv instead of the full set")
    p.add_argument("--n_groups",  type=int, default=12)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--robust", action="store_true",
                   help="True=Group DRO, False=ERM")
    p.add_argument("--robust_step_size", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--generalization_adjustment", default="0.0")
    p.add_argument("--no_group_balance", action="store_true",
                   help="disable group-balanced sampling (for ERM ablation)")
    p.add_argument("--metric_for_best_model", default="worst_group_acc")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    # Fine-tune exactly like michellejieli/NSFW_text_classifier to replicate their setup.
    model_name = "distilbert/distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "SFW", 1: "NSFW"},
        label2id={"SFW": 0, "NSFW": 1},
    )


    print(f"[model] num_labels = {model.config.num_labels}")
    print(f"[model] id2label   = {model.config.id2label}")
    print(f"[data]  expected:  label 0 = safe (benign),  label 1 = nsfw (toxic)")
    assert model.config.num_labels == 2

    # =====================================================
    # 2. Datasets
    #    Train on the FULL train_csv. val.csv is the held-out test set
    #    (test_ds below). We select no best checkpoint (final-epoch model,
    #    eval is disabled), so we do NOT carve an inner validation split out
    #    of train_csv: doing so only shrank the already tiny training set
    #    (~10% of SAE) and perturbed the intended SAE ratio, while leaving
    #    minority (dialect) groups with zero inner-val samples. Paraphrase
    #    leakage is already prevented by the train.csv / val.csv split.
    # =====================================================
    full_df = pd.read_csv(args.train_csv)

    if args.inner_split:
        # (Ablation only) Reproduce the original behavior: carve an inner 9:1
        # split out of train_csv and train on the 90%. inner_val is unused
        # (eval is off); this exists only to test whether the sampler-vs-loss
        # conclusion depends on that discarded 10%.
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=args.inner_val_ratio,
                                random_state=args.seed)
        tr_idx, _ = next(gss.split(full_df, groups=full_df["sample_id"]))
        counts_df = full_df.iloc[tr_idx].reset_index(drop=True)
        os.makedirs("./data/_tmp", exist_ok=True)
        train_src = f"./data/_tmp/inner_train_s{args.seed}.csv"
        counts_df.to_csv(train_src, index=False)
        src_label = "inner_train (90% of train_csv)"
    else:
        counts_df = full_df
        train_src = args.train_csv
        src_label = "full train_csv"

    print(f"\n[train] per-group counts ({src_label}):")
    for g in range(args.n_groups):
        n_tr = (counts_df["group"] == g).sum()
        print(f"  g={g:2d}   train={n_tr:6d}")

    train_ds = NSFWGroupDataset(train_src, tokenizer,
                                args.max_length, args.n_groups)
    test_ds  = NSFWGroupDataset(args.test_csv,  tokenizer,
                                args.max_length, args.n_groups)
    # Eval is disabled during training; the trainer's eval slot only exists to
    # build the metrics closure, which is rebound to test_ds before the final
    # evaluation. Point it at test_ds so no separate val set is needed.
    val_ds   = test_ds

    print(f"\n[size] train={len(train_ds):,}   held-out test={len(test_ds):,}")

    # =====================================================
    # 4. Generalization adjustment vector C
    # =====================================================
    adj = [float(c) for c in args.generalization_adjustment.split(",")]
    adj = np.array(adj * train_ds.n_groups) if len(adj) == 1 else np.array(adj)
    assert len(adj) == train_ds.n_groups

    # =====================================================
    # 5. TrainingArguments
    # =====================================================
    # warmup_ratio is deprecated in newer transformers; fallback to equivalent warmup_steps (0.19 matches original NSFW-T).
    WARMUP_RATIO = 0.19
    _ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in _ta_params:
        _warmup_kw = {"warmup_ratio": WARMUP_RATIO}
    elif "warmup_steps" in _ta_params:
        _world = max(1, torch.cuda.device_count())
        _eff_batch = 16 * 1 * _world          # per_device_batch x grad_accum x world
        _steps_per_epoch = math.ceil(len(train_ds) / _eff_batch)
        _total_steps = _steps_per_epoch * 3   # num_train_epochs
        _warmup_kw = {"warmup_steps": round(WARMUP_RATIO * _total_steps)}
        print(f"[compat] warmup_ratio unsupported fallback to warmup_steps={_warmup_kw['warmup_steps']} "
              f"(= {WARMUP_RATIO} x {_total_steps} steps)")
    else:
        raise RuntimeError(
            "transformers version incompatible; must support warmup_ratio or warmup_steps. Use transformers==5.12.1")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True, 
        do_eval=False,  # Original setup did not evaluate during training.
        eval_strategy="no",  # Original setup did not evaluate.
        save_strategy="no",  # Save disk space as only the final model is used.
        save_total_limit=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,                
        weight_decay=0.01,
        adam_beta1=0.9, 
        adam_beta2=0.999, 
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        num_train_epochs=3,                
        lr_scheduler_type="linear",
        **_warmup_kw,                      # Set dynamically based on version compatibility (equivalent to 0.19 ratio)
        logging_strategy="steps", 
        logging_steps=10,  # Match original setup of 10 steps.
        seed=args.seed,                           
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        optim="adamw_torch",                
        report_to=[],  # Avoid crash due to missing tensorboard; we only need test_metrics.json.
        run_name=os.path.basename(args.output_dir.rstrip("/")),
        load_best_model_at_end=False,  # Force use of the final model after 3 epochs to replicate original setup.
        metric_for_best_model=None,
        greater_is_better=None,
    )


    # =====================================================
    # 6. LossComputer + Trainer
    # =====================================================
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss_computer = LossComputer(
        criterion=criterion,
        is_robust=args.robust,
        dataset=train_ds,
        alpha=1.0, gamma=args.gamma,
        adj=adj, step_size=args.robust_step_size,
        normalize_loss=False, btl=False, min_var_weight=0,
    )

    trainer = GroupDROTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,                     # = test_ds; eval is off during training
        loss_computer=loss_computer,
        group_balanced_sampling=(not args.no_group_balance),
        compute_metrics=build_compute_metrics(val_ds),
    )

    # =====================================================
    # 7. Train
    # =====================================================
    trainer.train()
    # Saving is disabled to save space as it is unused.

    # =====================================================
    # 8. Held-out TEST evaluation (final, reported number)
    # =====================================================
    print("\n" + "=" * 60)
    print("HELD-OUT TEST EVALUATION  (final reported numbers)")
    print("=" * 60)
    trainer.compute_metrics = build_compute_metrics(test_ds)   # rebind metrics fn
    test_metrics = trainer.evaluate(eval_dataset=test_ds,
                                    metric_key_prefix="test")
    for k in sorted(test_metrics.keys()):
        v = test_metrics[k]
        if isinstance(v, float):
            print(f"  {k:30s} = {v:.4f}")
        else:
            print(f"  {k:30s} = {v}")

    out_json = os.path.join(args.output_dir, "test_metrics.json")
    pd.Series(test_metrics).to_json(out_json, indent=2)
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
