"""
Zero-shot evaluation of an off-the-shelf classifier on the test set.
Used for Exp1 (NSFW-T baseline) — no training, just inference.
"""
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from train_group_dro import NSFWGroupDataset, build_compute_metrics  # reuse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="michellejieli/NSFW_text_classifier")
    p.add_argument("--test_csv",   default="./data/val.csv")
    p.add_argument("--n_groups",   type=int, default=12)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--output_dir", default="./results/exp1_nsfwt_zeroshot")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    print(f"[model] {args.model_name}")
    print(f"[model] id2label = {model.config.id2label}")

    test_ds = NSFWGroupDataset(args.test_csv, tokenizer, args.max_length, args.n_groups)

    # Trainer just for the eval pipeline (no training)
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=64,
        do_train=False, do_eval=True,
        remove_unused_columns=False,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model, args=eval_args, 
        compute_metrics=build_compute_metrics(test_ds),
    )

    metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    print("\n=== NSFW-T zero-shot on test set ===")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            print(f"  {k:30s} = {v:.4f}")

    pd.Series(metrics).to_json(f"{args.output_dir}/test_metrics.json", indent=2)


if __name__ == "__main__":
    main()