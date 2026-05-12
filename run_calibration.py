"""
Run manual evaluation + calibration analysis on the fine-tuned Lab 7A model.

Outputs:
- figures/reliability-diagram.png
- calibration_metrics.json

This script intentionally avoids Trainer.predict and uses the manual
PyTorch inference loop implemented in manual_eval.py.
"""

import json
import os

# Must be set before any other import
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ── Load data FIRST, before torch/transformers touch the allocator ──────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lab import get_data_path

print("Loading dataset...")
data_path = get_data_path()
df = pd.read_csv(data_path)
_, test_df = train_test_split(df, test_size=0.2, random_state=42)
test_texts = test_df["text"].tolist()
y_true     = test_df["label"].to_numpy()


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from calibration import (
    expected_calibration_error,
    plot_reliability,
    reliability_diagram,
)
from manual_eval import (
    compute_classification_report_from_arrays,
    manual_predict,
)


def main():
    model_dir = "model"
    os.makedirs("figures", exist_ok=True)

    # Load fine-tuned model + tokenizer
    print("Loading fine-tuned model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    id2label  = model.config.id2label

    # Manual inference
    print("Running manual inference...")
    y_pred, probs = manual_predict(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        batch_size=8,
    )

    # Classification metrics
    print("Computing classification metrics...")
    report = compute_classification_report_from_arrays(
        y_true=y_true,
        y_pred=y_pred,
    )

    # Calibration metrics
    print("Computing calibration metrics...")
    centers, accs, counts = reliability_diagram(
        probs=probs,
        y_true=y_true,
        n_bins=10,
    )
    ece = expected_calibration_error(
        probs=probs,
        y_true=y_true,
        n_bins=10,
    )

    # Save reliability diagram
    plot_path = "figures/reliability-diagram.png"
    print(f"Saving reliability diagram to {plot_path}...")
    plot_reliability(
        centers=centers,
        accs=accs,
        counts=counts,
        output_path=plot_path,
    )

    # Save metrics JSON
    results = {
        "accuracy"         : float(report["accuracy"]),
        "macro_f1"         : float(report["macro_f1"]),
        "ece"              : float(ece),
        "label_names"      : id2label,
        "per_class"        : report["per_class"],
        "bucket_centers"   : centers.tolist(),
        "bucket_accuracies": accs.tolist(),
        "bucket_counts"    : counts.tolist(),
    }
    with open("calibration_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Console summary
    print("\n=== Manual Evaluation Results ===")
    print(f"Accuracy : {report['accuracy']:.4f}")
    print(f"Macro-F1 : {report['macro_f1']:.4f}")
    print(f"ECE      : {ece:.4f}")

    print("\nPer-class metrics:")
    for label_idx, metrics in report["per_class"].items():
        label_name = id2label.get(label_idx, str(label_idx))
        print(
            f"  {label_name:<10} "
            f"P={metrics['precision']:.4f}  "
            f"R={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}"
        )

    print("\nReliability buckets:")
    for i in range(len(centers)):
        print(
            f"  Bin {i:02d}  "
            f"center={centers[i]:.2f}  "
            f"accuracy={accs[i]:.4f}  "
            f"count={counts[i]}"
        )

    print("\nSaved:")
    print(f"  - {plot_path}")
    print("  - calibration_metrics.json")


if __name__ == "__main__":
    main()