"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list, batch_size: int = 8):
    """
    Run manual PyTorch inference over a list of texts.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    model.eval()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize with padding so all sequences in the batch are the same length
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )

        # Move tensors to whatever device the model is on
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits  # (batch, num_classes)

        # Softmax over class dimension -> probabilities
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)   # (N, num_classes)
    preds = np.argmax(probs, axis=1)            # (N,)
    return preds, probs


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.

    Returns:
      {
        "accuracy": float,
        "macro_f1": float,
        "per_class": {label_index: {"precision": ..., "recall": ..., "f1": ...}, ...},
      }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    per_class = {}

    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        # Guard divide-by-zero: if the denominator is 0, score is 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class[int(c)] = {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
        }

    accuracy = float(np.sum(y_pred == y_true) / len(y_true))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    return {
        "accuracy":  accuracy,
        "macro_f1":  macro_f1,
        "per_class": per_class,
    }