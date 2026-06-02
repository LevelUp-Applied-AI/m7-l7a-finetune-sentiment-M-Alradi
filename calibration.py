"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Binning convention: edges = np.linspace(0, 1, n_bins + 1).
    p falls in bin i if edges[i] <= p < edges[i+1], except the last bin which
    is inclusive on the right so p == 1.0 lands there.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    edges = np.linspace(0, 1, n_bins + 1)                    # (n_bins+1,)
    bucket_centers = (edges[:-1] + edges[1:]) / 2            # (n_bins,)

    # Confidence = max probability for each prediction; predicted class = argmax
    confidences = np.max(probs, axis=1)                       # (N,)
    pred_classes = np.argmax(probs, axis=1)                   # (N,)
    correct = (pred_classes == y_true).astype(float)          # (N,)

    bucket_accuracies = np.zeros(n_bins)
    bucket_counts     = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Last bin is inclusive on both sides to catch p == 1.0
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        bucket_counts[i] = int(np.sum(mask))
        bucket_accuracies[i] = float(np.mean(correct[mask])) if bucket_counts[i] > 0 else 0.0

    return bucket_centers, bucket_accuracies, bucket_counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    edges = np.linspace(0, 1, n_bins + 1)

    confidences  = np.max(probs, axis=1)
    pred_classes = np.argmax(probs, axis=1)
    correct = (pred_classes == y_true).astype(float)

    N   = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        count = int(np.sum(mask))
        if count == 0:
            continue

        bucket_accuracy   = float(np.mean(correct[mask]))
        bucket_confidence = float(np.mean(confidences[mask]))
        ece += (count / N) * abs(bucket_accuracy - bucket_confidence)

    return float(ece)


def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)