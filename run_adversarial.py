"""
Stretch Thursday — Adversarial Evaluation.

Load a fine-tuned classifier, run it against adversarial_set.csv, and write
results.csv. Read label names from model.config.id2label — do not hard-code.
"""

import os

import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    """
    Load model and tokenizer from a local path or HF Hub id.

    Defaults to local 'model' (your Lab 7A checkpoint). CI overrides via MODEL_PATH env.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer



def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Run the model on every row of adv_csv_path. Return a DataFrame with all
    original columns plus predicted_label, predicted_probability, correct.

    Read label names from model.config.id2label — do not hard-code class names.
    """
    df = pd.read_csv(adv_csv_path)
    
    predicted_labels = []
    predicted_probabilities = []
    corrects = []
    
    model.eval()

    for _, row in df.iterrows():
        inputs = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_index = torch.argmax(probabilities, dim=1).item()
            predicted_label = model.config.id2label[predicted_index]
            predicted_probability = probabilities[0][predicted_index].item()
            correct = predicted_label == row["expected_label"]

        predicted_labels.append(predicted_label)
        predicted_probabilities.append(predicted_probability)
        corrects.append(correct)
    df["predicted_label"] = predicted_labels
    df["predicted_probability"] = predicted_probabilities
    df["correct"] = corrects
    return df



def plot_error_counts(results_csv_path: str, output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(results_csv_path)

    error_counts = df.groupby("hypothesis_category")["correct"].apply(lambda x: (~x).sum())

    plt.figure()
    error_counts.sort_values().plot(kind="bar", color="red")

    plt.title("Number of Errors per Hypothesis Category")
    plt.ylabel("Error Count")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "error_counts.png"))
    plt.close()


def plot_heatmap(results_csv_path: str, output_dir: str = "figures"):

    df = pd.read_csv(results_csv_path)

    heatmap_data = df.groupby("hypothesis_category")["correct"].mean().to_frame()

    plt.figure(figsize=(6, 4))
    sns.heatmap(heatmap_data, annot=True, cmap="RdYlGn", vmin=0, vmax=1)

    plt.title("Adversarial Accuracy Heatmap")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "heatmap.png"))
    plt.close()

def main() -> None:
    """Orchestrate; write results.csv."""
    model_path = os.environ.get("MODEL_PATH", "model")
    adv_csv = os.environ.get("ADVERSARIAL_CSV", "adversarial_set.csv")
    out_csv = os.environ.get("RESULTS_CSV", "results.csv")

    model, tokenizer = load_model(model_path)
    df = run_against_set(adv_csv, model, tokenizer)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")

    # --- create figures directory ---
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    # --- plots ---
    plot_error_counts(out_csv, output_dir)
    plot_heatmap(out_csv, output_dir)


if __name__ == "__main__":
    main()
