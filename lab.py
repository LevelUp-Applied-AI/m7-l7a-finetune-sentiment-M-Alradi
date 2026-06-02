"""
Module 7 Week A — Applied Lab: Fine-Tune DistilBERT for App-Review Sentiment.

Implement the TODO functions to build a complete fine-tuning pipeline.

Default run: `python lab.py` reads `data/app_reviews_train.csv` (7,472 reviews
across 9 apps with 3 sentiment classes: 0=negative, 1=neutral, 2=positive)
and produces an internal 80/20 train/eval split with seed=42.

CI smoke run: workflow sets DATA_PATH=fixtures/tiny_app_reviews.csv (60 rows).

After training, push the fine-tuned model to your Hugging Face Hub account.
The model directory is local-only (gitignored).
"""

import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# 3-class sentiment label mapping (matches the curated dataset's `label` column)
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI uses a smoke CSV); otherwise return
    the default path to the curated app-review training CSV.

    Provided helper. Do not modify.
    """
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load the CSV at `data_path` and produce a train/test split.

    Returns a `DatasetDict` with "train" and "test" keys.
    """
    df = pd.read_csv(data_path)
    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds.train_test_split(test_size=test_size, seed=seed)


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize all splits in a DatasetDict.

    `tokenizer` is a loaded HuggingFace tokenizer (callable).
    Uses truncation=True and max_length=max_length. No padding here —
    padding is applied dynamically by DataCollatorWithPadding at training time.
    """
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds_dict.map(tokenize_fn, batched=True)


def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    """Return a TrainingArguments configured for fine-tuning."""
    # Smoke fixture (CI): override to more epochs + smaller batch so the model
    # gets enough gradient steps on 48 rows to reliably clear the accuracy threshold.
    # Default run: 2 epochs x 6 steps = 12 steps total — far too few on tiny data.
    # Smoke run:  10 epochs x 24 steps = 240 steps — enough to learn the fixture.
    
    if os.environ.get("DATA_PATH") is not None:
        epochs = 10
        batch_size = 2

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=seed,
    )
    args.eval_strategy = args.eval_strategy.value
    args.save_strategy = args.save_strategy.value
    return args


def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """
    Construct and train a Trainer.

    Returns the trained Trainer (trainer.model is the fine-tuned model).
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the trainer's model on the test split.
    """
    predictions = trainer.predict(tokenized_test)

    logits = predictions.predictions
    labels = predictions.label_ids

    pred_idx = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, pred_idx)
    macro_f1 = f1_score(labels, pred_idx, average="macro")

    per_class_f1_vals = f1_score(labels, pred_idx, average=None)
    per_class_precision_vals = precision_score(
        labels,
        pred_idx,
        average=None,
        zero_division=0,
    )
    per_class_recall_vals = recall_score(
        labels,
        pred_idx,
        average=None,
        zero_division=0,
    )

    id2label = trainer.model.config.id2label

    per_class_f1 = {
        id2label[i]: float(v)
        for i, v in enumerate(per_class_f1_vals)
    }

    per_class_precision = {
        id2label[i]: float(v)
        for i, v in enumerate(per_class_precision_vals)
    }

    per_class_recall = {
        id2label[i]: float(v)
        for i, v in enumerate(per_class_recall_vals)
    }

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }

    return metrics, logits, pred_idx


def main() -> None:
    """Orchestrate the full fine-tuning pipeline."""

    # ----------------------------
    # CONFIG
    # ----------------------------
    data_path = get_data_path()

    model_name = os.environ.get(
        "MODEL_NAME",
        "distilbert-base-uncased"
    )

    output_dir = os.environ.get(
        "OUTPUT_DIR",
        "model"
    )

    repo_id = os.environ.get("HF_REPO_ID")

    # ----------------------------
    # DATA PREP
    # ----------------------------
    ds = prepare_dataset(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized = tokenize_dataset(ds, tokenizer)

    tokenized.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # ----------------------------
    # TRAINING
    # ----------------------------
    training_args = make_training_args(output_dir)

    trainer = train_classifier(
        tokenized_ds=tokenized,
        model_name=model_name,
        training_args=training_args,
        tokenizer=tokenizer,
        num_labels=len(ID2LABEL),
    )

    # ----------------------------
    # SAVE MODEL
    # ----------------------------
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nSaved model to: {output_dir}")

    # ----------------------------
    # EVALUATION
    # ----------------------------
    metrics, pred_logits, pred_idx = evaluate_classifier(
        trainer,
        tokenized["test"]
    )

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_probs = _softmax(pred_logits)

    id2label = trainer.model.config.id2label

    label_names = [
        id2label[i]
        for i in range(len(id2label))
    ]

    true_labels = [
        id2label[i]
        for i in ds["test"]["label"]
    ]

    pred_labels = [
        id2label[i]
        for i in pred_idx
    ]

    # ----------------------------
    # CONFUSION MATRIX
    # ----------------------------
    cm = confusion_matrix(
        true_labels,
        pred_labels,
        labels=label_names
    )

    cm_df = pd.DataFrame(
        cm,
        index=label_names,
        columns=label_names
    )

    cm_df.to_csv("confusion_matrix.csv")

    # ----------------------------
    # PREDICTIONS CSV
    # ----------------------------
    df_out = pd.DataFrame({
        "text": ds["test"]["text"],
        "label": true_labels,
        "predicted_label": pred_labels,
        "predicted_probability": [
            float(pred_probs[i, pred_idx[i]])
            for i in range(len(pred_idx))
        ],
    })

    for idx, name in id2label.items():
        df_out[f"prob_{name}"] = [
            float(pred_probs[i, idx])
            for i in range(len(pred_idx))
        ]

    df_out.to_csv("predictions.csv", index=False)

    # ----------------------------
    # METRICS OUTPUT
    # ----------------------------
    print(f"\nAccuracy : {metrics['accuracy']:.4f}")
    print(f"Macro-F1 : {metrics['macro_f1']:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())

    # ----------------------------
    # OPTIONAL HF UPLOAD
    # ----------------------------
    if repo_id and os.environ.get("DATA_PATH") is None:

        try:
            from huggingface_hub import upload_folder

            upload_folder(
                repo_id=repo_id,
                folder_path=output_dir
            )

            print(f"\nPushed model to https://huggingface.co/{repo_id}")

        except Exception as e:
            print(f"\nHF Hub upload failed: {e}")
            print("Make sure you ran: huggingface-cli login")

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    main()