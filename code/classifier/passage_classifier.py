"""Train a BERT model to classify passages using HuggingFace."""

import argparse
import os
from typing import Tuple

import evaluate
import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import wandb

os.environ["WANDB_PROJECT"] = "force_doc_classification"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


MODEL_NAME = "bert-base-uncased"
DATA_PATH = "data/annotates_good_text.xlsm"
COLUMNS = ["_id", "cat", "content_scrubbed_light", "label"]
OUTPUT_DIR = "data/models"


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(prog="passage_classifier.py")
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help="Path to the data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save the model.",
    )
    return parser.parse_args()


def load_data(path: str) -> Tuple[DatasetDict, str]:
    """Loads data as a HuggingFace dataset.

    Args:
        path: Path to the data.

    Returns:
        A HuggingFace dataset, num_classes.
    """
    data = pd.read_excel(path)
    num_classes = data["cat"].nunique()
    classes = list(data["cat"].unique())
    data["label"] = data["cat"].astype("category").cat.codes

    data = data[COLUMNS]
    data = Dataset.from_pandas(
        data,
        features=Features(
            {
                "label": ClassLabel(num_classes=num_classes, names=classes),
                "content_scrubbed_light": Value("string"),
                "_id": Value("string"),
                "cat": Value("string"),
            }
        ),
    )
    data = data.rename_column("content_scrubbed_light", "text")
    data = data.train_test_split(test_size=0.2)
    test_valid = data["test"].train_test_split(test_size=0.5)
    ds = DatasetDict(
        {
            "train": data["train"],
            "test": test_valid["test"],
            "valid": test_valid["train"],
        }
    )
    return ds, num_classes


def main(args: argparse.Namespace) -> None:
    """Train a BERT model to classify passages using HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset, num_labels = load_data(args.data_path)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize, batched=True, batch_size=32)
    print(f"Dataset: {dataset['test'].features}")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        report_to="wandb",
        logging_steps=1,
        learning_rate=1e-4,
    )

    metrics = [evaluate.load(m) for m in ["f1", "precision", "recall"]]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            m.name: m.compute(
                predictions=predictions, references=labels, average="macro"
            )
            for m in metrics
        }

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.train()

    # Evaluate on the test set.
    test_metrics = trainer.evaluate(dataset["test"])
    print(f"Test metrics: {test_metrics}")

    # Save test predictions.
    outputs = trainer.predict(dataset["test"])
    predicted_labels = np.argmax(outputs[0], axis=1)
    predicted_labels = [
        dataset["test"].features["label"].int2str(int(p))
        for p in predicted_labels
    ]
    gold_labels = [
        dataset["test"].features["label"].int2str(int(p))
        for example in dataset["test"]
        for p in example["label"]
    ]
    dataset["test"] = dataset["test"].add_column(
        "predicted_label", predicted_labels
    )
    dataset["test"] = dataset["test"].add_column("gold_label", gold_labels)
    dataset["test"] = dataset["test"].remove_columns(
        ["input_ids", "token_type_ids", "attention_mask"]
    )
    dataset["test"].to_csv("data/test_predictions.csv")


if __name__ == "__main__":
    wandb.login()

    args = parse_args()
    main(args)
    wandb.finish()
