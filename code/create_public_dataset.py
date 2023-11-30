"""Scripts to create dataset for public release."""

import argparse
import json
import logging
from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
VERSION = "1.0"


def parse_args() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        A namespace object containing the arguments.
    """
    parser = argparse.ArgumentParser(prog="data_format.py")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        default=DATA_DIR,
        help="Directory containing the data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default=DATA_DIR,
        help="Directory to save the data.",
    )

    return parser.parse_args(args)


def load_data(data_dir: str) -> pd.DataFrame:
    """Loads the data from the data directory.

    The data is expected to be in a csv with the at least the following columns:
    type, doc_id, passage_id, passage, question, answer_span, and answer_text.

    Args:
        data_dir: The directory containing the data.

    Returns:
        A pandas dataframe containing the data.
    """
    logging.info("Loading data...")
    data = pd.read_csv(f"{data_dir}/data.csv")
    return data


def generate_qid(doc_id: str, passage_id: str, i: int) -> str:
    """Generates a unique id for a question.

    Args:
        doc_id: Document id.
        passage_id: Passage id.
        qa_id: Question id.

    Returns:
        A unique id for the question.
    """
    return f"{doc_id}_{passage_id}_{i}"


def format_document(passages: pd.DataFrame, doc_id: str) -> Dict[str, Any]:
    """Formats data for a single document.

    Args:
        passages: Dataframe containing the passages for a document.
        doc_id: Document id.

    Returns:
        A dictionary containing the passages and QAs pairs for a document.
    """
    formatted_document = {"document": doc_id}
    formatted_passages = []

    passage_qas = passages.groupby("passage_id")
    for passage_id, passage in passage_qas:
        logging.info(f"Formatting passage {passage_id}...")
        formatted_passage = {"context": passage["passage"].iloc[0], "qas": []}

        for i, qa in passage.iterrows():
            answers = (
                [
                    {
                        "text": qa["answer_text"],
                        "answer_start": qa["answer_span"],
                    }
                ]
                if not pd.isna(qa["answer_span"])
                else []
            )
            formatted_qa = {
                "question": qa["question"],
                "id": generate_qid(doc_id, passage_id, i),
                "answers": answers,
                "is_impossible": True if len(answers) == 0 else False,
            }
            formatted_passage["qas"].append(formatted_qa)

        formatted_passages.append(formatted_passage)

    formatted_document["paragraphs"] = formatted_passages
    return formatted_document


def format_dataset(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Formats the data for public release.

    Args:
        data: Dataframe containing the data.

    Returns:
        A list of dictionaries containing the data for each document.
    """
    logging.info("Formatting data...")
    formatted_data = []

    documents = data.groupby("doc_id")
    for doc_id, passages in documents:
        formatted_data.append(format_document(passages, doc_id))

    return {"version": VERSION, "data": formatted_data}


def main(args) -> None:
    """Formats the data for public release."""
    output_dir = args.output_dir

    data = load_data(args.data_dir)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    test, val = train_test_split(test, test_size=0.5, random_state=42)

    for name, split in zip(["train", "val", "test"], [train, val, test]):
        formatted_data = format_dataset(split)
        with open(f"{output_dir}/{name}.json", "w") as f:
            json.dump(formatted_data, f, indent=4)


if __name__ == "__main__":
    args = parse_args()

    main(args)