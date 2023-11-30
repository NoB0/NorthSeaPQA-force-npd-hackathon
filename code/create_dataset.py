"""Scripts to create data for public release."""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
DEFAULT_OPENAI_MODEL = "GPT35Turbo_DEPLOYMENT"
try:
    # Check if we are on github
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    logging.info("Loaded API key from environment")
except KeyError:
    # Apparently we are local, so lets load the key
    load_dotenv(".env.secret")
    logging.info("Loaded API key from .env.secret")


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
        "--model",
        type=str,
        required=True,
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model to use.",
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

    Args:
        data_dir: The directory containing the data.

    Returns:
        A pandas dataframe containing the data.
    """
    data = pd.read_csv(f"{data_dir}/data.csv")

    return data


def collect_qas(
    passage: str, openai_client: AzureOpenAI, model: str
) -> List[Dict[str, str]]:
    """Collects question and answers pairs for a given passage.

    Args:
        passage: A passage.
        openai_client: The OpenAI API client.
        model: The OpenAI model to use for the creation of QAs pairs.

    Returns:
        A list of question and answer pairs.
    """
    qas = []
    # TODO: Create a prompt for the creation of QAs pairs.
    prompt = f""
    output = openai_client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    # TODO: Format output to desired format
    return qas


def format(passages: List[str], model: str) -> Dict[str, Any]:
    """Formats the passages into a dictionary.

    Args:
        passages: A list of passages.
        model: The OpenAI model to use for the creation of QAs pairs.

    Returns:
        A dictionary containing the passages.
    """
    # OpenAI API client
    openai_client = AzureOpenAI(
        api_version=os.environ["OPENAI_API_VERSION"],
        azure_endpoint=os.environ["OPENAI_API_BASE"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    formatted_passages = []
    for passage in passages:
        formatted_passage = {"context": passage}
        query_answers = collect_qas(passage, openai_client, model)
        formatted_passage["qas"] = query_answers
        formatted_passages.append(formatted_passage)
    return {"paragraphs": formatted_passages}


def main(args) -> None:
    """Formats the data for public release."""
    output_dir = args.output_dir

    data = None
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    test, val = train_test_split(test, test_size=0.5, random_state=42)

    for name, split in zip(["train", "val", "test"], [train, val, test]):
        with open(f"{output_dir}/{name}.json", "w") as f:
            json.dump(split, f, indent=4)


if __name__ == "__main__":
    args = parse_args()

    main(args)
