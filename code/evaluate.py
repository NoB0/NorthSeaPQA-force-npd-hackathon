"""Evaluation script."""

import argparse
import collections
import json
import re
import statistics
import string
from typing import Any, Dict

from sacrebleu.metrics import BLEU

OPTS = None


def parse_args() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        A namespace object containing the arguments.
    """
    parser = argparse.ArgumentParser(prog="evaluate.py")
    parser.add_argument(
        "data_file",
        type=str,
        required=True,
        help="Input data JSON file.",
    )
    parser.add_argument(
        "pred_file",
        type=str,
        required=True,
        help="Model predictions.",
    )
    parser.add_argument(
        "out_file",
        type=str,
        required=True,
        help="Write accuracy metrics to file. Default is stdout.",
    )
    return parser.parse_args()


def normalize_answer(answer: str) -> str:
    """Normalizes answer.

    Args:
        answer: Answer.

    Returns:
        Normalized answer.
    """
    answer = answer.lower()
    # Remove punctuation
    answer = "".join(char for char in answer if char not in string.punctuation)
    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    # Remove extra whitespace
    answer = re.sub(r"\s\s+", " ", answer)
    return answer


def compute_exact_match(prediction: str, ground_truth: str) -> int:
    """Computes exact match between prediction and ground truth.

    Args:
        prediction: Prediction.
        ground_truth: Ground truth.

    Returns:
        1 if the prediction exactly matches the ground truth, 0 otherwise.
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """Computes F1 score between prediction and ground truth.

    Args:
        prediction: Prediction.
        ground_truth: Ground truth.

    Returns:
        F1 score.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(
        ground_truth_tokens
    )
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_bleu_score(prediction: str, ground_truth: str) -> float:
    """Computes BLEU score between prediction and ground truth.

    Args:
        prediction: Prediction.
        ground_truth: Ground truth.

    Returns:
        BLEU score.
    """
    bleu = BLEU()
    prediction_text = normalize_answer(prediction)
    ground_truth_text = normalize_answer(ground_truth)
    return bleu.corpus_score(
        prediction_text, [[ground_truth_text]], tokenize="none"
    ).score


def get_scores(
    dataset: Dict[str, Any], predictions: Dict[str, Any]
) -> Dict[str, Any]:
    """Computes scores for a set of metrics.

    Args:
        dataset: Dataset.
        predictions: Predictions.

    Returns:
        A dictionary containing the scores.
    """
    exact_matches = {}
    f1_scores = {}
    bleu_scores = {}
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                qid = qa["id"]
                if qid not in predictions:
                    print("Missing prediction for %s" % qid)
                    continue
                predicted_answer_span = predictions[qid].get("span", "")
                predicted_answer_text = predictions[qid].get("text", "")
                gold_answer_span = qa.get("answer", {}).get("span", "")
                gold_answer_text = qa.get("answer", {}).get("text", "")
                exact_matches[qid] = compute_exact_match(
                    predicted_answer_span, gold_answer_span
                )
                f1_scores[qid] = compute_f1_score(
                    predicted_answer_span, gold_answer_span
                )
                bleu_scores[qid] = compute_bleu_score(
                    predicted_answer_text, gold_answer_text
                )

    assert len(exact_matches) == len(f1_scores)
    total = len(exact_matches)

    return {
        "exact_match": 100.0 * sum(exact_matches.values()) / total,
        "f1": 100.0 * sum(f1_scores.values()) / total,
        "bleu": statistics.mean(list(bleu_scores.values())),
    }


def main(args: argparse.Namespace) -> None:
    with open(args.data_file) as f:
        dataset_json = json.load(f)
        dataset = dataset_json["data"]
    with open(args.pred_file) as f:
        predictions = json.load(f)

    out_eval = get_scores(dataset, predictions)

    if args.out_file:
        with open(args.out_file, "w") as f:
            json.dump(out_eval, f)
    else:
        print(json.dumps(out_eval, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)
