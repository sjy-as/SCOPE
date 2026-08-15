# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bz2
import argparse
import collections
import json
import os
import re
import string
from datetime import datetime

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    from openai import APIConnectionError, OpenAI, RateLimitError
except ImportError:
    APIConnectionError = RateLimitError = Exception
    OpenAI = None
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm

tokenizer = None


def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES


def attempt_api_call(client, model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None


def build_openai_client(api_key=None, base_url=None):
    """Build an OpenAI-compatible client, including DeepSeek's API endpoint."""
    if OpenAI is None:
        raise ImportError("Please install the openai package before running LLM evaluation.")
    kwargs = {}
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)


def parse_response(response: str):
    """
    Return a tuple of (explanation, score) from the response, 
    where score is 0 if the prediction is wrong, 1 if the prediction is correct.

    Need to handle
    Corner case 1:
        {"explanation": ...}
        Wait, no! I made a mistake. The prediction does not exactly match the ground truth. ...
        {...}

    Corner case 2:
        {"score": 0, "explanation": "The prediction does not contain item, nick "goose" bradshaw, that is in the ground truth."}
        return a tuple of (explanation, score)
    """
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"
    try:
        score = -1
        # Pattern to match the score
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score != 0 and score != 1:
                raise Exception("bad score: " + response)
        else:
            return "Parse Err: Score not found", -1

        # Pattern to match the explanation
        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        if explanation_match:
            explanation = explanation_match.group(1)
            return explanation, score
        else:
            return text, score
    except Exception as e:
        print(f"Parsing Error with resp: {response}")
        print(f"Error: {e}")
        return response, -1


def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens using Llama2 tokenizer"""
    global tokenizer
    if tokenizer is None:
        from transformers import LlamaTokenizerFast

        tokenizer = LlamaTokenizerFast.from_pretrained("tokenizer")
    max_token_length = 75
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction


def load_data_in_batches(dataset_path, batch_size):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.
    
    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.
    
    Yields:
    dict: A batch of data.
    """
    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": []}

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)
                    for key in batch:
                        batch[key].append(item[key])
                    
                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e



def generate_predictions(dataset_path, participant_model):
    """
    Processes batches of data from a dataset to generate predictions using a model.
    
    Args:
    dataset_path (str): Path to the dataset.
    participant_model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.
    
    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions = [], [], []
    batch_size = participant_model.get_batch_size()

    for batch in tqdm(load_data_in_batches(dataset_path, batch_size), desc="Generating predictions"):
        batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        batch_predictions = participant_model.batch_generate_answer(batch)
        
        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)
    
    return queries, ground_truths, predictions


def evaluate_predictions(queries, ground_truths_list, predictions, evaluation_model_name, client=None):
    """
    Evaluates the predictions generated by a model against ground truth answers.
    
    Args:
    queries (List[str]): List of queries.
    ground_truths_list (List[List[str]]): List of lists of ground truth answers. 
        Note each query can have multiple ground truth answers.
    predictions (list): List of predictions generated by the model.
    evaluation_model_name (str): Name of the evaluation model.
    
    Returns:
    dict: A dictionary containing evaluation results.
    """

    if "chat" in evaluation_model_name.lower():
        # now we are using chatgpt
        openai_client = client or build_openai_client()
        n_miss, n_correct = 0, 0
        system_message = get_system_message()

        for _idx, prediction in enumerate(tqdm(
            predictions, total=len(predictions), desc="Evaluating Predictions"
        )):
            query = queries[_idx]
            ground_truths = ground_truths_list[_idx]
            if isinstance(ground_truths, str):
                ground_truths = [ground_truths.strip()]
            # trim prediction to 75 tokens using Llama2 tokenizer
            prediction = trim_predictions_to_max_token_length(prediction)
            prediction = prediction.strip()
            prediction_lowercase = prediction.lower()

            if "i don't know" in prediction_lowercase:
                n_miss += 1
                continue

            accuracy = -1

            for ground_truth in ground_truths:
                ground_truth_lowercase = ground_truth.lower()
                messages = [
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
                    },
                ]
                if prediction_lowercase == ground_truth_lowercase:
                    # exact correct
                    accuracy = 1
                    break
                elif "invalid" in prediction_lowercase and "invalid" in ground_truth_lowercase:
                    accuracy = 1
                    break
                elif "invalid" in prediction_lowercase and "invalid" not in ground_truth_lowercase:
                    # hallucination
                    accuracy = 0
                    continue
                elif "invalid" not in prediction_lowercase and "invalid" in ground_truth_lowercase:
                    # hallucination
                    accuracy = 0
                    continue
                else:
                    # need to use the OpenAI evaluation model to get the accuracy result (0 means wrong, 1 means correct)
                    response = attempt_api_call(openai_client, evaluation_model_name, messages)
                    if response:
                        log_response(messages, response)
                        _, accuracy = parse_response(response)
                        if accuracy == 1:
                            # no need to check other ground truth(s)
                            break

            if accuracy == 1:
                n_correct += 1

        n = len(predictions)
        results = {
            "score": (2 * n_correct + n_miss) / n - 1,
            "accuracy": n_correct / n,
            "hallucination": (n - n_correct - n_miss) / n,
            "missing": n_miss / n,
            "n_miss": n_miss,
            "n_correct": n_correct,
            "n_hallucination": n - n_correct - n_miss,
            "total": n,
        }
        logger.info(results)
        return results
    elif "llama" in evaluation_model_name.lower():
        # now we are using llama model to evaluate
        # to be filled by Jiaqi
        raise NotImplementedError("Llama evaluation model is not implemented yet.")
    else:
        raise NotImplementedError(f"Unknown evaluation model: {evaluation_model_name}")


def normalize_answer(text):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def is_missing_prediction(prediction):
    prediction = normalize_answer(prediction)
    missing_markers = (
        "i dont know",
        "unknown",
        "not sure",
        "cannot answer",
        "cant answer",
        "do not know",
        "no answer",
    )
    return any(marker in prediction for marker in missing_markers)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_prediction_text(pred_row, prediction_field=None):
    fields = [prediction_field] if prediction_field else ["predicted", "prediction", "final"]
    for field in fields:
        if field and field in pred_row:
            value = pred_row[field]
            if isinstance(value, list):
                return " ".join(str(item) for item in value)
            return str(value)
    raise ValueError(f"Prediction row has none of these fields: {fields}")


def load_result_pairs(gold_path, prediction_path, prediction_field=None):
    gold_by_index = {int(row["index"]): row for row in load_jsonl(gold_path)}
    pairs = []
    missing_gold = []

    for pred_row in load_jsonl(prediction_path):
        index = int(pred_row["index"])
        gold_row = gold_by_index.get(index)
        if gold_row is None:
            missing_gold.append(index)
            continue
        pairs.append(
            {
                "index": index,
                "question": pred_row.get("question") or gold_row.get("question", ""),
                "prediction": get_prediction_text(pred_row, prediction_field),
                "ground_truths": gold_row.get("answers", pred_row.get("gold", [])),
            }
        )

    if missing_gold:
        raise ValueError(f"Prediction rows without gold index: {missing_gold[:10]}")
    if len(pairs) != len(gold_by_index):
        logger.warning(f"Loaded {len(pairs)} prediction rows for {len(gold_by_index)} gold rows.")
    return pairs


def evaluate_f1_pairs(pairs):
    scores = []
    exact = 0
    missing = 0

    for pair in pairs:
        prediction = pair["prediction"]
        ground_truths = pair["ground_truths"]
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        if is_missing_prediction(prediction):
            missing += 1
        best_f1 = max((token_f1(prediction, truth) for truth in ground_truths), default=0.0)
        best_exact = max(
            (normalize_answer(prediction) == normalize_answer(truth) for truth in ground_truths),
            default=False,
        )
        scores.append(best_f1)
        exact += int(best_exact)

    total = len(pairs)
    return {
        "metric": "f1",
        "f1": sum(scores) / total if total else 0.0,
        "exact_match": exact / total if total else 0.0,
        "missing": missing / total if total else 0.0,
        "n_missing": missing,
        "total": total,
    }


def evaluate_llm_pairs(pairs, evaluation_model_name, client, log_dir="api_responses"):
    n_miss, n_correct = 0, 0
    system_message = get_system_message()

    for pair in tqdm(pairs, total=len(pairs), desc="LLM evaluating"):
        prediction = str(pair["prediction"]).strip()
        prediction_lowercase = prediction.lower()
        if is_missing_prediction(prediction):
            n_miss += 1
            continue

        accuracy = 0
        ground_truths = pair["ground_truths"]
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]

        for ground_truth in ground_truths:
            ground_truth = str(ground_truth).strip()
            ground_truth_lowercase = ground_truth.lower()
            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": (
                        f"Question: {pair['question']}\n"
                        f"Ground truth: {ground_truth}\n"
                        f"Prediction: {prediction}\n"
                    ),
                },
            ]
            if prediction_lowercase == ground_truth_lowercase:
                accuracy = 1
                break
            if "invalid" in prediction_lowercase or "invalid" in ground_truth_lowercase:
                accuracy = int("invalid" in prediction_lowercase and "invalid" in ground_truth_lowercase)
                if accuracy == 1:
                    break
                continue

            response = attempt_api_call(client, evaluation_model_name, messages)
            if response:
                log_response(messages, response, log_dir)
                _, accuracy = parse_response(response)
                if accuracy == 1:
                    break

        if accuracy == 1:
            n_correct += 1

    total = len(pairs)
    return {
        "metric": "llm",
        "model": evaluation_model_name,
        "score": (2 * n_correct + n_miss) / total - 1 if total else 0.0,
        "accuracy": n_correct / total if total else 0.0,
        "hallucination": (total - n_correct - n_miss) / total if total else 0.0,
        "missing": n_miss / total if total else 0.0,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_hallucination": total - n_correct - n_miss,
        "total": total,
    }


def evaluate_result_files(args):
    pairs = load_result_pairs(args.gold_path, args.prediction_path, args.prediction_field)
    results = {}

    if args.metric in ("f1", "both"):
        results["f1"] = evaluate_f1_pairs(pairs)
    if args.metric in ("llm", "both"):
        client = build_openai_client(api_key=args.api_key, base_url=args.base_url)
        results["llm"] = evaluate_llm_pairs(
            pairs,
            args.evaluation_model_name,
            client,
            log_dir=args.log_dir,
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CRAG jsonl result files.")
    parser.add_argument(
        "--gold-path",
        default=r"E:\fff_atomr\CRAG_new\results\gold\CRAG_test.jsonl",
        help="Gold CRAG jsonl path.",
    )
    parser.add_argument(
        "--prediction-path",
        default=r"E:\fff_atomr\CRAG_new\results\Atomr\crag_test.jsonl",
        help="Prediction jsonl path. Auto-detects 'predicted', 'prediction', or 'final'.",
    )
    parser.add_argument("--prediction-field", default=None, help="Override the prediction field name.")
    parser.add_argument("--metric", choices=("f1", "llm", "both"), default="both")
    parser.add_argument("--evaluation-model-name", default=os.getenv("EVALUATION_MODEL_NAME", "deepseek-chat"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--log-dir", default="api_responses")
    parser.add_argument("--output-path", default=None)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run the original end-to-end generation flow instead of evaluating existing result files.",
    )
    parser.add_argument("--dataset-path", default="example_data/dev_data.jsonl.bz2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.generate:
        evaluate_result_files(args)
        raise SystemExit(0)

    from models.user_config import UserModel

    # Generate predictions
    participant_model = UserModel()
    queries, ground_truths, predictions = generate_predictions(args.dataset_path, participant_model)
    # Evaluate Predictions
    openai_client = build_openai_client(api_key=args.api_key, base_url=args.base_url)
    evaluation_results = evaluate_predictions(
        queries, ground_truths, predictions, args.evaluation_model_name, openai_client
    )
    print(json.dumps(evaluation_results, ensure_ascii=False, indent=2))
