"""
Adaptation of code previously written for a llava model to run InternVL on the DORI dataset - a custom

"""

import os
import random
import pandas as pd
import json
import argparse
import re
from tqdm import tqdm
from PIL import Image
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

RANDOM_SEED = 1998


# For reproducability, may not be nessesary
def set_all_seeds(seed):
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    print(f"Random seed set to {seed}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with InternVL model")
    parser.add_argument(
        "--q_num", type=str, required=True, help="Question number (e.g., q1, q2)"
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", required=True, help="Datasets to process"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="OpenGVLab/InternVL2_5-4B",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="exp_internvl2_5B",
        help="Model name for result saving",
    )
    parser.add_argument(
        "--promptfile",
        type=str,
        default="benchmark_exp.json",
        help="Prompt config file",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument(
        "--test", action="store_true", help="Run script in test mode with one sample"
    )
    parser.add_argument(
        "--level",
        choices=["coarse", "granular"],
        help="Specify whether to run only 'coarse' or 'granular' evaluations",
    )
    return parser.parse_args()


def load_answers(answer_path):
    answers = {}
    with open(answer_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                image_name = parts[0]
                answer = parts[2]
                answers[image_name] = answer
    return answers


def get_shuffled_options_with_letters(fixed_options):
    options = fixed_options.copy()
    random.shuffle(options)
    lettered_options = []
    options_map = {}
    for i, option in enumerate(options):
        letter = chr(65 + i)
        lettered_options.append(f"{letter}. {option}")
        options_map[letter] = option
    return "\n".join(lettered_options), options_map


def extract_answer(text, options_map):
    final_answer = "UNKNOWN"
    final_reasoning = ""
    available_letters = list(options_map.keys())
    letters_pattern = "[" + "".join(available_letters) + "]"

    patterns = [
        f"(?:^|\\s)({letters_pattern})(?:$|\\s|\\.|\\n|:|,)",
        f"(?:answer|choice|option)(?:\\s+is)?(?:\\s*:|,?\\s+)[^A-Za-z]*({letters_pattern})(?:\\b|\\.|\\s|$)",
        f"\\b({letters_pattern})\.",
        f"(?:choose|select|pick|opt for|go with)(?:\\s+option|\\s+answer)?(?:\\s+|\\s*:\\s*)({letters_pattern})(?:\\b|\\.|\\s|$)",
    ]

    all_matches = []
    for pattern in patterns:
        all_matches += re.findall(pattern, text, re.IGNORECASE)

    letter_counts = {}
    for letter in all_matches:
        if letter in options_map:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1

    if letter_counts:
        most_common_letter = max(letter_counts.items(), key=lambda x: x[1])[0]
        final_answer = options_map[most_common_letter]

    if final_answer == "UNKNOWN":
        for letter, option_text in options_map.items():
            if re.search(r"\b" + re.escape(option_text) + r"\b", text, re.IGNORECASE):
                final_answer = option_text
                break

    return final_answer, final_reasoning


def process_dataset(
    ground_truth,
    image_dir,
    base_question,
    fixed_options,
    remap,
    remap_dict,
    num_options,
    pipe,
    test_mode=False,
):
    results = []

    for idx, (image_name, correct_answer) in enumerate(
        tqdm(ground_truth.items(), desc="Processing images")
    ):
        if test_mode and idx > 0:
            break

        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            for ext in [".png", ".jpg", ".jpeg"]:
                if os.path.exists(image_path + ext):
                    image_path += ext
                    break
            else:
                print(f"Missing: {image_name}")
                continue

        try:
            image = load_image(image_path)
            options_str, options_map = get_shuffled_options_with_letters(
                list(remap_dict.values()) if remap else fixed_options
            )
            question = base_question.format(
                options=options_str, num_options=num_options
            )

            print(f"\nQuestion: {question}")
            response = pipe((question, image))
            text_output = response.text
            print(f"Model Output: {text_output}")

            model_answer, model_reason = extract_answer(text_output, options_map)
            is_correct = model_answer.lower() == correct_answer.lower()

            results.append(
                {
                    "image_name": image_name,
                    "ground_truth": correct_answer,
                    "model_prediction": model_answer,
                    "is_correct": is_correct,
                    "options_map": str(options_map),
                    "prompt": question,
                    "full_response": text_output,
                    "reasoning": model_reason,
                }
            )

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    return results


def save_results(
    results, model_name, q_num, q_level, dataset, image_dir, fixed_options, SEED
):
    results_df = pd.DataFrame(results)
    correct_count = sum(1 for r in results if r["is_correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"\nAccuracy: {accuracy:.4f} ({correct_count}/{total_count})")

    output_dir = os.path.join("results", q_num)
    os.makedirs(output_dir, exist_ok=True)
    csv_name = f"s{SEED}_{model_name}_{q_num}_{q_level}_{dataset}_results.csv"
    results_df.to_csv(os.path.join(output_dir, csv_name), index=False)


def run_batch_inference(
    q_num,
    datasets,
    PRETRAINED,
    MODEL_NAME,
    PROMPTFILE,
    SEED,
    test_mode=False,
    level=None,
):
    print(f"Running InternVL on question {q_num} for datasets: {datasets}")
    set_all_seeds(SEED)

    with open(PROMPTFILE, "r") as f:
        CONFIG = json.load(f)

    if q_num not in CONFIG:
        raise ValueError(f"Question {q_num} not found in config")

    config = CONFIG[q_num]
    pipe = pipeline(PRETRAINED, backend_config=TurbomindEngineConfig(session_len=8192))

    for dataset in datasets:
        levels_to_run = [level] if level else config["levels"].keys()

        for q_level in levels_to_run:
            print(f"\n--- {q_num} ({q_level}) on {dataset} ---")
            output_dir = os.path.join("results", q_num)
            os.makedirs(output_dir, exist_ok=True)
            csv_name = f"s{SEED}_{MODEL_NAME}_{q_num}_{q_level}_{dataset}_results.csv"

            dataset_config = config["datasets"][dataset]
            level_config = config["levels"][q_level]

            remap = level_config.get("remap", "false") == "true"
            IMAGE_DIR = dataset_config["image_dir"]
            ANSWER_PATH = dataset_config["answer_path"][q_level]
            FIXED_OPTIONS = level_config["options"]
            BASE_QUESTION = level_config["question"]
            num_options = len(FIXED_OPTIONS)

            ground_truth = load_answers(ANSWER_PATH)
            if remap:
                REMAP_DICT = level_config["options_remapped"]
                ground_truth = {k: REMAP_DICT[v] for k, v in ground_truth.items()}
            else:
                REMAP_DICT = None

            results = process_dataset(
                ground_truth,
                IMAGE_DIR,
                BASE_QUESTION,
                FIXED_OPTIONS,
                remap,
                REMAP_DICT,
                num_options,
                pipe,
                test_mode=test_mode,
            )

            save_results(
                results,
                MODEL_NAME,
                q_num,
                q_level,
                dataset,
                IMAGE_DIR,
                FIXED_OPTIONS,
                SEED,
            )


if __name__ == "__main__":
    args = parse_arguments()
    run_batch_inference(
        args.q_num,
        args.datasets,
        args.pretrained,
        args.model_name,
        args.promptfile,
        args.seed,
        test_mode=args.test,
        level=args.level,
    )
