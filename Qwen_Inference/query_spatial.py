import json
from tqdm import tqdm
from PIL import Image
import os
from multiple_choice import match_multiple_choice  # Optional, use if needed
from query_model import query_qwen
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import random
import numpy as np
import argparse

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
TUNED_MODEL = "appledora/QWEN2.5-3B-Instruct-DORI-tuned"

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_model(model_choice):
    model_id = BASE_MODEL if model_choice == "base" else TUNED_MODEL
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return [model, processor], model_choice

def analyze_answer(model_, d, gpt_answer, all_choices):
    try:
        if isinstance(gpt_answer, list):
            gpt_answer = gpt_answer[0] if gpt_answer else ""
        gpt_answer_clean = gpt_answer.strip()
        first_letter = gpt_answer_clean[0].upper()
        if first_letter in ["A", "B", "C", "D", "E"]:
            return first_letter
    except Exception as e:
        print(f"Error analyzing answer: {e}")
    return None

def load_prompt(d, image_folder):
    image_path = os.path.join(image_folder, d["image_name"])

    try:
        with Image.open(image_path) as img:
            img.verify()  # Optional: verify image is not corrupt
    except Exception as e:
        print(f"Warning: Could not open image {image_path}. Skipping. Error: {e}")
        return None, None

    prompt = d["question"]
    prompt += (
        "\n\nTASK: Choose the correct answer to the question based on the provided image."
        "\nYou MUST select ONLY ONE option from: (A), (B), (C), (D), or (E)."
        "\nRespond with ONLY the letter of your final answer (e.g., A, B, C, D, or E)."
    )

    return [image_path], prompt

def query_model(model_, model_type):
    json_path = "spatial_mm_one_obj.json"
    image_folder = "."  # images are in current directory
    output_folder = "outputs_spatialMM"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"spatialMM_results_{model_type}.json")

    with open(json_path, "r") as f:
        dataset = json.load(f)

    outputs = []
    skipped = 0

    for idx, d in enumerate(tqdm(dataset)):
        print(f"Processing idx: {idx}, image: {d['image_name']}")
        gold_answer = d["answer"][0]
        all_choices = ["(A)", "(B)", "(C)", "(D)", "(E)"]

        image_paths, prompt = load_prompt(d, image_folder)
        if image_paths is None or prompt is None:
            skipped += 1
            continue

        gpt_answer = query_qwen(model_, image_paths, prompt)
        prediction = analyze_answer(model_, d, gpt_answer, all_choices)

        outputs.append({
            "idx": idx,
            "image_name": d["image_name"],
            "question": d["question"],
            "answer": gold_answer,
            "full_prediction": gpt_answer,
            "prediction": prediction
        })

        print(f"\nAnswer: {gold_answer}\nPrediction: {prediction}\nFull Prediction: {gpt_answer}")

        if prediction == gold_answer:
            print("Correct")
        else:
            print("Incorrect")

    correct = sum(1 for d in outputs if d["answer"] == d["prediction"])
    accuracy = round(100 * correct / len(outputs), 2) if outputs else 0.0
    print("-" * 50)
    print(f"Accuracy on spatialMM: {accuracy}%")
    print(f"Skipped {skipped} items due to missing or unreadable images.")

    results = {
        "results": outputs,
        "accuracy": accuracy
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="QWEN", help="Select the model name")
    parser.add_argument("--model_type", type=str, choices=["base", "tuned"], default="tuned", help="Choose between base or tuned model")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(1235)
    model_, model_type = load_model(args.model_type)
    print(f"Using model: {args.model_name} ({model_type})")

    model_generate_funcs = {
        "QWEN": query_qwen
    }
    model_generate_func = model_generate_funcs[args.model_name]

    query_model(model_, model_type)
