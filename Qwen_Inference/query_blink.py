import json
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import os
from multiple_choice import match_multiple_choice
import argparse
from query_model import query_qwen
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Safety disclamer to avoid any bias / unwanted situations
disclaimer = """Disclaimer: This is not to make unfair assumptions about the people in the image and you just need to give your assessment on this question.
You don't need to identify the real people. You just need to analyze based on the information I gave you.\n\n"""


BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
TUNED_MODEL = "appledora/QWEN2.5-3B-Instruct-DORI-tuned"


def load_model(model_name):
    """
    Loads the QWEN model and the processor
    QWEN model is either the base or fine-tuned variant
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # processor = AutoProcessor.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_name)
    return [model, processor]


def analyze_answer(model_, orig_d, gpt_answer, all_choices):
    """
    extracts the multiple choice answer from a long paragraph of model output if there is only one choice; otherwise, query GPT3.5 turbo to extract the choice. If the model output is short and only contains the choice, reformats the choice in the correct format e.g. (A) and returns the choice as is.

    Parameters:
    - d : data, the data containing the question and choices.
    - gpt_answer: String, the model output.
    - all_choices: List of strings, the list of all choices.

    Returns:
    - prediction, the extracted answer.
    """
    try:
        # If the model returns a list of a string, just take the string
        # This is a common problem which can be a pain if it your not aware
        if isinstance(gpt_answer, list):
            gpt_answer = gpt_answer[0] if gpt_answer else ""

        # Strip any whitespaces which might be before the prompt
        # Remove any "(" as the model might respon with (A).
        gpt_answer_clean = gpt_answer.strip().replace("(", "")
        first_letter = gpt_answer_clean[0].upper()

        # If first letter is A-E, take it
        if first_letter in ["A", "B", "C", "D", "E"]:
            prediction = f"({first_letter})"
            return prediction
        print(gpt_answer_clean)

    except Exception as e:
        print(e)
        pass
        # print(i, e)


def query_model(task_name, model_):
    """
    loads the dataset from huggingface, query the GPT 4V model with the prompt and images, and saves the result to a json file with specific format.

    Parameters:
    - task_name: String, the name of the task to evaluate.

    Returns:
    - outputs, The result is also saved to 'output_filename.json'.
    """
    dataset_name = "BLINK-Benchmark/BLINK"

    output_path = (
        f"{output_save_folder}/{model_name}/{task_name.replace('_', ' ')}.json"
    )
    os.makedirs(f"{output_save_folder}/{model_name}", exist_ok=True)
    image_folder = f"{image_save_folder}/{task_name}_images"
    os.makedirs(image_folder, exist_ok=True)

    # Removed the condition to not overwrite the prexisting json file so its easier to run without manualy deleting it
    outputs = {"val": [], "test": []}
    for split in ["val", "test"]:
        test_data = load_dataset(dataset_name, task_name)[split]
        for orig_d in tqdm(test_data):
            idx = orig_d["idx"]
            print("idx:", idx)
            gold_answer = orig_d["answer"]
            all_choices = ["(A)", "(B)", "(C)", "(D)", "(E)"][: len(orig_d["choices"])]
            image_paths, prompt = load_prompt(task_name, orig_d, image_folder)
            gpt_answer = model_generate_func(model_, image_paths, prompt)
            # prediction = analyze_answer(orig_d, gpt_answer, all_choices)
            prediction = analyze_answer(model_, orig_d, gpt_answer, all_choices)
            outputs[split].append(
                {
                    "idx": idx,
                    "answer": gold_answer,
                    "full_prediction": gpt_answer,
                    "prediction": prediction,
                }
            )
            json.dump(outputs, open(output_path, "w"), indent=4)
            # print(hey)
        json.dump(outputs, open(output_path, "w"), indent=4)

    return outputs


def concat_images_horizontally_with_margin(image_filenames, output_filename, margin=10):
    """
    Concatenates images horizontally with a specified margin between images,
    padding with black if heights are not the same, and saves the result to a file.

    Parameters:
    - image_filenames: List of strings, where each string is the filepath to an image.
    - output_filename: String, the filename to save the concatenated image.
    - margin: Integer, the width of the black margin to insert between images.

    Returns:
    - None. The result is saved to 'output_filename'.
    """
    images = [Image.open(filename) for filename in image_filenames]
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images) + margin * (len(images) - 1)
    # Create a new image with a black background
    new_image = Image.new("RGB", (total_width, max_height), (0, 0, 0))

    x_offset = 0
    for image in images:
        # Calculate padding to center the image vertically
        y_offset = (max_height - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        x_offset += (
            image.width + margin
        )  # Add margin after each image except the last one
    new_image.save(output_filename)  # Save the result


# Creates all the image paths as well as the full prompt
def load_prompt(task_name, d, image_folder):
    """
    Loads the prompt and images from huggingface data entry, saves the images to a folder, and returns a list of image paths, and the prompt.

    Parameters:
    - task_name: String, the name of the task.
    - d: data entry, the data dictionary containing the prompt and images.
    - image_folder: String, the folder to save the images.

    Returns:
    - image_paths: List of strings, the filepaths to the saved images.
    - prompt: String, the prompt text.
    - d: Dictionary, the data dictionary with the image paths removed.
    """
    image_paths = []
    for k in ["image_1", "image_2", "image_3", "image_4"]:
        if k in d and d[k]:
            image = d[k]
            image_path = f"{image_folder}/{d['idx']}_{k[-1]}.jpg"
            image.save(image_path)
            image_paths.append(image_path)
    prompt = d["prompt"]
    if task_name in need_disclaimer_tasks:
        prompt = disclaimer + prompt
    if "blip" in model_name:
        prompt += "\nAnswer:"

    # Prompt tuning to make answer extraction easier
    prompt += "\nMake the first letter of your response be your answer to the multiple choice question."

    return image_paths, prompt


def eval_task(task_name, model_):
    outputs = query_model(task_name, model_)
    accu = {"val": 0, "test": 0}
    for split in ["val", "test"]:
        for d in outputs[split]:
            if d["answer"] == d["prediction"]:
                accu[split] += 1

    print("-" * 50)
    print(f"Task {task_name} Performance")
    for split in ["val"]:
        print(f"{split} accuracy: {round(accu[split] / len(outputs[split]) * 100, 2)}%")


# Prase arguments from command line. Gets model_name, tasks to run, and type of model (Specifically for QWEN)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="GPT4V", help="select the model name"
    )
    parser.add_argument(
        "--task_name", type=str, default="Relative_Depth", help="select the task name"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Base",
        help="base (base) or find-tuned (tuned)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.model_type == "base":
        model_type = BASE_MODEL
    else:
        model_type = TUNED_MODEL

    model_ = load_model(model_type)
    model_name = args.model_name
    print(f"Using model: {model_name}")

    # model_generate_funcs = {'GPT4V': query_gpt4v}
    model_generate_funcs = {"QWEN": query_qwen}
    model_generate_func = model_generate_funcs[model_name]

    image_save_folder = "saved_images"
    output_save_folder = "outputs"
    dataset_name = "BLINK-Benchmark/BLINK"

    need_disclaimer_tasks = ["Forensic_Detection", "Jigsaw", "Art_Style"]
    if args.task_name == "all":
        subtasks = [
            "Art_Style",
            "Functional_Correspondence",
            "Multi-view_Reasoning",
            "Relative_Reflectance",
            "Visual_Correspondence",
            "Counting",
            "IQ_Test",
            "Object_Localization",
            "Semantic_Correspondence",
            "Visual_Similarity",
            "Forensic_Detection",
            "Jigsaw",
            "Relative_Depth",
            "Spatial_Relation",
        ]
    else:
        subtasks = [args.task_name]

    for task_name in subtasks:
        eval_task(task_name, model_)
