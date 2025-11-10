This project evaluates vision-language models such as QWEN2.5-VL and InternVL on several datasets. It includes scripts for model inference, accuracy calculation, and converting CSV results into structured JSON summaries.

Files included:

1. internvl_eval_v3.py
   Runs InternVL inference on the DORI dataset.

   * Takes question number, datasets, and model parameters.
   * Uses lmdeploy pipeline for model inference.
   * Saves per-question results as CSVs inside the “results” folder.
     Example:
     python internvl_eval_v3.py --q_num q1 --datasets get_3d omniobject3d --level coarse

2. query_blink.py
   Runs QWEN models on the BLINK benchmark (tasks like Relative_Depth or Spatial_Relation).

   * Loads the dataset from HuggingFace.
   * Generates multiple-choice prompts and saves predictions.
     Example:
     python query_blink.py --model_name QWEN --task_name Relative_Depth --model_type Base

3. query_spatial.py
   Evaluates QWEN models on the SpatialMM dataset.

   * Loads image-question pairs.
   * Sends prompts to the model and extracts answers (A–E).
   * Computes and prints accuracy.
     Example:
     python query_spatial.py --model_name QWEN --model_type tuned

4. calc_accuracy_internvl.py
   Calculates accuracy from all InternVL CSV result files.

   * Scans the “results” directory for .csv files.
   * Computes average accuracy per question and evaluation level.
   * Saves a summary JSON file named accuracy_by_question_and_type.json.
     Example:
     python calc_accuracy_internvl.py

5. parseModelResultsCSV.py
   Converts combined CSV accuracy files into structured JSON.

   * Reads model_results.csv and outputs model_results.json.
   * Cleans formatting and converts numeric values.
     Example:
     python parseModelResultsCSV.py

6. model_results.csv
   Example input CSV containing all accuracy metrics.

7. model_results.json
   Output JSON summary produced by parseModelResultsCSV.py.

Dependencies:

* Python 3.9 or higher
* torch
* torchvision
* transformers
* tqdm
* pillow
* pandas
* datasets
* lmdeploy

Basic workflow:

1. Run the model evaluation scripts (query_spatial.py, query_blink.py, internvl_eval_v3.py).
2. Calculate accuracy summaries using calc_accuracy_internvl.py.
3. Combine results into a final JSON using parseModelResultsCSV.py.

Output files include both per-task accuracies and overall summary results.

---
