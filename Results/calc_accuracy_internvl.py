import os
import pandas as pd
import json

base_dir = "results"

# Store raw individual results
raw_results = {}

# Store averaged output: q1 -> {coarse: avg, granular: avg}
avg_results = {}

for q_folder in sorted(os.listdir(base_dir)):
    q_path = os.path.join(base_dir, q_folder)
    if not os.path.isdir(q_path):
        continue

    raw_results[q_folder] = {"coarse": {}, "granular": {}}
    avg_results[q_folder] = {"coarse": [], "granular": []}
    #  Test
    for filename in os.listdir(q_path):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(q_path, filename)
        df = pd.read_csv(file_path)

        # Calculate accuracy
        correct = df["is_correct"].sum()
        total = len(df)
        accuracy = 100 * correct / total if total > 0 else 0.0

        # Append accuracy to CSV
        num_cols = len(df.columns)
        acc_row_data = [""] * (num_cols - 1) + [f"Accuracy: {accuracy:.2f}"]
        acc_row = pd.DataFrame([acc_row_data], columns=df.columns)
        df = pd.concat([df, acc_row], ignore_index=True)
        df.to_csv(file_path, index=False)

        # Determine qtype and dataset
        fsplit = filename.split("_")
        if "coarse" in fsplit:
            qtype = "coarse"
            index = fsplit.index("coarse")
            dataset = "_".join(fsplit[index + 1 : -1])
        elif "granular" in fsplit:
            qtype = "granular"
            index = fsplit.index("granular")
            dataset = "_".join(fsplit[index + 1 : -1])
        else:
            print(f"Warning: Could not parse type from {filename}")
            continue

        raw_results[q_folder][qtype][dataset] = round(accuracy, 2)
        avg_results[q_folder][qtype].append(accuracy)

# Print final average results
for q, q_data in avg_results.items():
    for qtype in ["coarse", "granular"]:
        scores = q_data[qtype]
        if scores:
            avg_acc = sum(scores) / len(scores)
            print(f"{q} - {qtype} - {avg_acc:.2f}%")
        else:
            print(f"{q} - {qtype} - No data")

# Optional: Save full raw + avg results
with open(os.path.join(base_dir, "accuracy_by_question_and_type.json"), "w") as f:
    json.dump(raw_results, f, indent=2)
