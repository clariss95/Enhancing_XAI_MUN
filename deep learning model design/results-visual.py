import json
import matplotlib.pyplot as plt
import pandas as pd

# Load inner loop results
with open("results/all_inner_results.json", "r") as f:
    inner_results = json.load(f)

# Flatten inner results for visualization and table
flat_results = []
for outer_fold, combinations in inner_results.items():
    for result in combinations:
        flat_results.append({
            "Outer Fold": outer_fold,
            "Hyperparameters": tuple(result["Hyperparameters"]),
            "Validation Loss (Per Fold)": result["Validation Loss (Per Fold)"],
            "Average Validation Loss": result["Average Validation Loss"]
        })

# Sort by average validation loss for consistent plotting and tabulation
sorted_results = sorted(flat_results, key=lambda x: x["Average Validation Loss"])

# Prepare data for the plot
hyperparameters = [str(res["Hyperparameters"]) for res in sorted_results]
avg_validation_losses = [res["Average Validation Loss"] for res in sorted_results]

# Plot
plt.figure(figsize=(12, 8))
plt.barh(hyperparameters, avg_validation_losses, color="skyblue")
plt.xlabel("Average Validation Loss", fontsize=12)
plt.ylabel("Hyperparameters", fontsize=12)
plt.title("Average Validation Loss Across Hyperparameter Combinations", fontsize=14)
plt.tight_layout()
plt.savefig("results-graph.png")
plt.show()



import json
import pandas as pd

# Load the inner loop results from the JSON file
with open("results/all_inner_results.json", "r") as f:
    inner_results = json.load(f)

# Flatten inner results for processing
flat_results = []
for outer_fold, combinations in inner_results.items():
    for result in combinations:
        flat_results.append({
            "Hyperparameters": tuple(result["Hyperparameters"]),
            "Validation Loss (Per Fold)": result["Validation Loss (Per Fold)"],
            "Average Validation Loss": result["Average Validation Loss"]
        })

# Group by hyperparameter combinations and calculate averages
grouped_results = {}
for result in flat_results:
    hyperparams = result["Hyperparameters"]
    if hyperparams not in grouped_results:
        grouped_results[hyperparams] = {"Inner Fold Sums": [0] * len(result["Validation Loss (Per Fold)"]),
                                        "Count": 0}
    
    # Add inner fold losses
    for i, loss in enumerate(result["Validation Loss (Per Fold)"]):
        grouped_results[hyperparams]["Inner Fold Sums"][i] += loss
    
    # Increment the count
    grouped_results[hyperparams]["Count"] += 1

# Prepare the table
table_data = []
for hyperparams, data in grouped_results.items():
    count = data["Count"]
    # Calculate averages for each inner fold
    inner_fold_averages = [s / count for s in data["Inner Fold Sums"]]
    # Calculate overall average validation loss
    final_average_loss = sum(inner_fold_averages) / len(inner_fold_averages)
    row = {
        "Hyperparameters": hyperparams,
        **{f"Validation Loss (Inner Fold {i+1})": avg for i, avg in enumerate(inner_fold_averages)},
        "Final Average Validation Loss": final_average_loss
    }
    table_data.append(row)

# Convert to DataFrame
grouped_df = pd.DataFrame(table_data)

# Sort by final average validation loss
grouped_df = grouped_df.sort_values(by="Final Average Validation Loss").reset_index(drop=True)

# Save the results to a CSV file
grouped_df.to_csv("grouped_hyperparameters_with_folds.csv", index=False)

print("Results saved to 'grouped_hyperparameters_with_folds.csv'.")


