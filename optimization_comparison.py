import pandas as pd
import matplotlib.pyplot as plt

# Load tuning results
tuning_results = pd.read_csv('hyperparameter_tuning_results_baseline.csv')

# Extract validation losses and trial IDs
validation_loss = tuning_results['Final Score']
trial_ids = tuning_results['Trial ID']

# Highlight the baseline trial
baseline_loss = 0.4098
baseline_label = 'Baseline'

# Scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(trial_ids, validation_loss, label='Tuning Trials', color='blue')
plt.scatter(
    trial_ids[tuning_results['Trial ID'] == 'Baseline'], 
    validation_loss[tuning_results['Trial ID'] == 'Baseline'], 
    color='red', label=f'{baseline_label} ({baseline_loss})', zorder=5
)
plt.title('Validation Loss Across Tuning Trials')
plt.xlabel('Trial ID')
plt.ylabel('Validation Loss')
plt.axhline(y=baseline_loss, color='red', linestyle='--', label=f'Baseline (Loss: {baseline_loss})')
plt.xticks(rotation=45, ha='right')  # Adjust rotation and alignment of x-axis labels
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig('tuning_vs_baseline_scatter_with_highlight.png')
plt.show()
