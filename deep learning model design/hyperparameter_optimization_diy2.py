import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from itertools import product
import os
import json
import random




# Load data
event_features = np.load("event_features.npy")
case_features = np.load("case_features.npy")
labels = np.load("labels.npy")

print("Data loaded.")
print(event_features.shape, case_features.shape, labels.shape)

# Define sequence length and feature dimensions
sequence_length = 2
num_event_features = event_features.shape[2]
num_case_features = case_features.shape[1]

# Hyperparameter grid
hyperparameter_grid = {
    "lstm_units1": [32, 64, 128],
    "lstm_units2": [16, 32, 64],
    "dropout_rate": [0.2, 0.3, 0.5],
    "learning_rate": [1e-2, 1e-3, 1e-4],
}

# Define baseline hyperparameters
baseline_hyperparams = (64, 32, 0.5, 0.0001)

# Randomly sample hyperparameter combinations
random.seed(42)
hyperparameter_combinations = random.sample(
    list(product(
        hyperparameter_grid["lstm_units1"],
        hyperparameter_grid["lstm_units2"],
        hyperparameter_grid["dropout_rate"],
        hyperparameter_grid["learning_rate"]
    )),
    k=9  # Select 9 random combinations
)

# Add the baseline hyperparameters explicitly
if baseline_hyperparams not in hyperparameter_combinations:
    hyperparameter_combinations.append(baseline_hyperparams)

print("Hyperparameter combinations to test (including baseline):")
print(hyperparameter_combinations)

# Define a callback for validation loss threshold
class ValidationLossThreshold(Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
    
    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if epoch >= 2 and current_val_loss > self.threshold:
            print(f"\nPruning model at epoch {epoch+1} with validation loss {current_val_loss:.4f} exceeding threshold {self.threshold}")
            self.model.stop_training = True

# Function to build the LSTM model
def build_model(hyperparams):
    lstm_units1, lstm_units2, dropout_rate, learning_rate = hyperparams

    event_input = Input(shape=(sequence_length, num_event_features), name="event_input")
    lstm_out = LSTM(lstm_units1, return_sequences=True)(event_input)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    lstm_out = LSTM(lstm_units2, return_sequences=False)(lstm_out)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)

    case_input = Input(shape=(num_case_features,), name="case_input")
    combined = Concatenate()([lstm_out, case_input])
    output = Dense(1, activation="sigmoid")(combined)

    model = Model(inputs=[event_input, case_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Outer cross-validation
outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
final_results = []

# Path to save logs and checkpoints
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
results_log = []
inner_results_log = {}  # To store inner loop results

for outer_fold, (train_idx, test_idx) in enumerate(outer_kf.split(event_features)):
    print(f"\nStarting Outer Fold {outer_fold + 1}")

    # Save intermediate progress
    intermediate_results_path = f"results/intermediate_results_fold_{outer_fold + 1}.json"
    inner_results_path = f"results/inner_results_fold_{outer_fold + 1}.json"  # Path to save inner loop results
    model_checkpoint_path = f"checkpoints/best_model_fold_{outer_fold + 1}.h5"

    # Split training and test sets
    train_features_event = event_features[train_idx]
    test_features_event = event_features[test_idx]
    train_features_case = case_features[train_idx]
    test_features_case = case_features[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    # Standardize features
    scaler_event = StandardScaler()
    scaler_case = StandardScaler()
    train_features_event_scaled = scaler_event.fit_transform(
        train_features_event.reshape(-1, num_event_features)).reshape(train_features_event.shape[0], sequence_length, num_event_features)
    train_features_case_scaled = scaler_case.fit_transform(train_features_case)

    test_features_event_scaled = scaler_event.transform(
        test_features_event.reshape(-1, num_event_features)).reshape(test_features_event.shape[0], sequence_length, num_event_features)
    test_features_case_scaled = scaler_case.transform(test_features_case)

    # Inner cross-validation
    avg_validation_scores = {}
    inner_results_log[f"Outer Fold {outer_fold + 1}"] = []

    for hyperparams in hyperparameter_combinations:
        print(f"\nTesting Hyperparameters: {hyperparams}")
        validation_scores = []

        inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kf.split(train_features_event_scaled)):
            print(f"Inner Fold {inner_fold + 1}")
            
            # Split inner training and validation sets
            inner_train_event = train_features_event_scaled[inner_train_idx]
            inner_train_case = train_features_case_scaled[inner_train_idx]
            inner_val_event = train_features_event_scaled[inner_val_idx]
            inner_val_case = train_features_case_scaled[inner_val_idx]
            inner_train_labels = train_labels[inner_train_idx]
            inner_val_labels = train_labels[inner_val_idx]

            # Apply SMOTE to inner training data only
            inner_train_combined = np.hstack([
                inner_train_event.reshape(inner_train_event.shape[0], -1),
                inner_train_case
            ])
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            inner_features_resampled, inner_labels_resampled = smote.fit_resample(inner_train_combined, inner_train_labels)
            num_samples_inner = inner_features_resampled.shape[0]

            # Separate resampled features back into event and case features
            inner_train_event_resampled = inner_features_resampled[:, :num_event_features * sequence_length].reshape(
                num_samples_inner, sequence_length, num_event_features)
            inner_train_case_resampled = inner_features_resampled[:, num_event_features * sequence_length:]

            # Train model with pruning
            model = build_model(hyperparams)
            model.fit([inner_train_event_resampled, inner_train_case_resampled], inner_labels_resampled,
                      validation_data=([inner_val_event, inner_val_case], inner_val_labels),
                      epochs=5, batch_size=32, verbose=1,
                      callbacks=[
                          EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                          ValidationLossThreshold(threshold=0.5)
                      ])

            # Evaluate on the validation set (no resampling)
            val_loss = model.evaluate([inner_val_event, inner_val_case], inner_val_labels, verbose=1)[0]
            validation_scores.append(val_loss)

        # Calculate average validation loss for the hyperparameters
        avg_val_loss = np.mean(validation_scores)
        avg_validation_scores[hyperparams] = avg_val_loss

        # Save detailed results for the current hyperparameter combination
        inner_results_log[f"Outer Fold {outer_fold + 1}"].append({
            "Hyperparameters": hyperparams,
            "Validation Loss (Per Fold)": validation_scores,
            "Average Validation Loss": avg_val_loss
        })

    # Save inner loop results to a JSON file for the current outer fold
    with open(inner_results_path, "w") as f:
        json.dump(inner_results_log[f"Outer Fold {outer_fold + 1}"], f, indent=4)

    # Save best hyperparameters and evaluate on test set
    best_hyperparams = min(avg_validation_scores, key=avg_validation_scores.get)
    print(f"Best Hyperparameters for Outer Fold {outer_fold + 1}: {best_hyperparams}")
    final_model = build_model(best_hyperparams)
    final_model.fit([train_features_event_scaled, train_features_case_scaled], train_labels,
                    epochs=10, batch_size=32, verbose=1)
    test_loss, test_accuracy = final_model.evaluate([test_features_event_scaled, test_features_case_scaled], test_labels, verbose=1)

    # Save results
    final_model.save(model_checkpoint_path)
    results_log.append({"Outer Fold": outer_fold + 1, "Hyperparameters": best_hyperparams, "Test Loss": test_loss, "Test Accuracy": test_accuracy})
    with open(intermediate_results_path, "w") as f:
        json.dump(results_log, f, indent=4)

# Save final results
df_final_results = pd.DataFrame(results_log)
df_final_results.to_csv("results/final_results.csv", index=False)

# Save all inner results for all outer folds
with open("results/all_inner_results.json", "w") as f:
    json.dump(inner_results_log, f, indent=4)

print("Final results and inner cross-validation results saved.")
