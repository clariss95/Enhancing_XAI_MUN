import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.utils import plot_model
import joblib



# Load the processed data from .npy files
event_features = np.load('event_features.npy')
case_features = np.load('case_features.npy')
labels = np.load('labels.npy')
print(event_features[0], "case: ", case_features[0])

print("Data loaded from .npy files")
print(event_features.shape, case_features.shape, labels.shape)

# Count the occurrences of each class (0 and 1)
label_counts = Counter(labels)

# Print the counts for each class
print("Class distribution in labels:")
print(label_counts)

# Define the sequence length
sequence_length = 2

# Calculate the number of event features and case features
num_event_features = event_features.shape[2]
num_case_features = case_features.shape[1]

# Training, validation, and testing splits
train_idx = int(.6 * len(event_features))
val_idx = int(.8 * len(event_features))

train_features_event = event_features[:train_idx]
val_features_event = event_features[train_idx:val_idx]
test_features_event = event_features[val_idx:]

train_features_case = case_features[:train_idx]
val_features_case = case_features[train_idx:val_idx]
test_features_case = case_features[val_idx:]

train_labels = labels[:train_idx]
val_labels = labels[train_idx:val_idx]
test_labels = labels[val_idx:]

# Check original class distribution for train, validation, and test sets
original_train_distribution = Counter(train_labels)
original_val_distribution = Counter(val_labels)
original_test_distribution = Counter(test_labels)

print("Original class distribution in training set:", original_train_distribution)
print("Original class distribution in validation set:", original_val_distribution)
print("Original class distribution in test set:", original_test_distribution)

print(f"Number of samples in original test set: {len(test_features_event)}")
print(f"Number of samples in original test labels: {len(test_labels)}")

# Resample the training data using SMOTE
train_features_combined = np.hstack([train_features_event.reshape(train_features_event.shape[0], -1), train_features_case])
smote = SMOTE(sampling_strategy='auto')
train_features_resampled, train_labels_resampled = smote.fit_resample(train_features_combined, train_labels)

# Separate the resampled features back into event and case features
num_samples_train = train_features_resampled.shape[0]
train_features_event_resampled = train_features_resampled[:, :num_event_features * sequence_length].reshape(num_samples_train, sequence_length, num_event_features)
train_features_case_resampled = train_features_resampled[:, num_event_features * sequence_length:]

# Standardize the features
scaler_event = StandardScaler()
scaler_case = StandardScaler()

train_features_event_resampled = scaler_event.fit_transform(train_features_event_resampled.reshape(-1, num_event_features)).reshape(num_samples_train, sequence_length, num_event_features)
train_features_case_resampled = scaler_case.fit_transform(train_features_case_resampled)

val_features_event = scaler_event.transform(val_features_event.reshape(-1, num_event_features)).reshape(val_features_event.shape[0], sequence_length, num_event_features)
val_features_case = scaler_case.transform(val_features_case)

test_features_event = scaler_event.transform(test_features_event.reshape(-1, num_event_features)).reshape(test_features_event.shape[0], sequence_length, num_event_features)
test_features_case = scaler_case.transform(test_features_case)

# Save the scalers to files
joblib.dump(scaler_event, 'scaler_event.pkl')
joblib.dump(scaler_case, 'scaler_case.pkl')

# Create TensorFlow datasets
train_dataset_resampled = tf.data.Dataset.from_tensor_slices(((train_features_event_resampled, train_features_case_resampled), train_labels_resampled)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices(((val_features_event, val_features_case), val_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices(((test_features_event, test_features_case), test_labels)).batch(32)

# Verify the shape of the datasets
print(train_features_event_resampled.shape, train_features_case_resampled.shape, train_labels_resampled.shape)
print(val_features_event.shape, val_features_case.shape, val_labels.shape)
print(test_features_event.shape, test_features_case.shape, test_labels.shape)


#  # Define the LSTM model
event_feature_shape = (sequence_length, num_event_features)
case_feature_shape = (num_case_features,)

event_input = Input(shape=event_feature_shape, name='event_input')
lstm_out = LSTM(64, return_sequences=True)(event_input)
lstm_out = Dropout(0.5)(lstm_out)
lstm_out = LSTM(32, return_sequences=False)(lstm_out)
lstm_out = Dropout(0.5)(lstm_out)
lstm_out = BatchNormalization()(lstm_out)

case_input = Input(shape=case_feature_shape, name='case_input')
combined = Concatenate()([lstm_out, case_input])
output = Dense(1, activation='sigmoid')(combined)

model_enhanced = Model(inputs=[event_input, case_input], outputs=output)
model_enhanced.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# Save the model summary to a text file
with open('model_summary2.txt', 'w') as f:
    model_enhanced.summary(print_fn=lambda x: f.write(x + '\n'))


# # # Define early stopping and learning rate scheduler callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
# model_checkpoint = ModelCheckpoint('model_enhanced_prefix2_isdeceasedfix.h5', save_best_only=True, monitor='val_loss', mode='min')

# # # Train the model with early stopping, learning rate scheduler, and class weights
# # # class_weights = {0: 1.0, 1: 5.0}  # Adjusted class weights
# history_enhanced = model_enhanced.fit(
#      train_dataset_resampled, 
#      validation_data=val_dataset, 
#           epochs=50,  
#  # #     class_weight=class_weights,
#      callbacks=[early_stopping, reduce_lr, model_checkpoint]
#  )

# Load the best model
model_enhanced = load_model('model_enhanced_prefix2_isdeceasedfix.h5')



# Evaluate the model on the test set
test_loss, test_accuracy = model_enhanced.evaluate(test_dataset)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Assuming test_dataset and test_labels are already defined and the model is trained
test_probs = model_enhanced.predict(test_dataset).ravel()
test_labels_flat = test_labels.ravel()

# Compute ROC curve
fpr_test, tpr_test, thresholds_test = roc_curve(test_labels_flat, test_probs)

# Define your cost function
def cost_function(conf_matrix):
    TN, FP, FN, TP = conf_matrix.ravel()
    return FP * 1 + FN * 10  # Adjust the weights according to your cost considerations

# Initialize lists to store results
results = []

for threshold in thresholds_test:
    test_preds = (test_probs >= threshold).astype(int)
    
    precision = precision_score(test_labels_flat, test_preds, zero_division=0)
    recall = recall_score(test_labels_flat, test_preds, zero_division=0)
    f1 = f1_score(test_labels_flat, test_preds, zero_division=0)
    accuracy = accuracy_score(test_labels_flat, test_preds)
    conf_matrix = confusion_matrix(test_labels_flat, test_preds)
    cost = cost_function(conf_matrix)
    
    results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'cost': cost
    })

# Convert results to a DataFrame for easier visualization
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
results_df.to_csv('threshold_results.csv', index=False)

# Find the threshold with the highest accuracy
best_accuracy_threshold = results_df.loc[results_df['accuracy'].idxmax()]

# Find the threshold where sensitivity equals specificity
sensitivity_equals_specificity_threshold = results_df.loc[(results_df['recall'] - results_df['accuracy']).abs().idxmin()]

# Find the threshold with the lowest cost
best_cost_threshold = results_df.loc[results_df['cost'].idxmin()]

# Adding the 0.5 threshold explicitly
threshold_0_5 = results_df.loc[np.isclose(results_df['threshold'], 0.50075585)]

# Print the optimal thresholds
print("Best Accuracy Threshold:\n", best_accuracy_threshold)
print("Threshold where Sensitivity equals Specificity:\n", sensitivity_equals_specificity_threshold)
print("Threshold with Best Cost:\n", best_cost_threshold)
print("Threshold 0.5:\n", threshold_0_5)

# Compute ROC AUC
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_test, tpr_test, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
