import numpy as np
import pandas as pd
import json
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
import keras_tuner as kt
import joblib

# Load the processed data from .npy files
event_features = np.load('event_features.npy')
case_features = np.load('case_features.npy')
labels = np.load('labels.npy')

print("Data loaded from .npy files")
print(event_features.shape, case_features.shape, labels.shape)

# Count the occurrences of each class (0 and 1)
label_counts = Counter(labels)
print("Class distribution in labels:", label_counts)

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

# Create TensorFlow datasets
train_dataset_resampled = tf.data.Dataset.from_tensor_slices(((train_features_event_resampled, train_features_case_resampled), train_labels_resampled)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices(((val_features_event, val_features_case), val_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices(((test_features_event, test_features_case), test_labels)).batch(32)


def build_model(hp):
    lstm_units1 = hp.Choice('lstm_units1', values=[32, 64, 128], default=64)  # Baseline default: 64
    lstm_units2 = hp.Choice('lstm_units2', values=[16, 32, 64], default=32)  # Baseline default: 32
    dropout_rate = hp.Choice('dropout_rate', values=[0.2, 0.3, 0.5], default=0.5)  # Baseline default: 0.5
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4], default=1e-4)  # Baseline default: 0.0001
    
    event_input = Input(shape=(sequence_length, num_event_features), name='event_input')
    lstm_out = LSTM(lstm_units1, return_sequences=True)(event_input)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    lstm_out = LSTM(lstm_units2, return_sequences=False)(lstm_out)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)

    case_input = Input(shape=(num_case_features,), name='case_input')
    combined = Concatenate()([lstm_out, case_input])
    output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[event_input, case_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Set up the Keras Tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=20,
    factor=3,
    directory='hyperparameter_tuning',
    project_name='lstm_model_tuning_with_baseline'
)


# Perform the search
tuner.search(
    train_dataset_resampled,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
)


# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
print(f"LSTM Units 1: {best_hps.get('lstm_units1')}")
print(f"LSTM Units 2: {best_hps.get('lstm_units2')}")
print(f"Dropout Rate: {best_hps.get('dropout_rate')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")

# Train the best model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    train_dataset_resampled,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
               ModelCheckpoint('best_model_with_baseline.h5', save_best_only=True, monitor='val_loss')]
)

# Evaluate the best model on the test set
test_loss, test_accuracy = best_model.evaluate(test_dataset)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
